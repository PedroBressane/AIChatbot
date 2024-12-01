import os
from decouple import config

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

os.environ['GROQ_API_KEY'] = config('GROQ_API_KEY')

class AIBot:

    def __init__(self):
        self.__chat = ChatGroq(model='llama-3.1-70b-versatile')
        self.__retriever = self.__build_retriever()

    def __build_retriever(self):
        persist_directory = '/app/chroma_data'
        embedding = HuggingFaceEmbeddings()

        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding,
        )
        return vector_store.as_retriever(
            search_kwargs={'k', 30},
        )

    def __build_messages(self, history_messages, question):
        messages = []
        for message in history_messages:
            message_class = HumanMessage if message.get('fromMe') else AIMessage
            messages.append(message_class(content = message.get('body')))
        messages.append(HumanMessage(content=question))
        return messages

    def invoke(self, history_messages, question):
        system_template = """
        You are Julia, a virtual assistant from the DataGoo system, who helps with questions about the system and simple errors.
        If you are unable to resolve the issue based on the context of the system's files, inform the user that you will forward the issue 
        to one of the technical operators for resolution, or direct them if they request to speak with a technical operator.
        <context>
        {context}
        </context>
        """

        docs = self.__retriever.invoke(question)
        question_answering_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    'system',
                    system_template,
                ),
                MessagesPlaceholder(variable_name = 'messages'),
            ]
        )
        document_chain = create_stuff_documents_chain(self.__chat, question_answering_prompt)
        response = document_chain.invoke(
            {
                'context': docs,
                'messages': self.__build_messages(history_messages,question),
            }
        )
        return response
