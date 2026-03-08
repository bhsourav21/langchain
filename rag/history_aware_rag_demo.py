import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

document = TextLoader("product-data.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(document)
vector_store = Chroma.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are an assistant for answering questions.
        Use the provided context to respond.If the answer
        isn't clear, acknowledge that you don't know.
        Limit your response to three concise sentences.
        {context}
        
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)

history_for_chain = StreamlitChatMessageHistory()

create_history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt)
qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(create_history_aware_retriever, qa_chain)

chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history",
)

st.write("Chat with Document")
question = st.text_input("Your question: ")

if question:
    response = (
        chain_with_history.invoke(
            {"input": question},
            {"configurable":{"session_id": "abc123"}}
        )
    )
    st.write(response['answer'])
