import os
import json
from tkinter import Variable
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat.messagea_history.in_memory import ChatMessageHistory
from langchain_core.runnable.history import RunnableWithMessageHistory

load_dotenv()

st.title("Agile Guide App")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a agile coach. Answer any questions related to the agile process"),
        MessagesPlaceholder(variable_name="chat_history")
        ("human", "{input}")
    ]
)

input = st.text_input("Enter the question: ")

chain = prompt | llm
history_for_chain = ChatMessageHistory()
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history_for_chain,
    input_message_key="input",
    hist_message_key="chat_history",
)


if input:
    response = chain.invoke({"input": input})
    st.write(response.content)