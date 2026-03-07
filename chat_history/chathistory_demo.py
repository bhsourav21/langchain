import os
import json
from tkinter import Variable
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_community.chat.message_histories.in_memory import ChatMessageHistory
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
# from langchain_core.runnable.history import RunnableWithMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

st.title("Agile Guide App")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a agile coach. Answer any questions related to the agile process"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}")
    ]
)

user_input = st.text_input("Enter the question: ")

chain = prompt | llm
history_for_chain = ChatMessageHistory()
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history_for_chain,
    input_messages_key="user_input",
    # hist_message_key="chat_history",
    history_messages_key="chat_history",
)


# while True:
if user_input:
    response = (
        chain_with_history.invoke(
        {"user_input": user_input},
        config=
            {
                "configurable":
                    {
                        "session_id": "abc123"
                    }
            }
        )
    )
    print(response.content)

