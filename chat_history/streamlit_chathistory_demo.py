import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a agile coach. Answer any questions related to the agile process"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}")
    ]
)

chain = prompt | llm

history_for_chain = StreamlitChatMessageHistory()

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history_for_chain,
    input_messages_key="user_input",
    history_messages_key="chat_history",
)

st.title("Agile Guide")
user_input = st.text_input("Enter the question: ")


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
    st.write(response.content)

    st.write("HISTORY")
    st.write(history_for_chain)

