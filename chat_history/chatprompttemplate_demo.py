import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.title("Agile Guide App")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a agile coach. Answer any questions related to the agile process"),
        ("human", "{input}")
    ]
)

input = st.text_input("Enter the question: ")

chain = prompt | llm

if input:
    response = chain.invoke({"input": input})
    st.write(response.content)