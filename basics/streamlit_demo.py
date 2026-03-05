import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.globals import set_debug
from dotenv import load_dotenv

load_dotenv()

# set_debug(True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

st.title("LangChain OpenAI Demo")
st.write("This is a simple demo of the LangChain OpenAI API.")

question = st.text_input("Enter a question: ")
if question:
    response = llm.invoke(question)
    st.write(response.content)

