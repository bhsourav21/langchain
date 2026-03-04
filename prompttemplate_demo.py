import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()

st.title("LangChain prompt template Demo")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = PromptTemplate(
    input_variables=["country1", "country2"],
    template="""What is the capital of {country1} and {country2}?"""
)

country1 = st.text_input("Enter the first country: ")
country2 = st.text_input("Enter the second country: ")

if country1 and country2:
    prompt_value = prompt.format(country1=country1, country2=country2)
    response = llm.invoke(prompt_value)
    st.write(response.content)
