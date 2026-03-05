import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

st.title("LangChain prompt template Demo")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = PromptTemplate(
    input_variables=["country1", "country2", "number_of_sentences"],
    template="""What is the capital of {country1} and {country2}?
    Write {number_of_sentences} sentences on both mentioning why they are famous.
    Avoid giving information about fictional places. If the country
    is fictional, or non-existance, answer: I don't know."""
)

country1 = st.text_input("Enter the first country: ")
country2 = st.text_input("Enter the second country: ")
number_of_sentences = st.number_input("Enter the number of sentences: ", min_value=1, max_value=10, value=1)

if country1 and country2:
    prompt_value = prompt.format(country1=country1, country2=country2, number_of_sentences=number_of_sentences)
    response = llm.invoke(prompt_value)
    st.write(response.content)
