import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm1 = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm2 = ChatOpenAI(model="gpt-4o", temperature=0)

prompt_title = PromptTemplate(
    input_variables=["topic"],
    template="""You are an experienced speech writer.
    You need to craft an impactful title for a speech on the following topic: {topic}
    Answer exactly with one title.
    """
)

prompt_speech = PromptTemplate(
    input_variables=["title"],
    template="""You need to write a powerful speech of 350 words for the following title:
    {title}
    """
)

first_chain = prompt_title | llm1 | StrOutputParser() | (lambda title: (st.write(title), title)[1])
second_chain = prompt_speech | llm2
final_chain = first_chain | second_chain

st.title("Speech Generator")
topic = st.text_input("Enter the topic:")

if topic:
    response = final_chain.invoke({
        "topic": topic
    })
    
    st.write(response.content)