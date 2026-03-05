import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt_outline  = PromptTemplate(
    input_variables=["topic"],
    template="""You are a professional blogger.
    Create an outline for a blog post on the following topic: {topic}
    The outline should include:
        - Introduction
        - 3 main points with subpoints
        - Conclusion
    """
)

prompt_intro = PromptTemplate(
    input_variables=["outline"],
    template="""You are a professional blogger.
    Write an engaging introduction paragraph based on the following
    outline:{outline}
    The introductionshould hook the reader and provide a brief
    overview of the topic.
    """
)

first_chain = prompt_outline | llm | StrOutputParser() 
second_chain = prompt_intro | llm
final_chain = first_chain | second_chain

st.title("Blog post generator")
topic = st.text_input("Enter the topic:")

if topic:
    response = final_chain.invoke({
        "topic": topic
    })
    
    st.write(response.content)






