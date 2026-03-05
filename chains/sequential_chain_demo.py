import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt_title = PromptTemplate(
    input_variables=["topic"],
    template="""You are an experienced speech writer.
    You need to craft an impactful title for a speech on the following topic: {topic}
    Answer exactly with one title.
    """
)

prompt_speech = PromptTemplate(
    input_variables=["title", "emotion"],
    template="""You need to write a powerful {emotion} speech of 350 words for the following title:
    {title}
    Format the output with two keys: 'title', 'speech' and fill them with the respective values
    """
)

first_chain = prompt_title | llm | StrOutputParser() | (lambda title: (st.write(title), title)[1])
second_chain = prompt_speech | llm | JsonOutputParser()
final_chain = first_chain | (lambda title: {"title": title, "emotion": emotion}) | second_chain

st.title("Speech Generator")
topic = st.text_input("Enter the topic:")
emotion = st.text_input("Enter the emotion:")

if topic and emotion:
    response = final_chain.invoke({
        "topic": topic,
        "emotion": emotion
    })
    
    # st.write(response.content)
    st.write(response)