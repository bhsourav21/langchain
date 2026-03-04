import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

st.title("Interview Tips Generator")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = PromptTemplate(
    input_variables=["company_name", "position_title", "strengths", "weaknesses"],
    template="""You are a helpful assistant that generates interview tips 
    for a given company name, position title, interviewee's strengths, and weaknesses.
    Company name: {company_name}
    Position title: {position_title}
    Strengths: {strengths}
    Weaknesses: {weaknesses}
    Generate 5 interview tips for the given company name, position title, strengths, and weaknesses.

    Return the response as a valid JSON object with exactly the following keys:
        - "tips": [list of tips]
    Do not include any explanations, markdown, or text outside the JSON.
    """
)

company_name = st.text_input("Enter the company name: ", placeholder="e.g. Google, Apple, Microsoft")
position_title = st.text_input("Enter the position title: ", placeholder="e.g. Software Engineer, Product Manager, Data Scientist")
strengths = st.text_input("Enter the interviewee's strengths: ", placeholder="e.g. Problem solving, Communication, Leadership")
weaknesses = st.text_input("Enter the interviewee's weaknesses: ", placeholder="e.g. Time management, Attention to detail, Teamwork")

if company_name and position_title and strengths and weaknesses:
    prompt_value = prompt.format(
        company_name=company_name,
        position_title=position_title,
        strengths=strengths,
        weaknesses=weaknesses,
    )
    response = llm.invoke(prompt_value)
    content = response.content.strip()

    # Strip possible markdown code fences
    if "```" in content:
        parts = content.split("```")
        if len(parts) >= 3:
            content = parts[1].strip()
        else:
            content = parts[-1].strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        st.error("Could not parse the model response as JSON. Showing raw output instead.")
        st.write(response.content)
    else:
        # Be robust to weirdly formatted keys like '\n "tips"'
        tips = data.get("tips")
        if tips is None:
            for key, value in data.items():
                if "tips" in key:
                    tips = value
                    break
        if tips is None:
            tips = []

        st.subheader("Tips")
        st.write(tips)