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

prompt_product_name = PromptTemplate(
    input_variables=["product_name", "product_features"],
    template="""You are an experienced marketing specialist.
    Create a catchy subject line for a marketing email promoting the following product: {product_name}.
    Highlight these featues: {product_features}.
    Respond with only the subject line.
    """
)

prompt_product_feature = PromptTemplate(
    input_variables=["product_name", "subject_line", "target_audience"],
    template="""
    Write a marketing email of 100 words for the
    product: {product_name}. Use the subject line: {subject_line}.
    Tailor the message for the following target audience: {target_audience}.
    Format the output as a json object with the three keys: 'subject', 
    'audience', 'email_body' and fill them with respective values.
    """
)

first_chain = prompt_product_name | llm | StrOutputParser()
second_chain = prompt_product_feature | llm | JsonOutputParser()
final_chain = first_chain | (lambda subject_line: {"product_name": product_name, "subject_line": subject_line, "target_audience": target_audience}) | second_chain

st.title("Marketing email generator")
product_name = st.text_input("Enter the product name:")
product_features = st.text_input("Enter the product features (comma separated):")
target_audience = st.text_input("Enter the target audience:")

if product_name and product_features and target_audience:
    response = final_chain.invoke({
        "product_name": product_name,
        "product_features": product_features,
        "target_audience": target_audience
    })
    
    st.write(response)