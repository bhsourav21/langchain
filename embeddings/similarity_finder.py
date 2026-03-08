import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import numpy as np

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings()

text1 =input("Enter a text1: ")
text2 =input("Enter a text2: ")
response1 = llm.embed_query(text1)
response2 = llm.embed_query(text2)
similarity_score = np.dot(response1, response2)

print(f"similarity_score (higher the number, its more similar):{similarity_score}")