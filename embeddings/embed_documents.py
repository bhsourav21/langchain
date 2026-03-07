import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings()

response = embeddings.embed_documents(
    [
        "I love playing video games",
        "I am going to the movie",
        "I love coding",
        "Hello world!"
    ]
)
print(f"embeddings length:{len(response)}")
print(f"embedding for first sentence:{response[0]}")