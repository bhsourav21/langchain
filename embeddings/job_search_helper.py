import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings()

document = TextLoader("job_listingsed?
.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunks = text_splitter.split_documents(document)
db = Chroma.from_documents(chunks, embeddings)

text = input("Enter the query:")
embedding_vector=embeddings.embed_query(text)

docs = db.similarity_search_by_vector(embedding_vector)

for doc in docs:
    print(doc)
