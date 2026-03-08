import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

document = PyPDFLoader("academic_research_data.pdf").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(document)
vector_store = Chroma.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are an assistant for answering questions.
        Use the provided context to respond.If the answer
        isn't clear, acknowledge that you don't know.
        Limit your response to three concise sentences.
        {context}
        
        """),
        ("human", "{input}")
    ]
)

qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

print("Chat with Document")
question = input("Your question: ")

if question:
    response = rag_chain.invoke({"input": question})
    print(response['answer'])

