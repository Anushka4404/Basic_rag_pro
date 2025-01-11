import streamlit as st
import langchain
import langchain_community
from langchain.embeddings.google_palm import GooglePalmEmbeddings
#from langchain.google_genai import GoogleGenerativeAIEmbeddings  # Correct module for Google GenAI Embeddings
from langchain_community.document_loaders import PyPDFLoader  # Corrected import path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
#from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
import os
import pinecone
import pypdf
load_dotenv()

PINECONE_INDEX_NAME = "firstproject"

os.environ['PINECONE_API_KEY']="pcsk_ec4Cx_9cmU8Zcj2LqTeEpyjMnaEFdSU8CMaCxqZq3aa8oMjh5Fzvq6LrsUgtHagTQkRSA"

st.title("YOLOV Q&A APPLICATION")

# Example: Load a PDF, split text, and store in Pinecone
loader = PyPDFLoader("./GEMINI_RAG_DEMO/yolov.pdf")  # Replace with your file path
data = loader.load()
# loader


# Split text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

# print("Total no. of docs is:",len(docs))


os.environ["GOOGLE_API_KEY"] = ("AIzaSyAUGvXwP_fkDjWPI7uL8MLrX_BAdEIHOec")


embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
docsearch = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)

docsearch = Pinecone.from_existing_index(PINECONE_INDEX_NAME,embeddings)
retriever = docsearch.as_retriever(search_type="similarity",search_kwargs={"k":10})

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",temperature=0.3,max_tokens=500)

query = st.chat_input("Say something: ")
prompt = query

system_prompt = (
    "You are an assistant for question-answering tasks."
    "Use the following pieces of retrieved context to answer"
    "the question. If you dont know the answer, say that you"
    "don't know. Use three sentences maximum and keep the "
    "answer concise"
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human","{input}"),
    ]
)
if query:
    question_answer_chain = create_stuff_documents_chain(llm,prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": query})
    
    st.write(response["answer"])