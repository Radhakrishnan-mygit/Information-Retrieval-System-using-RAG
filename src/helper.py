import os
from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from tempfile import NamedTemporaryFile

#extracting text from pdf files


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_chunks(text_data):
  text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=50)
  chunks = text_splitter.split_text(text_data)
  return chunks

def get_vector_Store(chunks):
   chunk_texts = [chunk.page_content for chunk in chunks if isinstance(chunk, Document)]
    
    # Initialize the embedding model
   embedding = (OllamaEmbeddings(model='all-minilm'),)
    
    # Generate document embeddings
   #doc_embedding = embedding.embed_documents(chunk_texts)
   vector_store = FAISS.from_texts(chunk_texts,embedding)
    
   return vector_store



# def get_vector_Store(chunks):
#    #chunk_texts = [chunk for chunk in chunks if isinstance(chunk, str) and chunk.strip()] 
#    embedding=OllamaEmbeddings(model='all-minilm')
#    doc_embedding=embedding.embed_documents(chunks)
#    #vector_store=FAISS.from_documents(chunk_texts,doc_embedding)
#    vector_store = FAISS.from_texts(chunks, embedding=doc_embedding)
#    return vector_store

def get_conversational_chain(vector_store):
   llm=Ollama(model="llama3")
   memory=ConversationBufferMemory(memory_key="chat_history",return_messages=True)
   conversation=ConversationalRetrievalChain.from_llm(llm=llm,retriever=vector_store.as_retriever(),memory=memory)
   return conversation