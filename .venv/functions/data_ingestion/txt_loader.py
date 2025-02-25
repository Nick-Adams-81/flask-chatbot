from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# Load dat from a .txt file
def txt_loader(document_path):
    document = TextLoader(document_path).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents=document)
    return chunks

# Retriever
def retriever(chunks):
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = Chroma.from_documents(chunks, embedding)
    retriever = vector_store.as_retriever()
    return retriever