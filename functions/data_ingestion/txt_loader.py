from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Load data from a .txt file
def txt_loader(document_path):
    document = TextLoader(document_path).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents=document)
    return chunks

# Retriever using FAISS
def retriever(chunks):
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")

    # Convert text chunks into LangChain Document objects
    documents = [Document(page_content=chunk.page_content) for chunk in chunks]

    # Create FAISS vector store from documents
    vector_store = FAISS.from_documents(documents, embedding)

    # Return the retriever directly
    return vector_store.as_retriever()
