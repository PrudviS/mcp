from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader


from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'), override=True)


embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
embeddings_persistent_directory= os.path.join(current_directory, "vectors\\")

os.makedirs(os.path.dirname(embeddings_persistent_directory), exist_ok=True)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20,
    length_function=len
)

collectionName="aks_networking"

vector_store = Chroma(
    collection_name=collectionName,
    embedding_function=embeddings,
    persist_directory=f"{embeddings_persistent_directory}\\chroma_langchain_db"
)

def upload_website_to_collection(url:str):
    loader = WebBaseLoader(url)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    docs_split = text_splitter.split_documents(docs)
    for doc in docs_split:
        doc.metadata = {"source_url": url}

    vector_store.add_documents(docs_split)
    return f"Successfully uploaded {len(docs_split)} documents to collection {collectionName} from {url}"

chroma_db_file_path = f"{embeddings_persistent_directory}\\chroma_langchain_db"

urls=["https://learn.microsoft.com/en-us/azure/aks/concepts-network-isolated","https://learn.microsoft.com/en-us/azure/aks/core-aks-concepts","https://learn.microsoft.com/en-us/azure/aks/free-standard-pricing-tiers","https://learn.microsoft.com/en-us/azure/aks/upgrade","https://learn.microsoft.com/en-us/azure/backup/azure-kubernetes-service-backup-overview?toc=%2Fazure%2Faks%2Ftoc.json&bc=%2Fazure%2Faks%2Fbreadcrumb%2Ftoc.json","https://learn.microsoft.com/en-us/azure/aks/node-resource-reservations","https://learn.microsoft.com/en-us/azure/aks/aks-communication-manager"]

if os.path.exists(chroma_db_file_path):
    for url in urls:
        response = upload_website_to_collection(url)
        print(response)