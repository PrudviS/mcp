import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from langchain_chroma import Chroma

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
embeddings_persistent_directory= os.path.join(current_directory, "vectors\\")
persist_directory = embeddings_persistent_directory
client = chromadb.PersistentClient(path=persist_directory)



embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

collectionName="aks_networking"


vector_store = Chroma(
    collection_name=collectionName,
    embedding_function=embeddings,
    persist_directory=f"{embeddings_persistent_directory}\\chroma_langchain_db",
)


def search_with_query(query, vector_store, embeddings, k=3):
    query_embedding = embeddings.embed_query(query)
    results = vector_store.similarity_search_by_vector(query_embedding, k=k)

    return results


results = vector_store.similarity_search_by_vector(
    embedding=embeddings.embed_query("what is azure CNI?"), k=3
)
for doc in results:
    print(f"* {doc.page_content} [{doc.metadata}]")
