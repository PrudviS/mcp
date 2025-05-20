from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import httpx
import json
import os
from bs4 import BeautifulSoup
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from langchain_chroma import Chroma
import asyncio
from playwright.async_api import async_playwright


load_dotenv(os.path.join(os.path.dirname(__file__), '.env'), override=True)

mcp = FastMCP("docs")

SERPER_URL="https://google.serper.dev/search"

docs_urls = {
    "azure_aks": "learn.microsoft.com/en-us/azure/aks",
    "blueprism": "docs.blueprism.com/en-US",
}

async def search_web(query: str) -> dict | None:
    payload = json.dumps({"q": query, "num": 1})

    headers = {
        "X-API-KEY": os.getenv("SERPER_API_KEY"),
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                SERPER_URL, headers=headers, data=payload, timeout=30
            )
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException:
            return {"organic": []}

async def fetch_url(url: str):
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url)
            await page.wait_for_selector("body")
            await asyncio.sleep(30)
            page_content = await page.content()
            await browser.close()
            soup = BeautifulSoup(page_content, "html.parser")
            text = soup.get_text()
            return text

    except Exception as e:
        return f"An error occurred: {str(e)}"

@mcp.tool()
async def get_docs(query: str, library: str):
  """
  Search the latest docs for a given query and library.
  Supports aks and blueprism.

  Args:
    query: The query to search for (e.g. "saml authentication in blueprism Hub 5.1")
    library: The library to search in (e.g. "blueprism")

  Returns:
    Text from the docs
  """
  if library not in docs_urls:
    raise ValueError(f"Library {library} not supported by this tool")

  query = f"site:{docs_urls[library]} {query}"
  results = await search_web(query)
  if len(results["organic"]) == 0:
    return "No results found"

  text = ""
  for result in results["organic"]:
    text += await fetch_url(result["link"])
  return text


embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
collectionName="aks_networking"
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
embeddings_persistent_directory= os.path.join(current_directory, "vectorstore\\vectors\\")
vector_store = Chroma(
    collection_name=collectionName,
    embedding_function=embeddings,
    persist_directory=f"{embeddings_persistent_directory}\\chroma_langchain_db"
)



@mcp.tool()
async def search_vector_store(query: str,):
  """
  Search the internal vector store knowledge base to answer the user's question.
  The vector store currently has knowledge about Azure AKS Networking.Please use other available tools to answer questions not related to Azure AKS networking.
  If no answer is found in the vector store about Azure AKS Networking, simply respond to the user with the answer "Not found in knowledge base".
  Please do not answer the question using your own knowledge about the topic

  Args:
    query: The query to search for (e.g. "what is Azure CNI?")

  Returns:
    top 3 similarity vector search results from the vector store knowledge base
  """
  query_embedding = embeddings.embed_query(query)
  results = vector_store.similarity_search_by_vector(query_embedding, k=3)
  for doc in results:
    return(f"* {doc.page_content} [{doc.metadata}]")


if __name__ == "__main__":
     mcp.run(transport="stdio")