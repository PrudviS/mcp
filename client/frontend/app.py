import asyncio
import sys
import streamlit as st
from chatbot import Chatbot
from dotenv import load_dotenv
import os

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.append(utils_path)

try:
    from logger import logger
except ImportError as e:
    print("ImportError:", e)

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'), override=True)

async def main():
    if "server_connected" not in st.session_state:
        st.session_state["server_connected"] = False

    if "tools" not in st.session_state:
        st.session_state["tools"] = []

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    API_URL = os.getenv("MCP_SERVER_HOST_URL")
    if not API_URL:
        st.error("MCP SERVER HOST URL not found. Please add it to your .env file.")

    st.set_page_config(page_title="MCP Host", page_icon=":shark:",layout="wide")

    chatbot = Chatbot(API_URL)
    await chatbot.render()


if __name__ == "__main__":
    asyncio.run(main())