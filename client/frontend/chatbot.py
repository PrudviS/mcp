import streamlit as st
import httpx
from typing import Dict, Any
import json

class Chatbot:
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.current_tool_call = {"name": None, "args": None}
        self.messages = st.session_state["messages"]

    def display_message(self, message: Dict[str, Any]):
        # user message
        if message["role"] == "user" and type(message["parts"][0]["text"]) == str and message["parts"][0]["functionCall"] is None and message["parts"][0]["functionResponse"] is None:
            st.chat_message("user").markdown(message["parts"][0]["text"])

        # tool call
        if message["role"] == "model" and message["parts"][0]["functionCall"] is not None and message["parts"][0]["functionResponse"] is None:
                self.current_tool_call = {
                        "name": message["parts"][0]["functionCall"]["name"],
                        "args": message["parts"][0]["functionCall"]["args"]
                }


        # tool result
        if message["role"] == "tool" and message["parts"][0]["functionResponse"] is not None:
             for content in message["parts"]:
                with st.chat_message("assistant"):
                    st.write(f"Called tool: {self.current_tool_call['name']}")
                    st.json(
                        {
                            "name": self.current_tool_call["name"],
                            "args": self.current_tool_call["args"],
                            "content": content["functionResponse"]["response"]["result"][0]["text"],
                        },
                        expanded=False,
                    )

        # ai message
        if message["role"] == "model" and type(message["parts"][0]["text"]) == str and message["parts"][0]["functionCall"] is None and message["parts"][0]["functionResponse"] is None:
            st.chat_message("assistant").markdown(message["parts"][0]["text"])


    async def get_tools(self):
        async with httpx.AsyncClient(timeout=30, verify=False) as client:
            response = await client.get(
                f"{self.api_url}/tools",
                headers={"Content-Type": "application/json"},
            )
            return response.json()

    async def render(self):
        st.title("MCP Host Application")
        with st.sidebar:
            st.subheader("Settings")
            st.write("API URL: ", self.api_url)
            result = await self.get_tools()
            st.subheader("Tools")
            st.write([tool["name"] for tool in result["tools"]])

        for message in self.messages:
            self.display_message(message)

        query = st.chat_input("Enter your query here")
        if query:
            async with httpx.AsyncClient(timeout=120, verify=False) as client:
                try:
                    response = await client.post(
                        f"{self.api_url}/query",
                        json={"query": query},
                        headers={"Content-Type": "application/json"},
                    )
                    if response.status_code == 200:
                        messages = response.json()
                        st.session_state["messages"] = messages["messages"]
                        for message in st.session_state["messages"]:
                            self.display_message(message)
                except httpx.TimeoutException as e:
                    st.error("Error processing query: Timeout Exception")
                except Exception as e:
                    st.error(f"Frontend: Error processing query: {str(e)}")