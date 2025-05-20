from typing import Optional, List
from contextlib import AsyncExitStack
import traceback
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from datetime import datetime
import json
import os
import sys
import google.genai as genai
from google.genai import types
from google.genai.types import Tool, FunctionDeclaration

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.append(utils_path)

try:
    from logger import logger
except ImportError as e:
    print("ImportError:", e)

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'), override=True)

def read_config_json(config_path: str):

        if not config_path:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, "mcp_servers_config.json")
            print(f"⚠️  mcp servers config path is not provided. Falling back to: {config_path}")

        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Failed to read config file at '{config_path}': {e}")
            sys.exit(1)

class MCPClient:
    def __init__(self):
        self.sessions: Optional[List[ClientSession]] = []
        self.exit_stack = AsyncExitStack()
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found. Please add it to your .env file.")
        self.llm = genai.Client(api_key=gemini_api_key)
        self.tools = []
        self.messages = []
        self.message_logs=[]
        self.logger = logger
        self.sessions = []
        self.stdios = []
        self.writes = []


    async def connect_to_server(self, mcp_servers_config_path: str):
        try:
            config = read_config_json(mcp_servers_config_path)
            mcp_servers = config.get("mcpServers", {})
            if not mcp_servers:
                 print("❌ No MCP servers found in the configuration.")
                 return False

            for server_name, server_info in mcp_servers.items():
                print(f"\n🔗 Connecting to MCP Server: {server_name}...")

                server_params = StdioServerParameters(
                    command=server_info["command"],
                    args=server_info["args"]
                )

                stdio_transport = await self.exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                self.stdio, self.write = stdio_transport
                session = await self.exit_stack.enter_async_context(
                    ClientSession(self.stdio, self.write)
                )

                self.sessions.append(session)

                await self.sessions[-1].initialize()

                self.logger.info(f"Connected to MCP server: {server_name}")

            mcp_tools = await self.get_mcp_tools()

            for tool in mcp_tools:
                self.tools.append(tool)

            self.tools = convert_mcp_tools_to_gemini(self.tools)

            tools = [
                    {
                        "name": tool.function_declarations[0].name,
                        "description": tool.function_declarations[0].description,
                        #"input_schema": tool.function_declarations[0].parameters.json_schema.properties
                    }
                    for tool in self.tools
                ]

            self.logger.info(
                f"Available tools: {[tool for tool in tools]}"
            )

            return True

        except Exception as e:
            self.logger.error(f"Error connecting to MCP server: {e}")
            traceback.print_exc()
            raise

    # get mcp tool list
    async def get_mcp_tools(self):
        try:
            all_tools = []
            for _, session in enumerate(self.sessions):
                try:
                    response = await session.list_tools()
                    all_tools.extend(response.tools)
                except Exception as e:
                    self.logger.error(f"Error getting tools from session: {e}")
                    continue
            return all_tools
        except Exception as e:
            self.logger.error(f"Error getting MCP tools: {e}")
            raise

    # process user query
    async def process_query(self, query: str):
        try:
            self.logger.info(f"Processing query: {query}")
            user_message = types.Content(
                role='user',
                parts=[types.Part.from_text(text=query)]
            )
            self.messages=[user_message]
            final_text = []

            while True:
                response = await self.call_llm()
                for candidate in response.candidates:
                    if candidate.content.parts:
                        for part in candidate.content.parts:
                             if isinstance(part, types.Part): # Check if part is a valid Gemini response unit
                                 if part.function_call:  # If Gemini suggests a function call, process it
                                    # Extract function call details
                                    function_call_part = part
                                    tool_name = function_call_part.function_call.name
                                    tool_args = function_call_part.function_call.args
                                    self.messages.append(candidate.content)

                                    print(f"\n[Gemini requested tool call: {tool_name} with args {tool_args}]")

                                    # Execute the tool using the MCP server
                                    result = None
                                    error = None
                                    for session in self.sessions:
                                        try:
                                            result = await session.call_tool(tool_name, tool_args)
                                            if result:
                                                break
                                        except Exception as e:
                                            error = e
                                            continue

                                    if result:
                                        function_response = {"result": result.content}
                                    else:
                                        function_response = {"error": f"Tool {tool_name} not found in any session or failed: {error}"}

                                    # Format the tool response for Gemini in a way it understands
                                    function_response_part = types.Part.from_function_response(
                                        name=tool_name,
                                        response=function_response
                                    )

                                    # Structure the tool response as a Content object for Gemini
                                    function_response_content = types.Content(
                                        role='tool',
                                        parts=[function_response_part]
                                    )

                                    self.messages.append(function_response_content)

                                    # Send tool execution results back to Gemini for processing
                                    response = await self.call_llm()

                                    # Extract final response text from Gemini after processing the tool call
                                    final_text.append(response.candidates[0].content.parts[0].text)
                                    self.messages.append(response.candidates[0].content)
                                 else:
                                     self.messages.append(response.candidates[0].content)
                    else:
                        self.messages.append(response.candidates[0].content)
                        final_text.append(part.text)
                        break

                await self.log_conversation()

                return self.messages

        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            raise

    # call llm
    async def call_llm(self):
        try:
            self.logger.info("Calling LLM")
            return self.llm.models.generate_content(
                    model="gemini-2.0-flash-lite",
                    contents=self.messages,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        tools=self.tools,
                        #max_output_tokens=1000
                    )
                )
        except Exception as e:
            self.logger.error(f"Error calling LLM: {e}")
            raise

    # cleanup
    async def cleanup(self):
        try:
            await self.exit_stack.aclose()
            self.sessions.clear()
            self.logger.info("Disconnected from all MCP servers")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            traceback.print_exc()
            raise

    async def log_conversation(self):
        os.makedirs("conversations", exist_ok=True)

        serializable_conversation = []

        for message in self.messages:
            try:
                serializable_message = {"role": message.role, "parts": []}

                # Handle both string and list content
                if isinstance(message.parts, str):
                    serializable_message["parts"] = message.parts
                elif isinstance(message.parts, list):
                    for content_item in message.parts:
                        if hasattr(content_item, "to_dict"):
                            serializable_message["parts"].append(
                                content_item.to_dict()
                            )
                        elif hasattr(content_item, "dict"):
                            serializable_message["parts"].append(content_item.dict())
                        elif hasattr(content_item, "model_dump"):
                            serializable_message["parts"].append(
                                content_item.model_dump()
                            )
                        else:
                            serializable_message["parts"].append(content_item)

                serializable_conversation.append(serializable_message)
            except Exception as e:
                self.logger.error(f"Error processing message: {str(e)}")
                self.logger.debug(f"Message content: {message}")
                raise

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath = os.path.join("conversations", f"conversation_{timestamp}.json")

        try:
            with open(filepath, "w") as f:
                json.dump(serializable_conversation, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error writing conversation to file: {str(e)}")
            self.logger.debug(f"Serializable conversation: {serializable_conversation}")
            raise

def clean_schema(schema):
    """
    Recursively removes 'title' fields from the JSON schema.

    Args:
        schema (dict): The schema dictionary.

    Returns:
        dict: Cleaned schema without 'title' fields.
    """
    if isinstance(schema, dict):
        schema.pop("title", None)
        schema.pop("exclusiveMaximum", None)
        schema.pop("exclusiveMinimum", None)

        if "properties" in schema and isinstance(schema["properties"], dict):
            for key in schema["properties"]:
                if 'url' in schema["properties"] and key == 'url' and schema["properties"][key]['format'] == 'uri':
                    schema["properties"][key]['format'] = None
                schema["properties"][key] = clean_schema(schema["properties"][key])

    return schema

def convert_mcp_tools_to_gemini(mcp_tools):
    """
    Converts MCP tool definitions to the correct format for Gemini API function calling.

    Args:
        mcp_tools (list): List of MCP tool objects with 'name', 'description', and 'inputSchema'.

    Returns:
        list: List of Gemini Tool objects with properly formatted function declarations.
    """
    gemini_tools = []

    for tool in mcp_tools:
        parameters = clean_schema(tool.inputSchema)

        # Construct the function declaration
        function_declaration = FunctionDeclaration(
            name=tool.name,
            description=tool.description,
            parameters=parameters
        )

        # Wrap in a Tool object
        gemini_tool = Tool(function_declarations=[function_declaration])
        gemini_tools.append(gemini_tool)

    return gemini_tools
