# MCP

sample  implementation of  client and server  using the MCP Python SDK and Google's Gemini LLM


## Features

- **Client Components**
  - FastAPI backend for handling requests
  - Streamlit-based web user interface
  - Integration with Google's Gemini LLM
  - Configurable MCP server connections

- **Server Components**
  - Document search using vector embeddings
  - Persistent vector storage using Chroma DB
  - MCP servers & tools

## Prerequisites

- Python 3.12 or higher
- Environment variables set up (see `.env.example` files)
- Required API keys:
  - Google Gemini API key
  - Serper API key (for web search)

## Configuration

The client can be configured using `mcp_servers_config.json`:
```json
{
    "mcpServers": {
        "server_name": {
            "command": "command_to_run_server",
            "args": ["arg1", "arg2"]
        }
    }
}
```