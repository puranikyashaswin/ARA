# ARA MCP Server

This directory contains the Model Context Protocol (MCP) server implementation for ARA, allowing the v1 ReAct agent to be called by other agents and systems.

## Installation

```bash
pip install mcp fastmcp
```

## Testing

Verify the server works correctly:

```bash
python mcp_servers/test_server.py
```

Expected output:
```
✅ MCP Server Test PASSED!
```

## Running the Server

Start the MCP server (uses stdio transport):

```bash
python mcp_servers/ara_server.py
```

## Claude Desktop Integration

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "ara-reasoning": {
      "command": "python",
      "args": ["/path/to/ARA/mcp_servers/ara_server.py"],
      "env": {
        "OPENROUTER_API_KEY": "your-key-here"
      }
    }
  }
}
```

## Available Tools

### `solve_with_reasoning`

Solve a problem using step-by-step reasoning with tool support.

**Input Schema:**
```json
{
  "problem": "string (required) - The problem to solve",
  "max_steps": "integer (optional, default: 10) - Max reasoning steps",
  "enable_reflection": "boolean (optional, default: true) - Enable self-reflection"
}
```

**Output:**
```json
{
  "answer": "The extracted numeric answer or full response",
  "full_response": "Complete reasoning trace",
  "reasoning_steps": [...],
  "confidence": 0.85,
  "tools_used": ["calculator", "web_search"],
  "step_count": 5
}
```

## Integration Examples

### Python Client

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def call_ara():
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_servers/ara_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            result = await session.call_tool(
                "solve_with_reasoning",
                {"problem": "What is 15% of 200?"}
            )
            print(result)
```

## Architecture

```
User Query
    ↓
MCP Client (e.g., Claude Desktop)
    ↓
MCP Server (ara_server.py)
    ↓
ARA v1 Agent (src/agent/graph.py)
    ↓
Tools (calculator, code, search)
    ↓
Structured Response
```
