# MCP Usage Guide

## What is MCP?

Model Context Protocol (MCP) is an open standard that allows AI systems to share tools and capabilities. ARA exposes its reasoning agent as an MCP tool.

## Quick Start

### Installation

```bash
pip install mcp fastmcp
```

### Test the Server

```bash
python mcp_servers/test_server.py
```

### Run the MCP Server

```bash
python mcp_servers/ara_server.py
```

## Claude Desktop Integration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ara-reasoning": {
      "command": "python",
      "args": ["/absolute/path/to/ARA/mcp_servers/ara_server.py"],
      "env": {
        "OPENROUTER_API_KEY": "your-key-here"
      }
    }
  }
}
```

Restart Claude Desktop. You can now ask Claude to "use ARA to solve..." and it will invoke the reasoning agent.

## Available Tools

### `solve_with_reasoning`

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `problem` | string | ✅ | - | The problem to solve |
| `max_steps` | int | ❌ | 10 | Max reasoning steps |
| `enable_reflection` | bool | ❌ | true | Enable self-correction |

**Response:**
```json
{
  "answer": "42",
  "full_response": "Step-by-step reasoning...",
  "reasoning_steps": [...],
  "confidence": 0.85,
  "tools_used": ["calculator"],
  "step_count": 4
}
```

## Python Integration

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def solve_problem(query: str):
    server = StdioServerParameters(
        command="python",
        args=["mcp_servers/ara_server.py"]
    )
    
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            result = await session.call_tool(
                "solve_with_reasoning",
                {"problem": query}
            )
            return result

# Usage
import asyncio
answer = asyncio.run(solve_problem("What is 25% of 160?"))
```

## Publishing to MCP Registry

1. Create `mcp.json` manifest:
```json
{
  "name": "ara-reasoning",
  "version": "1.0.0",
  "description": "Advanced Reasoning Agent with step-by-step problem solving",
  "tools": ["solve_with_reasoning"],
  "command": "python mcp_servers/ara_server.py"
}
```

2. Submit to MCP registry (when available)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Connection refused | Ensure server is running |
| API key errors | Check OPENROUTER_API_KEY in env |
| Timeout | Increase timeout in client config |
