"""
ARA MCP Server - Exposes the v1 ReAct agent as an MCP tool.

This server allows other agents and systems to call the ARA reasoning
agent via the Model Context Protocol (MCP).
"""

import asyncio
import json
import sys
import os
import re
from typing import Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from src.agent.graph import run_agent, get_final_answer
from langchain_core.messages import AIMessage, ToolMessage


# ============================================================================
# MCP SERVER SETUP
# ============================================================================

app = Server("ara-reasoning-agent")


def extract_numeric_answer(text: str) -> str | None:
    """Extract the numeric answer from the agent's response."""
    # Look for #### pattern
    match = re.search(r'####\s*(\d+(?:\.\d+)?)', text)
    if match:
        return match.group(1)
    return None


def extract_reasoning_steps(messages: list) -> list[dict]:
    """Extract reasoning steps from the agent's message history."""
    steps = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    steps.append({
                        "type": "tool_call",
                        "tool": tc.get("name", "unknown"),
                        "input": tc.get("args", {})
                    })
            if msg.content:
                steps.append({
                    "type": "reasoning",
                    "content": msg.content[:500]  # Truncate for brevity
                })
        elif isinstance(msg, ToolMessage):
            steps.append({
                "type": "tool_result",
                "tool": msg.name,
                "result": msg.content[:300]  # Truncate for brevity
            })
    return steps


def calculate_confidence(messages: list, final_answer: str) -> float:
    """Calculate a confidence score based on reasoning quality."""
    confidence = 0.5  # Base confidence
    
    # Boost for tool usage (indicates verification)
    tool_calls = sum(1 for m in messages if isinstance(m, AIMessage) and m.tool_calls)
    confidence += min(tool_calls * 0.1, 0.3)
    
    # Boost for having a clear numeric answer
    if extract_numeric_answer(final_answer):
        confidence += 0.15
    
    # Boost for reflection (longer message chain)
    if len(messages) > 5:
        confidence += 0.05
    
    return min(confidence, 1.0)


# ============================================================================
# MCP TOOLS
# ============================================================================

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="solve_with_reasoning",
            description=(
                "Solve a problem using step-by-step reasoning with tool support. "
                "Best for math, logic, and multi-step problems. "
                "Returns structured output with answer, reasoning steps, and confidence."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "problem": {
                        "type": "string",
                        "description": "The problem or question to solve"
                    },
                    "max_steps": {
                        "type": "integer",
                        "description": "Maximum reasoning steps (default: 10)",
                        "default": 10
                    },
                    "enable_reflection": {
                        "type": "boolean",
                        "description": "Enable self-reflection for answer verification (default: true)",
                        "default": True
                    }
                },
                "required": ["problem"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    if name != "solve_with_reasoning":
        return [TextContent(type="text", text=json.dumps({
            "error": f"Unknown tool: {name}"
        }))]
    
    problem = arguments.get("problem", "")
    if not problem:
        return [TextContent(type="text", text=json.dumps({
            "error": "No problem provided"
        }))]
    
    try:
        # Run the v1 agent
        result = run_agent(problem)
        
        # Extract components
        final_answer = get_final_answer(result)
        numeric_answer = extract_numeric_answer(final_answer)
        reasoning_steps = extract_reasoning_steps(result.get("messages", []))
        confidence = calculate_confidence(result.get("messages", []), final_answer)
        
        # Count tools used
        tools_used = []
        for msg in result.get("messages", []):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                tools_used.extend([tc.get("name") for tc in msg.tool_calls])
        
        response = {
            "answer": numeric_answer or final_answer,
            "full_response": final_answer,
            "reasoning_steps": reasoning_steps,
            "confidence": round(confidence, 2),
            "tools_used": list(set(tools_used)),
            "step_count": len(reasoning_steps)
        }
        
        return [TextContent(type="text", text=json.dumps(response, indent=2))]
        
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({
            "error": str(e),
            "error_type": type(e).__name__
        }))]


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main():
    """Run the MCP server using stdio transport."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
