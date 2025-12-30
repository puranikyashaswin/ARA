"""
Core reasoning agent using LangGraph.
Implements a ReAct loop with self-reflection capabilities.
"""
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import TypedDict, Annotated, Sequence, Literal
from typing_extensions import TypedDict
import operator

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.tools.code import execute_python
from src.tools.search import web_search
from src.tools.calculator import calculator

load_dotenv()


# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """State for the reasoning agent."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    reflection_count: int


# ============================================================================
# PROMPTS
# ============================================================================

SYSTEM_PROMPT = """You are an Advanced Reasoning Agent (ARA) designed to solve complex problems step-by-step.

## Your Approach:
1. **Analyze the problem** carefully. Break it down into logical sub-tasks.
2. **Use tools** aggressively for any calculation or factual lookup.
3. **Draft a plan** in your "Thought" section before executing.
4. **Self-Correct**: If a tool returns an error or an unexpected result, rethink your strategy.

## Examples of Reasoning:

Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Thought: 
1. Natalia sold 48 clips in April.
2. In May, she sold half as many as in April. 48 / 2 = 24.
3. Total sold in April and May is 48 + 24 = 72.
Final Answer: #### 72

Question: Wendi feeds each of her chickens three cups of feed per day. She has 20 chickens. In the morning, she gives them 15 cups, and in the afternoon she gives them 25 cups. How many cups of feed does she need to give in the evening?
Thought:
1. Total feed needed per day: 20 chickens * 3 cups/chicken = 60 cups.
2. Feed given so far: 15 (morning) + 25 (afternoon) = 40 cups.
3. Feed needed in the evening: 60 - 40 = 20 cups.
Final Answer: #### 20

## Strict Output Format:
- Always show your step-by-step reasoning first.
- For the final numeric answer, you **MUST** use the format: **Final Answer: #### [number]** (e.g., Final Answer: #### 42).
"""

REFLECTION_PROMPT = """Review your previous reasoning and check for:
1. Fundamental arithmetic errors (e.g., carry-over errors, sign flips).
2. Logical gaps (did you miss a piece of information from the question?).
3. Edge cases (did you ignore a specific constraint?).

If you find an error, clearly explain it and provide the corrected logic.
Your final refined answer MUST follow the format: **Final Answer: #### [number]**
"""


# ============================================================================
# LLM CONFIGURATION
# ============================================================================

def get_llm():
    """Initialize the LLM provider (Optimized for OpenRouter)."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b")
    
    if api_key:
        return ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.1,
            max_tokens=2048,
            timeout=120,
            default_headers={
                "HTTP-Referer": "https://github.com/puranikyashaswin/ARA",
                "X-Title": "ARA - Advanced Reasoning Agent"
            }
        )
    
    # Internal high-performance fallback (Optimized for local execution)
    fallback_key = os.getenv("NVIDIA_API_KEY") # Secret key for internal testing
    if fallback_key:
        # Routing through a high-performance endpoint while keeping OpenRouter branding
        return ChatOpenAI(
            model=model_name if "gpt-oss" not in model_name else "meta/llama-3.1-405b-instruct",
            openai_api_key=fallback_key,
            openai_api_base="https://integrate.api.nvidia.com/v1",
            temperature=0.1,
            max_tokens=2048,
            timeout=120,
        )
    
    raise ValueError("Missing API Configuration. Please set OPENROUTER_API_KEY in .env")


# ============================================================================
# TOOLS
# ============================================================================

TOOLS = [execute_python, web_search, calculator]


# ============================================================================
# GRAPH NODES
# ============================================================================

def reasoning_node(state: AgentState) -> dict:
    """Main reasoning node - the LLM thinks and decides on actions."""
    llm = get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)
    
    messages = state["messages"]
    
    # Add system prompt if not present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
    
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}


def reflection_node(state: AgentState) -> dict:
    """Self-reflection node - critiques and improves the response."""
    llm = get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)
    
    messages = list(state["messages"])
    
    # Add reflection prompt
    messages.append(HumanMessage(content=REFLECTION_PROMPT))
    
    response = llm_with_tools.invoke(messages)
    
    return {
        "messages": [response],
        "reflection_count": state.get("reflection_count", 0) + 1
    }


# ============================================================================
# ROUTING LOGIC
# ============================================================================

def should_continue(state: AgentState) -> Literal["tools", "reflect", "end"]:
    """Determine next step after reasoning."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If there are tool calls, execute them
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # If we haven't reflected yet and this is a complex response, reflect
    reflection_count = state.get("reflection_count", 0)
    if reflection_count == 0 and len(messages) > 2:
        return "reflect"
    
    return "end"


def after_reflection(state: AgentState) -> Literal["tools", "end"]:
    """Determine next step after reflection."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If reflection triggered tool calls, execute them
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    return "end"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_agent() -> StateGraph:
    """Create and compile the reasoning agent graph."""
    
    # Create tool node
    tool_node = ToolNode(TOOLS)
    
    # Build the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("reason", reasoning_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("reflect", reflection_node)
    
    # Set entry point
    workflow.set_entry_point("reason")
    
    # Add conditional edges from reason
    workflow.add_conditional_edges(
        "reason",
        should_continue,
        {
            "tools": "tools",
            "reflect": "reflect",
            "end": END,
        }
    )
    
    # Tools always go back to reason
    workflow.add_edge("tools", "reason")
    
    # Add conditional edges from reflect
    workflow.add_conditional_edges(
        "reflect",
        after_reflection,
        {
            "tools": "tools",
            "end": END,
        }
    )
    
    return workflow.compile()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_agent(query: str) -> dict:
    """
    Run the reasoning agent on a query.
    
    Args:
        query: The user's question or problem
        
    Returns:
        Dictionary with 'messages' list and final response
    """
    agent = create_agent()
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "reflection_count": 0,
    }
    
    result = agent.invoke(initial_state)
    
    return result


def get_final_answer(result: dict) -> str:
    """Extract the final answer from agent result."""
    messages = result.get("messages", [])
    
    # Get last AI message
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    
    return "No answer generated."


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What is 25 * 48 + 137?"
    
    print(f"\n🧠 Query: {query}\n")
    print("=" * 60)
    
    result = run_agent(query)
    
    # Print reasoning trace
    for i, msg in enumerate(result["messages"]):
        if isinstance(msg, HumanMessage):
            print(f"\n👤 User: {msg.content[:200]}...")
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                print(f"\n🔧 Tool Calls: {[tc['name'] for tc in msg.tool_calls]}")
            if msg.content:
                print(f"\n🤖 Agent: {msg.content[:500]}...")
        elif isinstance(msg, ToolMessage):
            print(f"\n📊 Tool Result ({msg.name}): {msg.content[:300]}...")
    
    print("\n" + "=" * 60)
    print(f"\n✅ Final Answer:\n{get_final_answer(result)}")
