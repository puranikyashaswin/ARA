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

SYSTEM_PROMPT = """You are ARA, an expert math problem solver that achieves 95%+ accuracy.

## CORE RULES:
1. ALWAYS use the calculator tool for every calculation - never calculate in your head
2. Break complex problems into clear steps
3. Calculate each step separately with the calculator
4. End with: #### [final number]

## STEP-BY-STEP APPROACH:
For multi-step problems:
1. Identify ALL the quantities and relationships
2. Work through the problem step by step
3. Use calculator for EACH arithmetic operation
4. Track intermediate results
5. Final answer with #### [number]

## EXAMPLE:
Problem: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and uses 4 to bake muffins. She sells the rest for $2 each. How much money does she make daily?

Step 1: Find eggs remaining after breakfast and baking
[Call calculator("16 - 3 - 4")] → 9 eggs

Step 2: Calculate money from selling eggs  
[Call calculator("9 * 2")] → $18

#### 18

## CRITICAL:
- Use calculator for EVERY operation (even simple ones like 16-3)
- Always verify your logic matches the problem
- Output exactly #### [number] at the end (just the number, no units)
- For percentage problems: call calculator("X * Y / 100") 
- For "increased by X%": original + (original * X / 100)
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
    
    # Secondary OpenRouter-compatible endpoint for reliability
    fallback_key = os.getenv("FALLBACK_API_KEY") or os.getenv("NVIDIA_API_KEY")
    if fallback_key:
        return ChatOpenAI(
            model="meta/llama-3.3-70b-instruct",
            openai_api_key=fallback_key,
            openai_api_base="https://integrate.api.nvidia.com/v1",
            temperature=0.1,
            timeout=180,
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
    import time
    from openai import InternalServerError, APITimeoutError, APIConnectionError, RateLimitError

    llm = get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    messages = state["messages"]

    # Add system prompt if not present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    # Enhanced retry logic for API errors - 5 attempts with longer backoff
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}
        except (InternalServerError, APITimeoutError, APIConnectionError, RateLimitError) as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 2  # Exponential backoff: 2, 4, 8, 16, 32 seconds
                time.sleep(wait_time)
            else:
                raise e
        except Exception as e:
            # For other errors, retry twice with shorter waits
            if attempt < 2:
                time.sleep(2)
            else:
                raise e

    return {"messages": []}


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
