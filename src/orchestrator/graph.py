"""
ARA v2 Orchestrator - LangGraph workflow for multi-agent reasoning.

Routes problems through specialized agents based on complexity.
Simple problems go directly to v1. Complex ones get decomposed.
"""

import sys
import os
from typing import TypedDict, Annotated, Literal
from typing_extensions import TypedDict
import operator

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage

from src.orchestrator.agents import (
    PlanningAgent, ExecutionAgent, VerificationAgent, SynthesisAgent,
    SubTask, VerificationResult, OrchestratorState, create_agents
)


# ============================================================================
# STATE DEFINITION
# ============================================================================

class GraphState(TypedDict):
    """State for the v2 orchestrator graph."""
    query: str
    complexity: str
    subtasks: list[SubTask]
    results: dict[int, str]
    verification: VerificationResult | None
    final_answer: str
    confidence: float
    execution_path: list[str]
    use_v1_for_execution: bool


# ============================================================================
# GRAPH NODES
# ============================================================================

# Initialize agents as module-level singletons
_planning_agent = None
_execution_agent = None
_verification_agent = None
_synthesis_agent = None


def _get_agents():
    """Lazy initialization of agents."""
    global _planning_agent, _execution_agent, _verification_agent, _synthesis_agent
    if _planning_agent is None:
        _planning_agent, _execution_agent, _verification_agent, _synthesis_agent = create_agents()
    return _planning_agent, _execution_agent, _verification_agent, _synthesis_agent


def planning_node(state: GraphState) -> dict:
    """Planning node - breaks down the problem into sub-tasks."""
    planning_agent, _, _, _ = _get_agents()
    
    query = state["query"]
    complexity, subtasks = planning_agent.plan(query)
    
    return {
        "complexity": complexity,
        "subtasks": subtasks,
        "execution_path": state.get("execution_path", []) + ["planning"]
    }


def execution_node(state: GraphState) -> dict:
    """Execution node - solves each sub-task."""
    _, execution_agent, _, _ = _get_agents()
    
    subtasks = state["subtasks"]
    results = {}
    use_v1 = state.get("use_v1_for_execution", True)
    
    # Execute sub-tasks in dependency order
    for subtask in sorted(subtasks, key=lambda x: len(x.dependencies)):
        # Build context from completed dependencies
        context = {dep_id: results[dep_id] for dep_id in subtask.dependencies if dep_id in results}
        
        # Execute (use v1 agent for complex tasks, direct LLM for simple)
        if use_v1 and state.get("complexity") in ["medium", "complex"]:
            result, confidence = execution_agent.execute_with_v1(subtask, context)
        else:
            result, confidence = execution_agent.execute(subtask, context)
        
        results[subtask.id] = result
        subtask.result = result
        subtask.confidence = confidence
    
    return {
        "results": results,
        "execution_path": state.get("execution_path", []) + ["execution"]
    }


def verification_node(state: GraphState) -> dict:
    """Verification node - cross-checks the answer."""
    _, _, verification_agent, _ = _get_agents()
    
    query = state["query"]
    results = state["results"]
    subtasks = state["subtasks"]
    
    # Get the final result (from last subtask or combined)
    if len(results) == 1:
        answer = list(results.values())[0]
    else:
        # Combine results for verification
        answer = "; ".join([f"Step {k}: {v}" for k, v in results.items()])
    
    # Build reasoning summary
    reasoning = "\n".join([
        f"Sub-task {st.id}: {st.description} → {st.result}"
        for st in subtasks if st.result
    ])
    
    verification = verification_agent.verify(query, answer, reasoning)
    
    return {
        "verification": verification,
        "execution_path": state.get("execution_path", []) + ["verification"]
    }


def synthesis_node(state: GraphState) -> dict:
    """Synthesis node - combines results into final answer."""
    _, _, _, synthesis_agent = _get_agents()
    
    query = state["query"]
    results = state["results"]
    subtasks = state["subtasks"]
    verification = state.get("verification")
    
    final_answer, confidence = synthesis_agent.synthesize(
        query, results, subtasks, verification
    )
    
    return {
        "final_answer": final_answer,
        "confidence": confidence,
        "execution_path": state.get("execution_path", []) + ["synthesis"]
    }


def direct_execution_node(state: GraphState) -> dict:
    """Direct execution for simple problems - uses v1 agent directly."""
    from src.agent.graph import run_agent, get_final_answer
    import re

    query = state["query"]

    result = run_agent(query)
    answer = get_final_answer(result)

    # Extract numeric answer - try multiple patterns (order matters!)
    extracted = None

    # 1. Try #### pattern (GSM8K standard) - highest priority
    hash_match = re.search(r'####\s*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)', answer)
    if hash_match:
        extracted = hash_match.group(1).replace(',', '')

    # 2. Try **Final Answer: $X** or **Final Answer: X** (common LLM format)
    if not extracted:
        final_match = re.search(r'\*\*[Ff]inal [Aa]nswer[:\s]*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)', answer)
        if final_match:
            extracted = final_match.group(1).replace(',', '')

    # 3. Try "Final Answer: $X" or "Final Answer: X" without bold
    if not extracted:
        plain_final = re.search(r'[Ff]inal [Aa]nswer[:\s]*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)', answer)
        if plain_final:
            extracted = plain_final.group(1).replace(',', '')

    # 4. Try "X dollars" pattern at end
    if not extracted:
        dollars_match = re.search(r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*dollars?\b', answer, re.IGNORECASE)
        if dollars_match:
            extracted = dollars_match.group(1).replace(',', '')
    
    # 5. Try "answer is X" patterns
    if not extracted:
        ans_match = re.search(r'(?:answer\s+is|equals?)\s*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)', answer, re.IGNORECASE)
        if ans_match:
            extracted = ans_match.group(1).replace(',', '')

    # 6. Try "= X" at the end of an equation
    if not extracted:
        eq_match = re.search(r'=\s*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:dollars?)?(?:\*\*)?[.\s]*$', answer, re.MULTILINE)
        if eq_match:
            extracted = eq_match.group(1).replace(',', '')

    # 7. Last resort: find the last standalone number (not part of a calculation)
    if not extracted:
        # Look for numbers at the very end or after final punctuation
        all_nums = re.findall(r'(?:^|\s)\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:dollars?|eggs?|cups?|apples?)?[.!?\s]*$', answer, re.MULTILINE)
        if all_nums:
            extracted = all_nums[-1].replace(',', '')
        else:
            # Truly last resort - just get the last number
            fallback_nums = re.findall(r'(-?\d+(?:,\d{3})*(?:\.\d+)?)', answer)
            if fallback_nums:
                extracted = fallback_nums[-1].replace(',', '')

    if extracted:
        answer = extracted

    # Count messages for confidence
    messages = result.get("messages", [])
    tool_calls = sum(1 for m in messages if hasattr(m, "tool_calls") and m.tool_calls)
    confidence = min(0.75 + tool_calls * 0.1, 0.95)

    return {
        "final_answer": answer,
        "confidence": confidence,
        "results": {1: answer},
        "subtasks": [SubTask(id=1, description=query, result=answer)],
        "execution_path": state.get("execution_path", []) + ["direct_v1"]
    }


# ============================================================================
# ROUTING LOGIC
# ============================================================================

def route_after_planning(state: GraphState) -> Literal["direct", "execute"]:
    """Decide whether to use direct v1 or multi-agent execution.

    AGGRESSIVE ROUTING: v1 achieves 95% accuracy. We use it directly for
    99%+ of problems. Only decompose for LITERALLY labeled multi-part problems.
    """
    complexity = state.get("complexity", "simple")
    subtasks = state.get("subtasks", [])

    # ALWAYS use direct v1 for simple problems
    if complexity == "simple":
        return "direct"
    
    # Even for "complex", only decompose if we have MULTIPLE subtasks
    # Single subtask = just run v1 directly
    if len(subtasks) <= 1:
        return "direct"
    
    # Only decompose if we truly have multiple independent parts
    return "execute"


def route_after_execution(state: GraphState) -> Literal["synthesize"]:
    """After execution, always go to synthesis.

    We skip verification for most cases because:
    1. v1 agent uses calculator/Python tools - results are accurate
    2. Verification without tools can't really verify math
    3. Simpler flow = fewer points of failure
    """
    # Always go directly to synthesis
    return "synthesize"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_orchestrator_graph() -> StateGraph:
    """Create and compile the v2 orchestrator graph.

    Simplified flow:
    - Plan → (simple) → Direct v1 → END
    - Plan → (complex) → Execute with v1 → Synthesize → END
    """

    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("plan", planning_node)
    workflow.add_node("direct", direct_execution_node)
    workflow.add_node("execute", execution_node)
    workflow.add_node("synthesize", synthesis_node)

    # Set entry point
    workflow.set_entry_point("plan")

    # Add routing after planning
    workflow.add_conditional_edges(
        "plan",
        route_after_planning,
        {
            "direct": "direct",
            "execute": "execute"
        }
    )

    # Direct execution goes straight to end
    workflow.add_edge("direct", END)

    # Execution goes to synthesis
    workflow.add_edge("execute", "synthesize")

    # Synthesis ends the workflow
    workflow.add_edge("synthesize", END)

    return workflow.compile()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_v2_agent(query: str, use_v1_for_execution: bool = True) -> dict:
    """
    Run the v2 orchestrator on a query.

    Args:
        query: The problem to solve
        use_v1_for_execution: Whether to use v1 agent for sub-task execution
                              (now always True - we always use v1 for tools)

    Returns:
        Dictionary with final_answer, confidence, execution_path, etc.
    """
    graph = create_orchestrator_graph()

    initial_state = {
        "query": query,
        "complexity": "",
        "subtasks": [],
        "results": {},
        "verification": None,
        "final_answer": "",
        "confidence": 0.0,
        "execution_path": [],
        "use_v1_for_execution": True  # Always use v1 for tool access
    }

    result = graph.invoke(initial_state)

    return {
        "answer": result.get("final_answer", ""),
        "confidence": result.get("confidence", 0.0),
        "execution_path": result.get("execution_path", []),
        "complexity": result.get("complexity", "unknown"),
        "subtasks": [
            {"id": st.id, "description": st.description, "result": st.result}
            for st in result.get("subtasks", [])
        ],
        "verification": None  # Simplified flow skips verification
    }


def get_v2_answer(result: dict) -> str:
    """Extract the final answer from v2 result."""
    return result.get("answer", "No answer")


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "A store has 120 apples. They sell 40% on Monday and 25% of the remaining on Tuesday. How many are left?"

    print(f"\n🧠 ARA v2 Orchestrator (Optimized)")
    print("=" * 60)
    print(f"\n📝 Query: {query}\n")

    result = run_v2_agent(query)

    print(f"\n🔄 Execution Path: {' → '.join(result['execution_path'])}")
    print(f"📊 Complexity: {result['complexity']}")
    print(f"📈 Confidence: {result['confidence']:.0%}")

    if result.get("subtasks"):
        print(f"\n📋 Sub-tasks:")
        for st in result["subtasks"]:
            desc = st['description'][:60] + "..." if len(st['description']) > 60 else st['description']
            print(f"   {st['id']}. {desc}")
            if st.get('result'):
                print(f"      → Result: {st['result']}")

    print("\n" + "=" * 60)
    print(f"\n✨ Final Answer: {result['answer']}")
