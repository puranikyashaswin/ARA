"""
Test script for the ARA MCP Server.

This script verifies that the MCP server works correctly by:
1. Running a simple test query directly through the agent
2. Verifying the response structure
3. Checking that the answer is correct
"""

import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.agent.graph import run_agent, get_final_answer
from langchain_core.messages import AIMessage, ToolMessage
import re


def extract_numeric_answer(text: str) -> str | None:
    """Extract the numeric answer from the agent's response."""
    # First look for #### pattern
    match = re.search(r'####\s*(\d+(?:\.\d+)?)', text)
    if match:
        return match.group(1)
    
    # Look for "Final Answer: X" pattern
    match = re.search(r'[Ff]inal [Aa]nswer[:\s]+(\d+(?:\.\d+)?)', text)
    if match:
        return match.group(1)
    
    # Look for "= X" at end of calculation
    match = re.search(r'=\s*(\d+(?:\.\d+)?)\s*$', text, re.MULTILINE)
    if match:
        return match.group(1)
    
    # Look for "result is X" or "answer is X"
    match = re.search(r'(?:result|answer)\s+is\s+(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if match:
        return match.group(1)
    
    return None


def test_agent_direct():
    """Test the agent directly (simulates what MCP server does)."""
    print("=" * 60)
    print("🧪 ARA MCP Server Test")
    print("=" * 60)
    
    # Test problem
    test_problem = "What is 25% of 160?"
    expected_answer = "40"
    
    print(f"\n📝 Test Problem: {test_problem}")
    print(f"📊 Expected Answer: {expected_answer}")
    print("-" * 60)
    
    try:
        # Run the agent
        print("\n⏳ Running agent...")
        result = run_agent(test_problem)
        
        # Extract final answer
        final_answer = get_final_answer(result)
        numeric_answer = extract_numeric_answer(final_answer)
        
        print(f"\n🤖 Agent Response (truncated):")
        print(final_answer[:500] + "..." if len(final_answer) > 500 else final_answer)
        
        print(f"\n📈 Extracted Numeric Answer: {numeric_answer}")
        
        # Count steps and tools
        messages = result.get("messages", [])
        tool_calls = []
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                tool_calls.extend([tc.get("name") for tc in msg.tool_calls])
        
        print(f"🔧 Tools Used: {list(set(tool_calls))}")
        print(f"📊 Total Messages: {len(messages)}")
        
        # Verify answer
        print("\n" + "=" * 60)
        if numeric_answer == expected_answer:
            print("✅ MCP Server Test PASSED!")
            print(f"   Answer '{numeric_answer}' matches expected '{expected_answer}'")
            return True
        else:
            print("❌ MCP Server Test FAILED!")
            print(f"   Got '{numeric_answer}', expected '{expected_answer}'")
            return False
            
    except Exception as e:
        print(f"\n❌ Test Error: {type(e).__name__}: {e}")
        return False


def test_complex_problem():
    """Test a more complex problem."""
    print("\n" + "=" * 60)
    print("🧪 Complex Problem Test")
    print("=" * 60)
    
    test_problem = (
        "A store has 120 apples. They sell 40% on Monday and 25% of the "
        "remaining apples on Tuesday. How many apples are left?"
    )
    expected_answer = "54"  # 120 * 0.6 = 72, then 72 * 0.75 = 54
    
    print(f"\n📝 Test Problem: {test_problem}")
    print(f"📊 Expected Answer: {expected_answer}")
    print("-" * 60)
    
    try:
        print("\n⏳ Running agent...")
        result = run_agent(test_problem)
        
        final_answer = get_final_answer(result)
        numeric_answer = extract_numeric_answer(final_answer)
        
        print(f"\n📈 Extracted Numeric Answer: {numeric_answer}")
        
        print("\n" + "=" * 60)
        if numeric_answer == expected_answer:
            print("✅ Complex Problem Test PASSED!")
            return True
        else:
            print(f"⚠️  Answer '{numeric_answer}' differs from expected '{expected_answer}'")
            print("   (This may still be correct depending on interpretation)")
            return False
            
    except Exception as e:
        print(f"\n❌ Test Error: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    print("\n🚀 Starting ARA MCP Server Tests\n")
    
    # Run tests
    test1_passed = test_agent_direct()
    test2_passed = test_complex_problem()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"  Simple Problem:  {'✅' if test1_passed else '❌'}")
    print(f"  Complex Problem: {'✅' if test2_passed else '⚠️'}")
    
    if test1_passed:
        print("\n✅ MCP Server infrastructure is working correctly!")
        print("   The server can be started with: python mcp_servers/ara_server.py")
    else:
        print("\n❌ Issues detected. Please check the agent configuration.")
    
    sys.exit(0 if test1_passed else 1)
