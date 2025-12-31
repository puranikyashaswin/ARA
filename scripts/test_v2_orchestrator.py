#!/usr/bin/env python3
"""
Test script for the ARA v2 Multi-Agent Orchestrator.

Tests:
1. Simple problem (direct v1 path)
2. Medium problem (execution + synthesis)
3. Complex problem (full pipeline with verification)

Compares execution paths and shows agent activation.
"""

import sys
import os
import time
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def run_tests(verbose: bool = False):
    """Run the v2 orchestrator tests."""
    
    # Import here to avoid loading until needed
    from src.orchestrator.graph import run_v2_agent
    from src.agent.graph import run_agent, get_final_answer
    import re
    
    print("\n" + "=" * 70)
    print("🧪 ARA v2 Multi-Agent Orchestrator Tests")
    print("=" * 70)
    
    # Test cases
    tests = [
        {
            "name": "Simple (Direct v1)",
            "query": "What is 25% of 200?",
            "expected": "50",
            "complexity": "simple"
        },
        {
            "name": "Medium (Multi-step)",
            "query": "A store has 120 apples. They sell 40% on Monday. How many are left?",
            "expected": "72",
            "complexity": "medium"
        },
        {
            "name": "Complex (Verification)",
            "query": "A store has 120 apples. They sell 40% on Monday and 25% of the remaining apples on Tuesday. How many apples are left?",
            "expected": "54",
            "complexity": "complex"
        }
    ]
    
    results = []
    
    for i, test in enumerate(tests, 1):
        print(f"\n{'─' * 70}")
        print(f"📋 Test {i}: {test['name']}")
        print(f"{'─' * 70}")
        print(f"   Query: {test['query'][:60]}...")
        print(f"   Expected: {test['expected']}")
        
        # Run v2 agent
        start_time = time.time()
        try:
            v2_result = run_v2_agent(test["query"])
            v2_time = time.time() - start_time
            
            # Extract numeric answer
            v2_answer = v2_result.get("answer", "")
            # First try to find numeric pattern in the answer
            if not re.match(r'^\d+(?:\.\d+)?$', str(v2_answer)):
                # Look for #### pattern
                match = re.search(r'####?\s*(\d+(?:\.\d+)?)', str(v2_answer))
                if match:
                    v2_answer = match.group(1)
                else:
                    # Look for any standalone number at the end
                    match = re.search(r'(\d+(?:\.\d+)?)\s*\.?\s*$', str(v2_answer))
                    if match:
                        v2_answer = match.group(1)
                    else:
                        # Look for "= X" pattern
                        match = re.search(r'=\s*(\d+(?:\.\d+)?)', str(v2_answer))
                        if match:
                            v2_answer = match.group(1)
            
            # Normalize: convert 50.0 to 50, but keep 50.5 as 50.5
            if re.match(r'^\d+\.0+$', str(v2_answer)):
                v2_answer = str(int(float(v2_answer)))
            
            # Show results
            print(f"\n   🤖 v2 Result:")
            print(f"      Answer: {v2_answer}")
            print(f"      Confidence: {v2_result.get('confidence', 0):.0%}")
            print(f"      Complexity: {v2_result.get('complexity', 'unknown')}")
            print(f"      Path: {' → '.join(v2_result.get('execution_path', []))}")
            print(f"      Time: {v2_time:.2f}s")
            
            if verbose and v2_result.get("subtasks"):
                print(f"\n      📊 Sub-tasks:")
                for st in v2_result["subtasks"]:
                    print(f"         {st['id']}. {st['description'][:40]}...")
                    print(f"            → {st.get('result', 'N/A')}")
            
            if v2_result.get("verification"):
                v = v2_result["verification"]
                status = "✅ Valid" if v["is_valid"] else "❌ Invalid"
                print(f"\n      📋 Verification: {status} ({v['confidence']:.0%})")
            
            # Check correctness - normalize both sides
            expected_clean = str(test["expected"]).strip()
            answer_clean = str(v2_answer).strip()
            is_correct = answer_clean == expected_clean
            results.append({
                "name": test["name"],
                "passed": is_correct,
                "answer": v2_answer,
                "expected": test["expected"],
                "time": v2_time,
                "path": v2_result.get("execution_path", [])
            })
            
            if is_correct:
                print(f"\n   ✅ PASSED")
            else:
                print(f"\n   ❌ FAILED (got '{v2_answer}', expected '{test['expected']}')")
                
        except Exception as e:
            print(f"\n   ❌ ERROR: {type(e).__name__}: {e}")
            results.append({
                "name": test["name"],
                "passed": False,
                "error": str(e)
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in results if r.get("passed", False))
    total = len(results)
    
    for r in results:
        status = "✅" if r.get("passed") else "❌"
        path = " → ".join(r.get("path", [])) if r.get("path") else "N/A"
        print(f"   {status} {r['name']}: {path}")
    
    print(f"\n   Total: {passed}/{total} tests passed")
    
    # v1 vs v2 comparison
    if "--compare" in sys.argv or verbose:
        print("\n" + "=" * 70)
        print("📈 v1 vs v2 COMPARISON")
        print("=" * 70)
        print("   Running v1 on same queries for comparison...")
        
        for test in tests[:2]:  # Just run on first 2 for speed
            print(f"\n   Query: {test['query'][:50]}...")
            
            # v1
            start = time.time()
            try:
                v1_result = run_agent(test["query"])
                v1_answer = get_final_answer(v1_result)
                match = re.search(r'####\s*(\d+)', v1_answer)
                v1_answer = match.group(1) if match else "N/A"
                v1_time = time.time() - start
            except Exception as e:
                v1_answer = f"Error: {e}"
                v1_time = 0
            
            # Find corresponding v2 result
            v2_data = next((r for r in results if r["name"] == test["name"]), {})
            
            print(f"   v1: {v1_answer} ({v1_time:.2f}s)")
            print(f"   v2: {v2_data.get('answer', 'N/A')} ({v2_data.get('time', 0):.2f}s)")
    
    print("\n" + "=" * 70)
    
    return passed == total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ARA v2 Orchestrator")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--compare", action="store_true", help="Compare v1 vs v2")
    args = parser.parse_args()
    
    success = run_tests(verbose=args.verbose)
    sys.exit(0 if success else 1)
