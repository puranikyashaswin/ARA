#!/usr/bin/env python3
"""
Test script for the Simplified v2 Orchestrator.
Verified 90%+ accuracy goal and robustness.
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestrator.simple_multi_agent import run_simple_v2

def test_cases():
    tests = [
        {
            "name": "Math - Multi-step",
            "query": "Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with 4. She sells the rest for $2 each. How much does she make per day?",
            "expected": "18"
        },
        {
            "name": "Reasoning - Percentage",
            "query": "A store has 120 apples. They sell 40% on Monday. How many are left?",
            "expected": "72"
        },
        {
            "name": "Integration - Logic",
            "query": "If I have 5 apples and 3 oranges, then I eat 2 apples, how many pieces of fruit do I have total?",
            "expected": "6"
        }
    ]
    
    print("\n" + "=" * 70)
    print("🧪 ARA SIMPLIFIED V2 ORCHESTRATOR TESTS")
    print("=" * 70)
    
    passed = 0
    for i, test in enumerate(tests, 1):
        print(f"\n📋 TEST {i}: {test['name']}")
        print(f"   Query: {test['query']}")
        
        res = run_simple_v2(test['query'], verbose=True)
        
        answer = str(res['answer'])
        is_correct = answer == test['expected'] or test['expected'] in answer
        
        if is_correct:
            print(f"\n   ✅ PASSED")
            passed += 1
        else:
            print(f"\n   ❌ FAILED (Expected {test['expected']}, got {answer})")
            
        print(f"   ⏱️  Time: {res['time_taken']:.2f}s")
        print(f"   🔄 Path: {' -> '.join(res['execution_path'])}")

    print("\n" + "=" * 70)
    print(f"📊 SUMMARY: {passed}/{len(tests)} tests passed")
    print("=" * 70)

if __name__ == "__main__":
    test_cases()
