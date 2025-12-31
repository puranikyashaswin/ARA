"""
Simplified v2 Orchestrator - Robust Multi-Agent Reasoning.

This version prioritizes reliability and accuracy by calling the stable v1 agent 
for each sub-task and using simplified synthesis logic.
"""

import sys
import os
import re
import time
import json
import logging
from typing import Any, Tuple, List, Dict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.orchestrator.agents import PlanningAgent, VerificationAgent, SynthesisAgent, SubTask, VerificationResult
from src.agent.graph import run_agent, get_final_answer

# Global timeout for agent calls (seconds)
AGENT_TIMEOUT = 45

class SimpleOrchestrator:
    """A more robust multi-agent orchestrator that leverages v1 agent directly."""
    
    def __init__(self, verbose: bool = True):
        self.planning_agent = PlanningAgent()
        self.verification_agent = VerificationAgent()
        self.synthesis_agent = SynthesisAgent()
        self.verbose = verbose
        
        # Setup basic logging
        logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
        self.logger = logging.getLogger("SimpleOrchestrator")

    def _extract_numeric(self, text: str) -> str:
        """Robust numeric extraction from text."""
        if not text:
            return ""
            
        # 1. Look for #### pattern
        match = re.search(r'####\s*(\d+(?:,\d+)*(?:\.\d+)?)', text)
        if match:
            return match.group(1).replace(",", "")
            
        # 2. Look for "Final Answer: X"
        match = re.search(r'[Ff]inal [Aa]nswer[:\s]+(\d+(?:,\d+)*(?:\.\d+)?)', text)
        if match:
            return match.group(1).replace(",", "")
            
        # 3. Look for "= X" at the end
        match = re.search(r'=\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*\.?\s*$', text)
        if match:
            return match.group(1).replace(",", "")
            
        # 4. Fallback: last number in text
        nums = re.findall(r'(\d+(?:\.\d+)?)', text)
        if nums:
            return nums[-1]
            
        return text.strip()

    def run(self, query: str) -> Dict[str, Any]:
        """Run the simplified multi-agent flow."""
        start_time = time.time()
        execution_path = ["planning"]
        
        if self.verbose:
            print(f"\n[1/4] 📋 Planning for: {query[:60]}...")
            
        # Step 1: Planning
        try:
            complexity, subtasks = self.planning_agent.plan(query)
        except Exception as e:
            self.logger.error(f"Planning failed: {e}")
            complexity = "simple"
            subtasks = [SubTask(id=1, description=query)]

        if self.verbose:
            print(f"      Complexity: {complexity}, Subtasks: {len(subtasks)}")

        # Step 2: Execution (Sequential with dependency handling)
        results = {}
        subtask_data = []
        execution_path.append("execution")
        
        for i, st in enumerate(subtasks, 1):
            if self.verbose:
                print(f"[2/4] ⚡ Executing Sub-task {st.id}/{len(subtasks)}: {st.description[:50]}...")
            
            # Build context from dependencies
            context_str = ""
            if results:
                dep_results = [f"- Result of '{subtasks[j-1].description}': {results[j]}" for j in results]
                context_str = "Context from previously solved sub-tasks:\n" + "\n".join(dep_results)
            
            # Call v1 agent with FULL context to avoid "missing info" errors
            # We tell it specifically what to solve now.
            st_query = f"""You are executing a sub-task as part of a larger plan.
Original Goal: {query}

{context_str}

YOUR SPECIFIC TASK: {st.description}

INSTRUCTIONS:
1. Solve the CURRENT TASK using the provided context and the original goal.
2. The information you need is available in the 'Original Goal' or 'Context'. 
3. DO NOT say "I don't have enough information" if the numbers are present in the goal or context.
4. Provide a step-by-step solution for THIS task only.
5. End your response with: Final Answer: #### [numeric result]"""

            st_start = time.time()
            
            try:
                st_result = run_agent(st_query)
                st_answer = get_final_answer(st_result)
                st_numeric = self._extract_numeric(st_answer)
                
                results[st.id] = st_numeric
                st.result = st_numeric
                st.confidence = 0.9 # High confidence for v1 tool-based execution
                
                subtask_data.append({
                    "id": st.id,
                    "description": st.description,
                    "result": st_numeric,
                    "full_response": st_answer[:200] + "..."
                })
            except Exception as e:
                self.logger.error(f"Sub-task {st.id} failed: {e}")
                results[st.id] = "Error"
            
            if self.verbose:
                print(f"      Result: {results[st.id]} (Time: {time.time() - st_start:.2f}s)")

        # Step 3: Verification (Adaptive)
        verification = None
        # The final answer is typically the result of the LAST sub-task in a multi-step plan
        if subtasks:
            primary_answer = subtasks[-1].result
            reasoning = "Execution Trace:\n" + "\n".join([f"Step {s['id']} ({s['description']}): {s['result']}" for s in subtask_data])
            
            # Simple problems skip verification unless they look wrong
            should_verify = complexity in ["medium", "complex"] or primary_answer == "Error"
            
            if should_verify:
                if self.verbose:
                    print(f"[3/4] ✅ Verifying final result: {primary_answer}...")
                
                execution_path.append("verification")
                try:
                    verification = self.verification_agent.verify(query, str(primary_answer), reasoning)
                    if self.verbose:
                        status = "Valid" if verification.is_valid else "Invalid"
                        print(f"      Status: {status} (Confidence: {verification.confidence:.0%})")
                except Exception as e:
                    self.logger.error(f"Verification failed: {e}")
        else:
            primary_answer = "No answer generated"

        # Step 4: Final Answer (Synthesis)
        if self.verbose:
            print(f"[4/4] 🔗 Synthesizing final result...")
        
        execution_path.append("synthesis")
        
        # Build results map for synthesis
        results_map = {st.id: st.result for st in subtasks}
        
        try:
            final_answer, confidence = self.synthesis_agent.synthesize(
                query, results_map, subtasks, verification
            )
        except Exception as e:
            self.logger.error(f"Synthesis failed: {e}")
            final_answer = subtasks[-1].result if subtasks else "Error"
            confidence = 0.5

        # Scenario A: Result is still Error or None - Use v1 directly as fallback
        if not final_answer or final_answer == "Error" or str(final_answer).lower() == "none":
            if self.verbose:
                print("      ⚠️ Synthesis failed or returned None. Triggering v1 fallback...")
            try:
                fb_result = run_agent(query)
                final_answer = self._extract_numeric(get_final_answer(fb_result))
                execution_path.append("v1_fallback")
            except Exception as e:
                self.logger.error(f"Fallback failed: {e}")
                final_answer = "Failed to generate answer"

        # Final Cleanup
        total_time = time.time() - start_time
        
        return {
            "answer": final_answer,
            "confidence": confidence,
            "complexity": complexity,
            "subtasks": subtask_data,
            "verification": {
                "is_valid": verification.is_valid if verification else True,
                "confidence": verification.confidence if verification else 0.5,
                "issues": verification.issues if verification else "No verification"
            } if verification else None,
            "execution_path": execution_path,
            "time_taken": total_time
        }

def run_simple_v2(query: str, verbose: bool = True) -> Dict[str, Any]:
    """Helper to run the simple orchestrator."""
    orchestrator = SimpleOrchestrator(verbose=verbose)
    return orchestrator.run(query)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        res = run_simple_v2(q)
        print(f"\n✨ FINAL RESULT: {res['answer']}")
