import argparse
import json
import re
import sys
import os
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
import signal

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

from src.agent.graph import run_agent, get_final_answer
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage

load_dotenv()

# ============================================================================
# ANSWER EXTRACTION & NORMALIZATION
# ============================================================================

def extract_numeric_answer(text: str) -> Optional[float]:
    """Extract the final numeric answer from text, handling units and currency."""
    if not text:
        return None
    
    # Priority 1: GSM8K standard delimiter ####
    gsm8k_match = re.search(r'####\s*([+-]?[\d,]+\.?\d*)', text)
    if gsm8k_match:
        return float(gsm8k_match.group(1).replace(',', ''))
        
    # Priority 2: "Final Answer:" pattern (handle $ and units)
    final_match = re.search(r'Final Answer[:\s]*\$?\s*([+-]?[\d,]+\.?\d*)', text, re.IGNORECASE)
    if final_match:
        return float(final_match.group(1).replace(',', ''))
    
    # Priority 3: find all numbers and take the last one (least reliable)
    # Filter for numbers that look like final answers (not part of problem text)
    numbers = re.findall(r'[+-]?[\d,]+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1].replace(',', ''))
        except ValueError:
            pass
    
    return None


def normalize_answer(answer: str) -> Optional[float]:
    """Normalize a ground truth answer string to a number."""
    match = re.search(r'####\s*([+-]?[\d,]+\.?\d*)', answer)
    if match:
        return float(match.group(1).replace(',', ''))
    try:
        return float(answer.replace(',', ''))
    except (ValueError, TypeError):
        return None


def is_correct(predicted: Optional[float], ground_truth: Optional[float], tolerance: float = 0.01) -> bool:
    """Check if predicted answer matches ground truth within tolerance."""
    if predicted is None or ground_truth is None:
        return False
    
    # Handle integer matching exactly
    if abs(ground_truth - round(ground_truth)) < 1e-9 and abs(predicted - round(predicted)) < 1e-9:
        return int(round(predicted)) == int(round(ground_truth))
    
    # Floating point comparison
    if ground_truth == 0:
        return abs(predicted) < tolerance
    return abs(predicted - ground_truth) / abs(ground_truth) < tolerance

# ============================================================================
# VALIDATION REPORTING
# ============================================================================

def generate_validation_report(results: Dict, output_dir: str):
    """Generate a high-fidelity validation report for auditing accuracy."""
    output_path = Path(output_dir)
    report_file = output_path / "validation_report.md"
    
    # Select 3 random correct samples for auditing
    correct_samples = [d for d in results["details"] if d.get("correct") is True]
    audit_samples = random.sample(correct_samples, min(3, len(correct_samples))) if correct_samples else []
    
    # Select all failed samples
    failed_samples = [d for d in results["details"] if not d.get("correct") and d.get("predicted") is not None]
    error_samples = [d for d in results["details"] if d.get("error")]

    md_content = f"""# 🛡️ ARA Benchmark Validation Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Random Seed: {results.get('seed', 'N/A')}

## 📑 Verification Summary
| Metric | Value | Status |
| :--- | :--- | :--- |
| **Total Samples** | {results['num_samples']} | - |
| **Verified Accuracy** | {results['accuracy']:.1%} | {'✅ HIGH' if results['accuracy'] > 0.9 else '🟡 MODERATE'} |
| **API/System Errors** | {results['errors']} | {'🟢 STABLE' if results['errors'] < 2 else '🔴 UNSTABLE'} |

"""
    if results['accuracy'] >= 0.95:
        md_content += "> [!IMPORTANT]\n> **FLAG: Suspiciously Perfect Score.** This accuracy level (95%+) suggests either an elite model/prompt synergy or potential dataset leakage. Manual audit recommended below.\n\n"

    md_content += "## 🧠 Audit: Correct Reasoning Traces (Random Selection)\n"
    for s in audit_samples:
        md_content += f"### Sample #{s['sample_id']}\n"
        md_content += f"**Question**: {s['question']}\n\n"
        md_content += f"**Trace**:\n```text\n{s.get('full_trace', 'Trace not captured')}\n```\n"
        md_content += f"**Result**: Predicted `{s['predicted']}` | Truth `{s['ground_truth']}`\n\n---\n"

    md_content += "## ❌ Root Cause Analysis: Failures\n"
    for s in failed_samples + error_samples:
        md_content += f"### Sample #{s['sample_id']}\n"
        md_content += f"**Question**: {s['question']}\n"
        md_content += f"**Expected**: `{s['ground_truth']}` | **Actual**: `{s.get('predicted', 'ERROR')}`\n"
        md_content += f"**Failure Log**: {s.get('error', 'Incorrect Logic / Extraction failure')}\n\n"

    with open(report_file, "w") as f:
        f.write(md_content)
    return report_file

# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def run_gsm8k_eval(num_samples: int = 100, output_dir: str = "benchmarks", validate: bool = False, seed: int = 42):
    """Run GSM8K evaluation with resumability and tool tracking."""
    random.seed(seed)
    print(f"\n🧪 Initializing GSM8K Benchmark ({num_samples} samples | Seed: {seed})")
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    checkpoint_file = output_path / "gsm8k_checkpoint.json"
    
    # Load dataset
    dataset = load_dataset("gsm8k", "main", split="test")
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)
    selected_indices = all_indices[:min(num_samples, len(dataset))]
    samples = [dataset[idx] for idx in selected_indices]
    
    results = {
        "dataset": "GSM8K",
        "num_samples": len(samples),
        "seed": seed,
        "correct": 0,
        "incorrect": 0,
        "errors": 0,
        "total_tokens": 0,
        "tool_stats": {"calculator": 0, "execute_python": 0, "web_search": 0},
        "details": [],
        "latencies": []
    }
    
    start_index = 0
    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            checkpoint = json.load(f)
            if checkpoint.get("seed") == seed and checkpoint["num_samples"] == num_samples:
                results = checkpoint
                start_index = len(results["details"])
                print(f"🔄 Resuming from checkpoint (Sample {start_index+1})")
    
    pbar = tqdm(total=num_samples, initial=start_index, desc="Benchmarking ARA")
    
    for i in range(start_index, len(samples)):
        sample = samples[i]
        question = sample["question"]
        ground_truth = normalize_answer(sample["answer"])
        
        start_time = time.time()
        try:
            def handler(signum, frame):
                raise TimeoutError("Problem execution timed out (180s)")
            
            max_retries = 3
            agent_result = None
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    signal.signal(signal.SIGALRM, handler)
                    signal.alarm(180)
                    try:
                        agent_result = run_agent(question)
                        break
                    finally:
                        signal.alarm(0)
                except Exception as e:
                    last_error = e
                    time.sleep(2)
            
            if agent_result is None: raise last_error
            
            latency = time.time() - start_time
            answer_text = get_final_answer(agent_result)
            predicted = extract_numeric_answer(answer_text)
            correct = is_correct(predicted, ground_truth)
            
            # Capture full trace for audit
            full_trace = ""
            for m in agent_result.get("messages", []):
                if isinstance(m, AIMessage):
                    full_trace += f"Thought: {m.content}\n"
                elif isinstance(m, ToolMessage):
                    full_trace += f"Observation: {m.content}\n"
            
            # Metrics
            for msg in agent_result.get("messages", []):
                if hasattr(msg, "tool_calls"):
                    for tc in msg.tool_calls:
                        tool_name = tc["name"]
                        results["tool_stats"][tool_name] = results["tool_stats"].get(tool_name, 0) + 1
                if hasattr(msg, "response_metadata") and "token_usage" in msg.response_metadata:
                    results["total_tokens"] += msg.response_metadata["token_usage"].get("total_tokens", 0)

            if correct: results["correct"] += 1
            else: results["incorrect"] += 1
            
            results["details"].append({
                "sample_id": i,
                "question": question,
                "ground_truth": ground_truth,
                "predicted": predicted,
                "correct": correct,
                "full_trace": full_trace,
                "latency": round(latency, 2)
            })
            results["latencies"].append(latency)
            
        except Exception as e:
            results["errors"] += 1
            results["details"].append({"sample_id": i, "question": question, "ground_truth": ground_truth, "error": str(e)})

        if (i + 1) % 5 == 0:
            with open(checkpoint_file, "w") as f: json.dump(results, f, indent=2)
        
        pbar.update(1)
        pbar.set_postfix({"Acc": f"{results['correct']/(i+1):.1%}"})

    pbar.close()
    
    results["accuracy"] = results["correct"] / num_samples
    results["avg_time_per_sample"] = sum(results["latencies"]) / len(results["latencies"]) if results["latencies"] else 0
    
    results_file = output_path / "gsm8k_results.json"
    with open(results_file, "w") as f: json.dump(results, f, indent=2)
    
    if validate:
        val_file = generate_validation_report(results, output_dir)
        print(f"🛡️ Validation Report generated: {val_file}")
    
    if checkpoint_file.exists(): os.remove(checkpoint_file)
    return results_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ARA on GSM8K")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--validate", action="store_true", help="Generate detailed validation report")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()
    
    res_file = run_gsm8k_eval(num_samples=args.samples, validate=args.validate, seed=args.seed)
    print(f"\n✅ Evaluation Complete! Results in {res_file}")
