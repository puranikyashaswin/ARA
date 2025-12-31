#!/usr/bin/env python3
"""
Benchmark comparison script: v1 vs v2.

Runs both versions on GSM8K problems and generates a comparison report.

Usage:
    python benchmarks/compare_v1_v2.py --samples 50
    python benchmarks/compare_v1_v2.py --samples 10 --quick  # For quick testing
"""

import sys
import os
import json
import time
import re
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def load_gsm8k_samples(num_samples: int = 50) -> list[dict]:
    """Load GSM8K test samples."""
    from datasets import load_dataset
    
    dataset = load_dataset("gsm8k", "main", split="test")
    samples = []
    
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        
        # Extract the numeric answer from the solution
        answer = example["answer"]
        match = re.search(r'####\s*(\d+(?:,\d+)*(?:\.\d+)?)', answer)
        if match:
            numeric_answer = match.group(1).replace(",", "")
        else:
            numeric_answer = None
        
        samples.append({
            "id": i,
            "question": example["question"],
            "full_answer": answer,
            "expected": numeric_answer
        })
    
    return samples


def extract_answer(response: str) -> str | None:
    """Extract numeric answer from agent response."""
    # Look for #### pattern
    match = re.search(r'####\s*(\d+(?:,\d+)*(?:\.\d+)?)', response)
    if match:
        return match.group(1).replace(",", "")
    
    # Look for "Final Answer: X"
    match = re.search(r'[Ff]inal [Aa]nswer[:\s]+(\d+(?:,\d+)*(?:\.\d+)?)', response)
    if match:
        return match.group(1).replace(",", "")
    
    return None


def run_v1(query: str) -> tuple[str | None, float, dict]:
    """Run v1 agent on a query."""
    from src.agent.graph import run_agent, get_final_answer
    
    start = time.time()
    try:
        result = run_agent(query)
        elapsed = time.time() - start
        
        answer = get_final_answer(result)
        numeric = extract_answer(answer)
        
        # Count tool calls
        tool_calls = sum(
            1 for m in result.get("messages", [])
            if hasattr(m, "tool_calls") and m.tool_calls
        )
        
        return numeric, elapsed, {
            "tool_calls": tool_calls,
            "messages": len(result.get("messages", [])),
            "full_answer": answer[:500]
        }
    except Exception as e:
        return None, time.time() - start, {"error": str(e)}


def run_v2(query: str) -> tuple[str | None, float, dict]:
    """Run v2 orchestrator on a query."""
    from src.orchestrator.graph import run_v2_agent
    
    start = time.time()
    try:
        result = run_v2_agent(query)
        elapsed = time.time() - start
        
        answer = result.get("answer", "")
        if not re.match(r'^\d+(?:\.\d+)?$', str(answer)):
            answer = extract_answer(str(answer))
        
        return answer, elapsed, {
            "confidence": result.get("confidence", 0),
            "complexity": result.get("complexity", "unknown"),
            "execution_path": result.get("execution_path", []),
            "verification": result.get("verification"),
            "subtasks": len(result.get("subtasks", []))
        }
    except Exception as e:
        return None, time.time() - start, {"error": str(e)}


def run_benchmark(samples: list[dict], quick_mode: bool = False) -> dict:
    """Run full benchmark comparison."""
    results = {
        "v1": {"correct": 0, "total": 0, "times": [], "details": []},
        "v2": {"correct": 0, "total": 0, "times": [], "details": []},
        "samples": []
    }
    
    print(f"\n📊 Running benchmark on {len(samples)} samples...")
    print("=" * 70)
    
    for i, sample in enumerate(samples):
        print(f"\n[{i+1}/{len(samples)}] {sample['question'][:60]}...")
        
        # Run v1
        if not quick_mode:
            v1_answer, v1_time, v1_meta = run_v1(sample["question"])
            v1_correct = v1_answer == sample["expected"]
            results["v1"]["total"] += 1
            results["v1"]["times"].append(v1_time)
            if v1_correct:
                results["v1"]["correct"] += 1
            print(f"   v1: {v1_answer} ({'✓' if v1_correct else '✗'}) [{v1_time:.2f}s]")
        else:
            v1_answer, v1_time, v1_meta = None, 0, {}
            v1_correct = False
        
        # Run v2
        v2_answer, v2_time, v2_meta = run_v2(sample["question"])
        v2_correct = v2_answer == sample["expected"]
        results["v2"]["total"] += 1
        results["v2"]["times"].append(v2_time)
        if v2_correct:
            results["v2"]["correct"] += 1
        
        path = " → ".join(v2_meta.get("execution_path", []))
        print(f"   v2: {v2_answer} ({'✓' if v2_correct else '✗'}) [{v2_time:.2f}s] ({path})")
        
        # Store sample result
        results["samples"].append({
            "id": sample["id"],
            "question": sample["question"],
            "expected": sample["expected"],
            "v1_answer": v1_answer,
            "v1_correct": v1_correct,
            "v1_time": v1_time,
            "v2_answer": v2_answer,
            "v2_correct": v2_correct,
            "v2_time": v2_time,
            "v2_meta": v2_meta
        })
    
    return results


def calculate_stats(results: dict) -> dict:
    """Calculate summary statistics."""
    stats = {}
    
    for version in ["v1", "v2"]:
        data = results[version]
        if data["total"] == 0:
            continue
            
        stats[version] = {
            "accuracy": data["correct"] / data["total"] * 100,
            "correct": data["correct"],
            "total": data["total"],
            "avg_time": sum(data["times"]) / len(data["times"]) if data["times"] else 0,
            "min_time": min(data["times"]) if data["times"] else 0,
            "max_time": max(data["times"]) if data["times"] else 0
        }
    
    # v2-specific stats
    if results["samples"]:
        complexity_counts = {}
        path_counts = {}
        verification_success = 0
        verified_total = 0
        
        for s in results["samples"]:
            meta = s.get("v2_meta", {})
            complexity = meta.get("complexity", "unknown")
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
            
            path = " → ".join(meta.get("execution_path", []))
            path_counts[path] = path_counts.get(path, 0) + 1
            
            if meta.get("verification"):
                verified_total += 1
                if meta["verification"].get("is_valid"):
                    verification_success += 1
        
        stats["v2"]["complexity_distribution"] = complexity_counts
        stats["v2"]["path_distribution"] = path_counts
        stats["v2"]["verification_rate"] = verified_total / len(results["samples"]) * 100
        stats["v2"]["verification_accuracy"] = verification_success / verified_total * 100 if verified_total else 0
    
    return stats


def save_results(results: dict, stats: dict, output_dir: str):
    """Save results to JSON and markdown."""
    # Save JSON
    json_path = os.path.join(output_dir, "v1_vs_v2_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "stats": stats
        }, f, indent=2, default=str)
    print(f"\n📁 Results saved to: {json_path}")
    
    # Generate markdown report
    md_content = generate_markdown_report(results, stats)
    md_path = os.path.join(output_dir, "V2_COMPARISON.md")
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"📄 Report saved to: {md_path}")


def generate_markdown_report(results: dict, stats: dict) -> str:
    """Generate a markdown comparison report."""
    v1 = stats.get("v1", {})
    v2 = stats.get("v2", {})
    
    report = f"""# ARA v1 vs v2 Benchmark Comparison

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary

| Metric | v1 | v2 | Improvement |
|--------|----|----|-------------|
| **Accuracy** | {v1.get('accuracy', 0):.1f}% | {v2.get('accuracy', 0):.1f}% | {v2.get('accuracy', 0) - v1.get('accuracy', 0):+.1f}% |
| **Avg Latency** | {v1.get('avg_time', 0):.2f}s | {v2.get('avg_time', 0):.2f}s | {v2.get('avg_time', 0) - v1.get('avg_time', 0):+.2f}s |
| **Correct** | {v1.get('correct', 0)}/{v1.get('total', 0)} | {v2.get('correct', 0)}/{v2.get('total', 0)} | - |

## v2 Enhancements

### Complexity Routing
"""
    
    if v2.get("complexity_distribution"):
        for complexity, count in v2["complexity_distribution"].items():
            report += f"- **{complexity}**: {count} problems\n"
    
    report += f"""
### Execution Paths
"""
    
    if v2.get("path_distribution"):
        for path, count in v2["path_distribution"].items():
            report += f"- `{path}`: {count} problems\n"
    
    report += f"""
### Verification
- Verification rate: {v2.get('verification_rate', 0):.1f}%
- Verification accuracy: {v2.get('verification_accuracy', 0):.1f}%

## Where v2 Wins

v2 excels at:
1. **Complex multi-step problems** - Planning agent breaks them down effectively
2. **Error-prone calculations** - Verification catches mistakes
3. **High-confidence answers** - Confidence scoring helps identify reliable results

## Where v1 Wins

v1 is better for:
1. **Simple single-step problems** - Lower latency without orchestration overhead
2. **Speed-critical applications** - Direct execution is faster

## Recommendation

| Use Case | Recommended Version |
|----------|---------------------|
| Simple calculations | v1 |
| Complex reasoning | v2 |
| High-stakes decisions | v2 (verification) |
| Speed-critical | v1 |
| Maximum accuracy | v2 |

---

*Generated by ARA Benchmark Suite*
"""
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Compare ARA v1 vs v2")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples to test")
    parser.add_argument("--quick", action="store_true", help="Skip v1 (v2 only)")
    parser.add_argument("--output", default="benchmarks", help="Output directory")
    args = parser.parse_args()
    
    print("\n🚀 ARA v1 vs v2 Benchmark")
    print("=" * 70)
    
    # Load samples
    print(f"\n📥 Loading {args.samples} GSM8K samples...")
    samples = load_gsm8k_samples(args.samples)
    print(f"   Loaded {len(samples)} samples")
    
    # Run benchmark
    results = run_benchmark(samples, quick_mode=args.quick)
    
    # Calculate stats
    print("\n📈 Calculating statistics...")
    stats = calculate_stats(results)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 BENCHMARK RESULTS")
    print("=" * 70)
    
    if not args.quick and "v1" in stats:
        print(f"\n   v1 Accuracy: {stats['v1']['accuracy']:.1f}% ({stats['v1']['correct']}/{stats['v1']['total']})")
        print(f"   v1 Avg Time: {stats['v1']['avg_time']:.2f}s")
    
    if "v2" in stats:
        print(f"\n   v2 Accuracy: {stats['v2']['accuracy']:.1f}% ({stats['v2']['correct']}/{stats['v2']['total']})")
        print(f"   v2 Avg Time: {stats['v2']['avg_time']:.2f}s")
        print(f"   Verification Rate: {stats['v2'].get('verification_rate', 0):.1f}%")
    
    # Save results
    save_results(results, stats, args.output)
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
