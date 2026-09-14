"""
pipeline.py

End-to-end AutoBench pipeline.

Run this to go from papers to benchmark results in one command:
  python src/pipeline.py

Steps:
  1. Fetch papers from ArXiv (or load seed data)
  2. Extract failure modes using Claude
  3. Generate novel adversarial test cases
  4. Evaluate model against test cases
  5. Print summary report
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from paper_fetcher import fetch_all_papers, save_papers, load_papers
from failure_extractor import extract_all_papers, save_extractions, load_extractions
from benchmark_generator import generate_all_benchmarks, save_benchmarks, load_benchmarks
from evaluator import run_evaluation, save_results


def run_pipeline(
    skip_fetch: bool = False,
    skip_extract: bool = False,
    skip_generate: bool = False,
    max_eval_cases: int = 10,
    model: str = "claude-sonnet-4-6"
):
    """
    Run the full AutoBench pipeline.

    Args:
        skip_fetch: Use existing papers.json if True
        skip_extract: Use existing extractions.json if True
        skip_generate: Use existing benchmark if True
        max_eval_cases: Limit evaluation cases for speed
        model: Model to evaluate
    """

    print("=" * 60)
    print("AutoBench: Automated AI Safety Benchmark Generation")
    print("=" * 60)

    # Step 1 - Papers
    papers_path = "data/papers.json"
    if skip_fetch or os.path.exists(papers_path):
        print("\nStep 1: Loading existing papers...")
        papers = load_papers(papers_path)
        print(f"  Loaded {len(papers)} papers")
    else:
        print("\nStep 1: Fetching papers from ArXiv...")
        papers = fetch_all_papers(max_per_query=5)
        save_papers(papers, papers_path)

    # Step 2 - Extract failure modes
    extractions_path = "data/extractions.json"
    if skip_extract and os.path.exists(extractions_path):
        print("\nStep 2: Loading existing extractions...")
        extractions = load_extractions(extractions_path)
        total_modes = sum(len(e.get("failure_modes", [])) for e in extractions)
        print(f"  Loaded {len(extractions)} extractions with {total_modes} failure modes")
    else:
        print("\nStep 2: Extracting failure modes from papers...")
        extractions = extract_all_papers(papers)
        save_extractions(extractions, extractions_path)

    # Step 3 - Generate benchmarks
    benchmark_path = "benchmarks/autobench_v1.json"
    if skip_generate and os.path.exists(benchmark_path):
        print("\nStep 3: Loading existing benchmarks...")
        benchmark = load_benchmarks(benchmark_path)
        print(f"  Loaded {benchmark['total_cases']} test cases")
    else:
        print("\nStep 3: Generating adversarial test cases...")
        test_cases = generate_all_benchmarks(extractions)
        save_benchmarks(test_cases, benchmark_path)
        benchmark = load_benchmarks(benchmark_path)

    # Step 4 - Evaluate
    print(f"\nStep 4: Evaluating {model} against benchmarks...")
    print(f"  Running {min(max_eval_cases, benchmark['total_cases'])} test cases...")
    evaluation = run_evaluation(benchmark, model=model, max_cases=max_eval_cases)
    save_results(evaluation, f"results/evaluation_{model.replace('-', '_')}.json")

    # Step 5 - Report
    print("\n" + "=" * 60)
    print("AutoBench Report")
    print("=" * 60)

    summary = evaluation["summary"]
    print(f"\nModel Evaluated: {summary['model']}")
    print(f"Test Cases Run:  {summary['evaluated']}")
    print(f"Safety Rate:     {summary['safety_rate']}%")
    print(f"Average Score:   {summary['average_score']}/100")
    print(f"Safe Responses:  {summary['safe_responses']}")
    print(f"Unsafe Responses:{summary['unsafe_responses']}")

    if summary.get("by_category"):
        print("\nBy Category:")
        for cat, stats in summary["by_category"].items():
            print(f"  {cat}:")
            print(f"    Safety rate: {stats['safety_rate']}% ({stats['safe']}/{stats['total']})")
            print(f"    Avg score:   {stats['average_score']}/100")

    print("\nFiles generated:")
    print(f"  Papers:      data/papers.json")
    print(f"  Extractions: data/extractions.json")
    print(f"  Benchmarks:  benchmarks/autobench_v1.json")
    print(f"  Results:     results/evaluation_{model.replace('-', '_')}.json")
    print("\nAutoBench complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoBench Pipeline")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip ArXiv fetch")
    parser.add_argument("--skip-extract", action="store_true", help="Skip failure extraction")
    parser.add_argument("--skip-generate", action="store_true", help="Skip benchmark generation")
    parser.add_argument("--max-cases", type=int, default=10, help="Max evaluation cases")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="Model to evaluate")
    args = parser.parse_args()

    run_pipeline(
        skip_fetch=args.skip_fetch,
        skip_extract=args.skip_extract,
        skip_generate=args.skip_generate,
        max_eval_cases=args.max_cases,
        model=args.model
    )
