#!/usr/bin/env python3
"""Compare ReAct vs Multi-Agent frameworks on end-to-end imaging tasks.

Usage:
    python compare_frameworks.py \
        --task eht_black_hole_original \
        --model cds/Claude-4.6-opus \
        --base-url https://api.example.com/v1 \
        --api-key $OPENAI_API_KEY \
        [--max-iterations 10] \
        [--output results]

This script runs the same task with both frameworks and produces a
side-by-side comparison report.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation_harness.config import LLMConfig, RunConfig, TaskConfig
from evaluation_harness.runner import BenchmarkRunner


def run_framework(
    framework: str,
    task: str,
    model: str,
    base_url: str,
    api_key: str,
    max_iterations: int,
    timeout: int,
    output_dir: Path,
) -> dict:
    """Run a single framework and return the result dict."""
    repo_root = Path(__file__).resolve().parent
    task_dir = repo_root / "tasks" / task
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path("logs") / "interactions" / f"{task}_end_to_end_{framework}_{ts}.md"

    config = RunConfig(
        llm=LLMConfig(
            model=model,
            base_url=base_url,
            api_key=api_key,
        ),
        task=TaskConfig(
            task_name=task,
            task_dir=task_dir,
            mode="end_to_end",
        ),
        max_iterations=max_iterations,
        timeout_seconds=timeout,
        output_dir=output_dir,
        log_file=log_file,
        framework=framework,
    )

    logging.info("=" * 60)
    logging.info("Running framework: %s", framework)
    logging.info("=" * 60)

    runner = BenchmarkRunner(config)
    result = runner.run()

    from dataclasses import asdict
    return asdict(result)


def print_comparison(react_result: dict, multi_result: dict) -> None:
    """Print a side-by-side comparison table."""
    print("\n" + "=" * 70)
    print("  FRAMEWORK COMPARISON: ReAct vs Multi-Agent Pipeline")
    print("=" * 70)
    print(f"{'Metric':<30} {'ReAct':>18} {'Multi-Agent':>18}")
    print("-" * 70)

    # Status
    print(f"{'Stopped Reason':<30} {react_result.get('stopped_reason', 'N/A'):>18} {multi_result.get('stopped_reason', 'N/A'):>18}")

    # Iterations
    print(f"{'Iterations':<30} {react_result.get('iterations', 0):>18} {multi_result.get('iterations', 0):>18}")

    # Quality metrics
    rq = react_result.get("quality_metrics") or {}
    mq = multi_result.get("quality_metrics") or {}

    r_nrmse = rq.get("nrmse", "N/A")
    m_nrmse = mq.get("nrmse", "N/A")
    r_ncc = rq.get("ncc", "N/A")
    m_ncc = mq.get("ncc", "N/A")

    r_nrmse_str = f"{r_nrmse:.4f}" if isinstance(r_nrmse, (int, float)) else str(r_nrmse)
    m_nrmse_str = f"{m_nrmse:.4f}" if isinstance(m_nrmse, (int, float)) else str(m_nrmse)
    r_ncc_str = f"{r_ncc:.4f}" if isinstance(r_ncc, (int, float)) else str(r_ncc)
    m_ncc_str = f"{m_ncc:.4f}" if isinstance(m_ncc, (int, float)) else str(m_ncc)

    print(f"{'NRMSE (↓ better)':<30} {r_nrmse_str:>18} {m_nrmse_str:>18}")
    print(f"{'NCC (↑ better)':<30} {r_ncc_str:>18} {m_ncc_str:>18}")

    # Tokens
    print(f"{'Total Tokens':<30} {react_result.get('total_tokens', 0):>18,} {multi_result.get('total_tokens', 0):>18,}")
    print(f"{'Prompt Tokens':<30} {react_result.get('prompt_tokens', 0):>18,} {multi_result.get('prompt_tokens', 0):>18,}")
    print(f"{'Completion Tokens':<30} {react_result.get('completion_tokens', 0):>18,} {multi_result.get('completion_tokens', 0):>18,}")

    # Time
    r_time = react_result.get("wall_time_seconds", 0)
    m_time = multi_result.get("wall_time_seconds", 0)
    print(f"{'Wall Time (s)':<30} {r_time:>18.1f} {m_time:>18.1f}")

    # Files created
    r_files = len(react_result.get("files_created", []))
    m_files = len(multi_result.get("files_created", []))
    print(f"{'Files Created':<30} {r_files:>18} {m_files:>18}")

    print("=" * 70)

    # Winner determination
    winner_quality = None
    if isinstance(r_nrmse, (int, float)) and isinstance(m_nrmse, (int, float)):
        if r_nrmse < m_nrmse:
            winner_quality = "ReAct"
        elif m_nrmse < r_nrmse:
            winner_quality = "Multi-Agent"
        else:
            winner_quality = "Tie"

    winner_efficiency = None
    if react_result.get("total_tokens", 0) > 0 and multi_result.get("total_tokens", 0) > 0:
        if react_result["total_tokens"] < multi_result["total_tokens"]:
            winner_efficiency = "ReAct"
        elif multi_result["total_tokens"] < react_result["total_tokens"]:
            winner_efficiency = "Multi-Agent"
        else:
            winner_efficiency = "Tie"

    print(f"\n📊 Quality Winner: {winner_quality or 'Undetermined'}")
    print(f"💰 Efficiency Winner (tokens): {winner_efficiency or 'Undetermined'}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare ReAct vs Multi-Agent frameworks on imaging tasks"
    )
    parser.add_argument("--task", required=True, help="Task name (e.g. eht_black_hole_original)")
    parser.add_argument("--model", required=True, help="LLM model name")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=["react", "multi_agent"],
        choices=["react", "multi_agent"],
        help="Which frameworks to run (default: both)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("Error: --api-key or OPENAI_API_KEY env var required", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output)
    results = {}

    for framework in args.frameworks:
        print(f"\n{'='*60}")
        print(f"  Running {framework.upper()} framework on task: {args.task}")
        print(f"{'='*60}\n")

        t0 = time.time()
        try:
            result = run_framework(
                framework=framework,
                task=args.task,
                model=args.model,
                base_url=args.base_url,
                api_key=api_key,
                max_iterations=args.max_iterations,
                timeout=args.timeout,
                output_dir=output_dir,
            )
            results[framework] = result
            elapsed = time.time() - t0
            print(f"\n✅ {framework.upper()} completed in {elapsed:.1f}s")
        except Exception as e:
            logging.error("Framework %s failed: %s", framework, e, exc_info=True)
            results[framework] = {"error": str(e), "framework": framework}

    # Print comparison if both ran
    if "react" in results and "multi_agent" in results:
        if "error" not in results["react"] and "error" not in results["multi_agent"]:
            print_comparison(results["react"], results["multi_agent"])

    # Save combined comparison report
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"comparison_{args.task}_{ts}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump({
            "task": args.task,
            "model": args.model,
            "timestamp": ts,
            "results": results,
        }, f, indent=2)
    print(f"\n📄 Comparison report saved to: {report_path}")


if __name__ == "__main__":
    main()
