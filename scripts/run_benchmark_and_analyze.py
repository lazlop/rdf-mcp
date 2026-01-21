#!/usr/bin/env python3
"""
Comprehensive Benchmark Runner and Analyzer

This script orchestrates the complete benchmark workflow:
1. Runs the SPARQL query generation benchmark
2. Computes aggregate metrics
3. Analyzes failures (optional)
4. Generates comprehensive reports

Usage:
    python scripts/run_benchmark_and_analyze.py [--analyze-failures] [--max-message-chars 4000]
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark import main_async as run_benchmark
from aggregate_metrics import load_csv_data, compute_aggregate_metrics, format_report, export_json
from analyze_failures import analyze_failures_async


def create_timestamped_filename(base_name: str, extension: str = "csv") -> str:
    """Create a filename with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"


async def main_async(args):
    """Main async workflow."""
    print("=" * 80)
    print("BENCHMARK AND ANALYSIS WORKFLOW")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Step 1: Run Benchmark
    print("\n" + "=" * 80)
    print("STEP 1: Running Benchmark")
    print("=" * 80)
    
    # The benchmark will create its own timestamped file
    await run_benchmark()
    
    # Find the most recent CSV file in the results directory
    # Check both relative to project root and current directory
    possible_results_dirs = [
        Path(__file__).parent.parent / "results",  # Project root results dir
        Path("results"),  # Relative to current dir
        Path("../results"),  # One level up
        Path("."),  # Current directory
    ]
    
    csv_files = []
    for results_dir in possible_results_dirs:
        if results_dir.exists():
            found_files = sorted(results_dir.glob("*_run_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
            if found_files:
                csv_files = found_files
                break
    
    if not csv_files:
        print("ERROR: No benchmark results file found!")
        print(f"Searched in: {[str(d) for d in possible_results_dirs]}")
        return
    
    benchmark_csv = csv_files[0]
    print(f"\nBenchmark results saved to: {benchmark_csv}")
    
    # Step 2: Compute Aggregate Metrics
    print("\n" + "=" * 80)
    print("STEP 2: Computing Aggregate Metrics")
    print("=" * 80)
    
    print(f"Loading data from {benchmark_csv}...")
    data = load_csv_data(str(benchmark_csv))
    print(f"Loaded {len(data)} records")
    
    print("Computing aggregate metrics...")
    metrics = compute_aggregate_metrics(data)
    
    # Generate report
    report = format_report(metrics)
    print("\n" + report)
    
    # Save text report (use the same directory as the CSV file)
    report_filename = benchmark_csv.stem + "_metrics_report.txt"
    report_path = benchmark_csv.parent / report_filename
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nText report saved to: {report_path}")
    
    # Export JSON (use the same directory as the CSV file)
    json_filename = benchmark_csv.stem + "_metrics.json"
    json_path = benchmark_csv.parent / json_filename
    export_json(metrics, str(json_path))
    print(f"JSON metrics saved to: {json_path}")
    
    # Step 3: Analyze Failures (if requested)
    if args.analyze_failures:
        print("\n" + "=" * 80)
        print("STEP 3: Analyzing Failures")
        print("=" * 80)
        
        failure_output = benchmark_csv.stem + "_failure_analysis.csv"
        failure_path = benchmark_csv.parent / failure_output
        
        try:
            analyzed_df = await analyze_failures_async(
                csv_path=str(benchmark_csv),
                output_path=str(failure_path),
                model_name=args.model,
                api_key_file=args.api_key_file,
                row_matching_threshold=args.threshold,
                max_message_history_chars=args.max_message_chars,
            )
            print(f"\nFailure analysis saved to: {failure_path}")
        except Exception as e:
            print(f"\nWARNING: Failure analysis encountered an error: {e}")
            print("Continuing without failure analysis...")
    
    # Summary
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE")
    print("=" * 80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nGenerated files:")
    print(f"  - Benchmark results: {benchmark_csv}")
    print(f"  - Metrics report: {report_path}")
    print(f"  - Metrics JSON: {json_path}")
    if args.analyze_failures:
        print(f"  - Failure analysis: {failure_path}")
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run benchmark and analyze results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmark and compute metrics only
  python scripts/run_benchmark_and_analyze.py
  
  # Run benchmark, compute metrics, and analyze failures
  python scripts/run_benchmark_and_analyze.py --analyze-failures
  
  # Customize failure analysis parameters
  python scripts/run_benchmark_and_analyze.py --analyze-failures --max-message-chars 8000 --threshold 0.9
        """
    )
    
    parser.add_argument(
        "--analyze-failures",
        action="store_true",
        help="Run failure analysis after benchmark (requires LLM API access)"
    )
    
    parser.add_argument(
        "--model",
        default="lbl/cborg-coder",
        help="LLM model to use for failure analysis (default: lbl/cborg-coder)"
    )
    
    parser.add_argument(
        "--api-key-file",
        default="analysis-config.json",
        help="Path to JSON file with API credentials (default: analysis-config.json)"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="row_matching_f1 threshold for failures (default: 1.0)"
    )
    
    parser.add_argument(
        "--max-message-chars",
        type=int,
        default=4000,
        help="Maximum characters from message history for failure analysis (default: 4000)"
    )
    
    args = parser.parse_args()
    
    # Run the async workflow
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
