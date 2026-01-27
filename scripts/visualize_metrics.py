#!/usr/bin/env python3
"""
Visualize Performance Metrics Across Multiple CSV Files

This script creates visualizations comparing model performance across multiple
CSV files, including:
- Bar charts comparing mean performance by building (distinguishing baselines from test runs)
- Scatter plots comparing mean performance vs token count
"""

import csv
import statistics
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_csv_data(filepath):
    """Load CSV data and return a list of row‑dictionaries."""
    csv.field_size_limit(10 * 1024 * 1024)
    with open(filepath, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def convert_to_numeric(value, default=0.0):
    """Convert a CSV cell to float – fallback to *default* on failure."""
    try:
        return float(value) if value and value.strip() else default
    except (ValueError, AttributeError):
        return default


# ----------------------------------------------------------------------
# Building-level metrics computation
# ----------------------------------------------------------------------
def compute_building_metrics(data):
    """Compute building-level metrics from CSV data."""
    by_building = defaultdict(lambda: {
        "count": 0,
        "arity_matching_f1": [],
        "exact_match_f1": [],
        "entity_set_f1": [],
        "row_matching_f1": [],
        "total_tokens": []
    })
    
    for row in data:
        query_id = row.get("query_id", "unknown")
        building = query_id.split("_")[0] if "_" in query_id else query_id
        
        # F1 scores
        arity_f1 = convert_to_numeric(row.get("arity_matching_f1"))
        exact_f1 = convert_to_numeric(row.get("exact_match_f1"))
        entity_f1 = convert_to_numeric(row.get("entity_set_f1"))
        row_f1 = convert_to_numeric(row.get("row_matching_f1"))
        total_tokens = convert_to_numeric(row.get("total_tokens"))
        
        bd = by_building[building]
        bd["count"] += 1
        bd["arity_matching_f1"].append(arity_f1)
        bd["exact_match_f1"].append(exact_f1)
        bd["entity_set_f1"].append(entity_f1)
        bd["row_matching_f1"].append(row_f1)
        bd["total_tokens"].append(total_tokens)
    
    # Compute statistics
    building_stats = {}
    for building, metrics in by_building.items():
        building_stats[building] = {
            "count": metrics["count"],
            "f1_scores": {},
            "total_tokens": {
                "mean": statistics.mean(metrics["total_tokens"]) if metrics["total_tokens"] else 0.0
            }
        }
        
        for f1_type in ["arity_matching_f1", "exact_match_f1", "entity_set_f1", 
                        "row_matching_f1"]:
            values = metrics[f1_type]
            mean_val = statistics.mean(values) if values else 0.0
            stdev_val = statistics.stdev(values) if len(values) > 1 else 0.0
            # Compute coefficient of variation (avoid division by zero)
            cv_val = (stdev_val / mean_val) if mean_val > 0 else 0.0
            
            building_stats[building]["f1_scores"][f1_type] = {
                "mean": mean_val,
                "stdev": stdev_val,
                "cv": cv_val
            }
    
    return building_stats


# ----------------------------------------------------------------------
# Overall metrics computation (across entire CSV)
# ----------------------------------------------------------------------
def compute_overall_metrics(data):
    """Compute overall metrics from CSV data (not per building)."""
    metrics = {
        "arity_matching_f1": [],
        "exact_match_f1": [],
        "entity_set_f1": [],
        "row_matching_f1": [],
        "total_tokens": []
    }
    
    for row in data:
        # F1 scores
        arity_f1 = convert_to_numeric(row.get("arity_matching_f1"))
        exact_f1 = convert_to_numeric(row.get("exact_match_f1"))
        entity_f1 = convert_to_numeric(row.get("entity_set_f1"))
        row_f1 = convert_to_numeric(row.get("row_matching_f1"))
        total_tokens = convert_to_numeric(row.get("total_tokens"))
        
        metrics["arity_matching_f1"].append(arity_f1)
        metrics["exact_match_f1"].append(exact_f1)
        metrics["entity_set_f1"].append(entity_f1)
        metrics["row_matching_f1"].append(row_f1)
        metrics["total_tokens"].append(total_tokens)
    
    # Compute statistics
    overall_stats = {
        "count": len(data),
        "f1_scores": {},
        "total_tokens": {
            "mean": statistics.mean(metrics["total_tokens"]) if metrics["total_tokens"] else 0.0
        }
    }
    
    for f1_type in ["arity_matching_f1", "exact_match_f1", "entity_set_f1", 
                    "row_matching_f1"]:
        values = metrics[f1_type]
        mean_val = statistics.mean(values) if values else 0.0
        stdev_val = statistics.stdev(values) if len(values) > 1 else 0.0
        cv_val = (stdev_val / mean_val) if mean_val > 0 else 0.0
        
        overall_stats["f1_scores"][f1_type] = {
            "mean": mean_val,
            "stdev": stdev_val,
            "cv": cv_val
        }
    
    return overall_stats


# ----------------------------------------------------------------------
# Bar chart: Mean by Building with baseline/test distinction
# ----------------------------------------------------------------------
def plot_building_comparison_bars(
    baseline_dict: Dict[str, dict],
    test_dict: Dict[str, dict],
    f1_metric: str = "exact_match_f1",
    output_path: str = "building_comparison.png"
):
    """
    Create grouped bar chart comparing mean F1 scores by building across CSVs.
    
    Args:
        baseline_dict: Dict mapping baseline names to building stats
        test_dict: Dict mapping test names to building stats
        f1_metric: Which F1 metric to plot
        output_path: Where to save the figure
    """
    # Get all unique buildings across all CSVs
    all_buildings = set()
    for stats in list(baseline_dict.values()) + list(test_dict.values()):
        all_buildings.update(stats.keys())
    buildings = sorted(all_buildings)
    
    # Prepare data
    x = np.arange(len(buildings))
    total_bars = len(baseline_dict) + len(test_dict)
    width = 0.8 / total_bars  # Width of bars
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # Baseline colors (grays) and patterns
    baseline_colors = ['#A9A9A9', '#808080', '#696969', '#5C5C5C']
    hatches = ['///', '\\\\\\', '|||', '---']
    
    # Test colors (vibrant)
    test_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    bar_idx = 0
    
    # Plot baselines first
    for i, (name, stats) in enumerate(baseline_dict.items()):
        means = []
        for building in buildings:
            if building in stats:
                means.append(stats[building]["f1_scores"][f1_metric]["mean"])
            else:
                means.append(0.0)
        
        offset = width * (bar_idx - total_bars / 2 + 0.5)
        color = baseline_colors[i % len(baseline_colors)]
        hatch = hatches[i % len(hatches)]
        ax.bar(x + offset, means, width, label=f"{name}", 
               alpha=0.7, color=color, hatch=hatch, edgecolor='black', linewidth=0.5)
        bar_idx += 1
    
    # Plot test runs
    for i, (name, stats) in enumerate(test_dict.items()):
        means = []
        for building in buildings:
            if building in stats:
                means.append(stats[building]["f1_scores"][f1_metric]["mean"])
            else:
                means.append(0.0)
        
        offset = width * (bar_idx - total_bars / 2 + 0.5)
        color = test_colors[i % len(test_colors)]
        ax.bar(x + offset, means, width, label=name, 
               alpha=0.8, color=color)
        bar_idx += 1
    
    ax.set_xlabel('Building', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{f1_metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
    ax.set_title(f'Performance Comparison by Building\n({f1_metric.replace("_", " ").title()})', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(buildings, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)  # Common y-axis for F1 scores
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Bar chart saved to: {output_path}")
    plt.close()


# ----------------------------------------------------------------------
# Scatter plot: Mean Performance vs Token Count (Overall)
# ----------------------------------------------------------------------
def plot_performance_vs_tokens_scatter(
    baseline_overall_dict: Dict[str, dict],
    test_overall_dict: Dict[str, dict],
    output_path: str = "performance_vs_tokens.png"
):
    """
    Create scatter plot of mean row matching F1 score vs mean token count (overall).
    
    Args:
        baseline_overall_dict: Dict mapping baseline names to overall stats
        test_overall_dict: Dict mapping test names to overall stats
        output_path: Where to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Baseline markers (grays)
    baseline_colors = ['#A9A9A9', '#808080', '#696969', '#5C5C5C']
    baseline_markers = ['s', '^', 'D', 'v']  # Square, triangle, diamond, inverted triangle
    
    # Test markers (vibrant)
    test_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    test_markers = ['o', 'o', 'o', 'o', 'o', 'o']  # Circles
    
    # Plot baselines
    for i, (name, stats) in enumerate(baseline_overall_dict.items()):
        tokens = stats["total_tokens"]["mean"]
        f1_score = stats["f1_scores"]["row_matching_f1"]["mean"]
        
        color = baseline_colors[i % len(baseline_colors)]
        marker = baseline_markers[i % len(baseline_markers)]
        ax.scatter(tokens, f1_score, label=f"{name}", 
                  alpha=0.6, s=200, color=color, marker=marker, edgecolors='black', linewidth=1.5)
        
        # Add label
        ax.annotate(name, (tokens, f1_score), 
                   fontsize=7, alpha=0.7, 
                   xytext=(8, 8), textcoords='offset points')
    
    # Plot test runs
    for i, (name, stats) in enumerate(test_overall_dict.items()):
        tokens = stats["total_tokens"]["mean"]
        f1_score = stats["f1_scores"]["row_matching_f1"]["mean"]
        
        color = test_colors[i % len(test_colors)]
        marker = test_markers[i % len(test_markers)]
        ax.scatter(tokens, f1_score, label=name, 
                  alpha=0.8, s=200, color=color, marker=marker, edgecolors='black', linewidth=1)
        
        # Add label
        ax.annotate(name, (tokens, f1_score), 
                   fontsize=7, alpha=0.7, 
                   xytext=(7, 2), textcoords='offset points')
    
    ax.set_xlabel('Mean Total Tokens', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Row Matching F1', fontsize=12, fontweight='bold')
    ax.set_title('Performance vs Token Usage', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0,155000)
    # ax.set_ylim(0, 1.0)  # Common y-axis for F1 scores
    ax.set_ylim(0.2, 0.7)  # Common y-axis for F1 scores
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Scatter plot saved to: {output_path}")
    plt.close()


# ----------------------------------------------------------------------
# Single-row multi-panel visualization with selected metrics (MEAN)
# ----------------------------------------------------------------------
def plot_all_f1_metrics_comparison(
    baseline_dict: Dict[str, dict],
    test_dict: Dict[str, dict],
    output_path: str = "all_metrics_comparison.png"
):
    """Create a single-row figure comparing selected F1 metrics (mean values)."""
    # Metrics in specified order: arity, exact match, row match, entity set
    f1_metrics = ["arity_matching_f1", "exact_match_f1", "row_matching_f1", "entity_set_f1"]
    metric_titles = ["Arity Matching F1", "Exact Match F1", "Row Matching F1", "Entity Set F1"]
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 4))
    
    # Get all unique buildings
    all_buildings = set()
    for stats in list(baseline_dict.values()) + list(test_dict.values()):
        all_buildings.update(stats.keys())
    buildings = sorted(all_buildings)
    
    x = np.arange(len(buildings))
    total_bars = len(baseline_dict) + len(test_dict)
    width = 0.8 / total_bars
    
    # Baseline colors and patterns
    baseline_colors = ['#A9A9A9', '#808080', '#696969', '#5C5C5C']
    hatches = ['///', '\\\\\\', '|||', '---']
    
    # Test colors
    test_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for idx, (f1_metric, title) in enumerate(zip(f1_metrics, metric_titles)):
        ax = axes[idx]
        bar_idx = 0
        
        # Plot baselines
        for i, (name, stats) in enumerate(baseline_dict.items()):
            means = []
            for building in buildings:
                if building in stats:
                    means.append(stats[building]["f1_scores"][f1_metric]["mean"])
                else:
                    means.append(0.0)
            
            offset = width * (bar_idx - total_bars / 2 + 0.5)
            color = baseline_colors[i % len(baseline_colors)]
            hatch = hatches[i % len(hatches)]
            label = f"{name}" if idx == 0 else None
            ax.bar(x + offset, means, width, label=label, 
                   alpha=0.7, color=color, hatch=hatch, edgecolor='black', linewidth=0.5)
            bar_idx += 1
        
        # Plot test runs
        for i, (name, stats) in enumerate(test_dict.items()):
            means = []
            for building in buildings:
                if building in stats:
                    means.append(stats[building]["f1_scores"][f1_metric]["mean"])
                else:
                    means.append(0.0)
            
            offset = width * (bar_idx - total_bars / 2 + 0.5)
            color = test_colors[i % len(test_colors)]
            label = name if idx == 0 else None
            ax.bar(x + offset, means, width, label=label, 
                   alpha=0.8, color=color)
            bar_idx += 1
        
        ax.set_xlabel('Building', fontsize=11, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
        else:
            ax.set_yticklabels([])  # Remove y-axis labels for non-leftmost plots
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(buildings, rotation=45, ha='right', fontsize=9)
        if idx == 0:
            ax.legend(fontsize=9, loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.0)  # Common y-axis for all F1 scores
    
    fig.suptitle('Performance Comparison Across Metrics (Mean)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.subplots_adjust(wspace=0.05)  # Make plots touch each other
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Multi-panel comparison saved to: {output_path}")
    plt.close()


# ----------------------------------------------------------------------
# Single-row multi-panel visualization with selected metrics (CV)
# ----------------------------------------------------------------------
def plot_all_f1_metrics_cv_comparison(
    baseline_dict: Dict[str, dict],
    test_dict: Dict[str, dict],
    output_path: str = "all_metrics_cv_comparison.png"
):
    """Create a single-row figure comparing selected F1 metrics (coefficient of variation values)."""
    # Metrics in specified order: arity, exact match, row match, entity set
    f1_metrics = ["arity_matching_f1", "exact_match_f1", "row_matching_f1", "entity_set_f1"]
    metric_titles = ["Arity Matching F1", "Exact Match F1", "Row Matching F1", "Entity Set F1"]
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    
    # Get all unique buildings
    all_buildings = set()
    for stats in list(baseline_dict.values()) + list(test_dict.values()):
        all_buildings.update(stats.keys())
    buildings = sorted(all_buildings)
    
    x = np.arange(len(buildings))
    total_bars = len(baseline_dict) + len(test_dict)
    width = 0.8 / total_bars
    
    # Baseline colors and patterns
    baseline_colors = ['#A9A9A9', '#808080', '#696969', '#5C5C5C']
    hatches = ['///', '\\\\\\', '|||', '---']
    
    # Test colors
    test_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for idx, (f1_metric, title) in enumerate(zip(f1_metrics, metric_titles)):
        ax = axes[idx]
        bar_idx = 0
        
        # Plot baselines
        for i, (name, stats) in enumerate(baseline_dict.items()):
            cvs = []
            for building in buildings:
                if building in stats:
                    cvs.append(stats[building]["f1_scores"][f1_metric]["cv"])
                else:
                    cvs.append(0.0)
            
            offset = width * (bar_idx - total_bars / 2 + 0.5)
            color = baseline_colors[i % len(baseline_colors)]
            hatch = hatches[i % len(hatches)]
            label = f"{name}" if idx == 0 else None
            ax.bar(x + offset, cvs, width, label=label, 
                   alpha=0.7, color=color, hatch=hatch, edgecolor='black', linewidth=0.5)
            bar_idx += 1
        
        # Plot test runs
        for i, (name, stats) in enumerate(test_dict.items()):
            cvs = []
            for building in buildings:
                if building in stats:
                    cvs.append(stats[building]["f1_scores"][f1_metric]["cv"])
                else:
                    cvs.append(0.0)
            
            offset = width * (bar_idx - total_bars / 2 + 0.5)
            color = test_colors[i % len(test_colors)]
            label = name if idx == 0 else None
            ax.bar(x + offset, cvs, width, label=label, 
                   alpha=0.8, color=color)
            bar_idx += 1
        
        ax.set_xlabel('Building', fontsize=11, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Coefficient of Variation', fontsize=11, fontweight='bold')
        else:
            ax.set_yticklabels([])  # Remove y-axis labels for non-leftmost plots
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(buildings, rotation=45, ha='right', fontsize=9)
        if idx == 0:
            ax.legend(fontsize=9, loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        # Auto-scale y-axis for CV
    
    fig.suptitle('Performance Variability Across Metrics (Coefficient of Variation)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.subplots_adjust(wspace=0.05)  # Make plots touch each other
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Multi-panel CV comparison saved to: {output_path}")
    plt.close()


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize performance metrics across multiple CSV files"
    )
    parser.add_argument(
        "--baseline",
        action='append',
        default=[],
        help="Baseline CSV files (format: name:path or just path). Can be specified multiple times."
    )
    parser.add_argument(
        "--test",
        action='append',
        default=[],
        help="Test CSV files (format: name:path or just path). Can be specified multiple times."
    )
    parser.add_argument(
        "--f1-metric",
        default="exact_match_f1",
        choices=["arity_matching_f1", "exact_match_f1", "entity_set_f1", 
                 "row_matching_f1"],
        help="F1 metric to use for primary visualizations"
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to save output plots"
    )
    args = parser.parse_args()
    
    if not args.baseline and not args.test:
        parser.error("Must specify at least one --baseline or --test CSV file")
    
    # Parse baseline CSV files
    baseline_dict = {}
    baseline_overall_dict = {}
    for csv_arg in args.baseline:
        if ':' in csv_arg:
            name, path = csv_arg.split(':', 1)
        else:
            path = csv_arg
            name = Path(path).stem
        
        print(f"Loading baseline {name} from {path}...")
        data = load_csv_data(path)
        building_stats = compute_building_metrics(data)
        overall_stats = compute_overall_metrics(data)
        baseline_dict[name] = building_stats
        baseline_overall_dict[name] = overall_stats
        print(f"  Loaded {len(data)} records, {len(building_stats)} buildings")
    
    # Parse test CSV files
    test_dict = {}
    test_overall_dict = {}
    for csv_arg in args.test:
        if ':' in csv_arg:
            name, path = csv_arg.split(':', 1)
        else:
            path = csv_arg
            name = Path(path).stem
        
        print(f"Loading test {name} from {path}...")
        data = load_csv_data(path)
        building_stats = compute_building_metrics(data)
        overall_stats = compute_overall_metrics(data)
        test_dict[name] = building_stats
        test_overall_dict[name] = overall_stats
        print(f"  Loaded {len(data)} records, {len(building_stats)} buildings")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    plot_building_comparison_bars(
        baseline_dict,
        test_dict,
        f1_metric=args.f1_metric,
        output_path=str(output_dir / f"building_comparison_{args.f1_metric}.png")
    )
    
    plot_performance_vs_tokens_scatter(
        baseline_overall_dict,
        test_overall_dict,
        output_path=str(output_dir / "performance_vs_tokens.png")
    )
    
    plot_all_f1_metrics_comparison(
        baseline_dict,
        test_dict,
        output_path=str(output_dir / "all_metrics_comparison.png")
    )
    
    plot_all_f1_metrics_cv_comparison(
        baseline_dict,
        test_dict,
        output_path=str(output_dir / "all_metrics_cv_comparison.png")
    )
    
    print("\nAll visualizations complete!")


if __name__ == "__main__":
    main()