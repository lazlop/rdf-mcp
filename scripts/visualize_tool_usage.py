#!/usr/bin/env python3
"""
Visualize Tool Usage Metrics

This script creates visualizations for:
- Tool impact on performance (delta F1 and frequency)
- Tool count per building
- Correlation between tool count and performance
"""

import csv
import re
import statistics
from collections import defaultdict, Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import sys
from scipy import stats
import seaborn as sns

# Increase CSV field size limit
csv.field_size_limit(sys.maxsize)


# ----------------------------------------------------------------------
# Tool extraction
# ----------------------------------------------------------------------
def extract_tool_calls(message_history: str) -> List[str]:
    """Extract tool names from ToolCallPart entries in the message history."""
    pattern = r"ToolCallPart\(tool_name='([^']+)'"
    matches = re.findall(pattern, message_history)
    return matches


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_csv_data(filepath):
    """Load CSV data and return a list of row-dictionaries."""
    with open(filepath, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def convert_to_numeric(value, default=0.0):
    """Convert a CSV cell to float – fallback to default on failure."""
    try:
        return float(value) if value and value.strip() else default
    except (ValueError, AttributeError):
        return default


# ----------------------------------------------------------------------
# Compute metrics
# ----------------------------------------------------------------------
def compute_tool_count_by_building(data):
    """Compute tool count statistics grouped by building."""
    by_building = defaultdict(lambda: {
        "query_count": 0,
        "total_tool_calls": 0,
    })
    
    for row in data:
        query_id = row.get("query_id", "unknown")
        building = query_id.split("_")[0] if "_" in query_id else query_id
        message_history = row.get("message_history", "")
        
        tools = extract_tool_calls(message_history)
        tools = [t for t in tools if t != "final_result"]
        
        bd = by_building[building]
        bd["query_count"] += 1
        bd["total_tool_calls"] += len(tools)
    
    # Compute statistics
    building_stats = {}
    for building, metrics in by_building.items():
        building_stats[building] = {
            "query_count": metrics["query_count"],
            "total_tool_calls": metrics["total_tool_calls"],
            "avg_tools_per_query": metrics["total_tool_calls"] / metrics["query_count"] 
                                   if metrics["query_count"] > 0 else 0.0,
        }
    
    return building_stats


def compute_tool_success_correlation(data):
    """Compute per-query tool usage and success metrics."""
    query_data = []
    
    for row in data:
        query_id = row.get("query_id", "unknown")
        building = query_id.split("_")[0] if "_" in query_id else "unknown"
        message_history = row.get("message_history", "")
        row_f1 = convert_to_numeric(row.get("row_matching_f1", 0))
        
        tools = extract_tool_calls(message_history)
        tools = [t for t in tools if t != "final_result"]
        
        query_data.append({
            "building": building,
            "tool_count": len(tools),
            "row_f1": row_f1,
            "tools": tools
        })
    
    return query_data


def compute_tool_impact_stats(query_data, min_occurrences=5):
    """Compute impact statistics for each tool."""
    tool_stats = defaultdict(lambda: {
        'with_tool_f1': [],
        'without_tool_f1': [],
        'usage_count': 0
    })
    
    # Get all unique tools
    all_tools = set()
    for query in query_data:
        all_tools.update(query['tools'])
    
    # For each tool, track F1 scores when present vs absent
    for tool in all_tools:
        for query in query_data:
            if tool in query['tools']:
                tool_stats[tool]['with_tool_f1'].append(query['row_f1'])
                tool_stats[tool]['usage_count'] += 1
            else:
                tool_stats[tool]['without_tool_f1'].append(query['row_f1'])
    
    # Compute statistics
    tool_metrics = {}
    for tool, stats in tool_stats.items():
        if stats['usage_count'] >= min_occurrences:
            with_f1 = statistics.mean(stats['with_tool_f1']) if stats['with_tool_f1'] else 0
            without_f1 = statistics.mean(stats['without_tool_f1']) if stats['without_tool_f1'] else 0
            
            tool_metrics[tool] = {
                'with_f1': with_f1,
                'without_f1': without_f1,
                'delta_f1': with_f1 - without_f1,
                'usage_count': stats['usage_count']
            }
    
    return tool_metrics


# ----------------------------------------------------------------------
# Visualization 1: Tool impact on performance
# ----------------------------------------------------------------------
def plot_tool_impact(tool_metrics, output_path="tool_impact.png"):
    """Create visualization showing tool impact on performance."""
    
    if not tool_metrics:
        print("No tool metrics to plot")
        return
    
    # Sort tools by delta F1 (impact on performance)
    sorted_tools = sorted(tool_metrics.items(), 
                        key=lambda x: x[1]['delta_f1'], 
                        reverse=True)
    
    tools = [t[0] for t in sorted_tools]
    delta_f1s = [t[1]['delta_f1'] for t in sorted_tools]
    usage_counts = [t[1]['usage_count'] for t in sorted_tools]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Delta F1 (impact on performance)
    colors = ['#2ca02c' if d > 0 else '#d62728' for d in delta_f1s]
    bars = ax1.barh(tools, delta_f1s, color=colors, alpha=0.7, 
                   edgecolor='black', linewidth=0.5)
    
    ax1.axvline(x=0, color='black', linewidth=1, linestyle='-', alpha=0.3)
    ax1.set_xlabel('F1 Score Impact\n(With Tool - Without Tool)', 
                  fontsize=11, fontweight='bold')
    ax1.set_ylabel('Tool', fontsize=11, fontweight='bold')
    ax1.set_title('Tool Impact on Performance', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Calculate x-axis limits to accommodate labels
    max_abs_delta = max(abs(min(delta_f1s)), abs(max(delta_f1s)))
    x_margin = max_abs_delta * 0.25  # 25% margin for labels
    ax1.set_xlim(-max_abs_delta - x_margin, max_abs_delta + x_margin)
    
    # Add value labels
    for i, (tool, delta) in enumerate(zip(tools, delta_f1s)):
        x_pos = delta + (0.01 if delta > 0 else -0.01)
        ha = 'left' if delta > 0 else 'right'
        ax1.text(x_pos, i, f'{delta:+.3f}', 
                va='center', ha=ha, fontsize=9)
    
    # Plot 2: Tool frequency
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(usage_counts)))
    bars = ax2.barh(tools, usage_counts, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('Tool Call Frequency', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Tool', fontsize=11, fontweight='bold')
    ax2.set_title('Tool Usage Frequency', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Calculate x-axis limit to accommodate labels
    max_count = max(usage_counts)
    x_margin = max_count * 0.20  # 20% margin for labels
    ax2.set_xlim(0, max_count + x_margin)
    
    # Add value labels
    for i, (tool, count) in enumerate(zip(tools, usage_counts)):
        ax2.text(count + (max_count * 0.02), i, f'{count}', 
                va='center', ha='left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Tool impact plot saved to: {output_path}")
    plt.close()


# ----------------------------------------------------------------------
# Visualization 2: Average tools per building
# ----------------------------------------------------------------------
def plot_tools_by_building(building_stats, output_path="tools_by_building.png"):
    """Create bar chart of average tool calls per building."""
    
    buildings = sorted(building_stats.keys())
    avg_tools = [building_stats[b]["avg_tools_per_query"] for b in buildings]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(buildings))
    bars = ax.bar(x, avg_tools, width=0.6, alpha=0.8, color='#1f77b4',
                  edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Building', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Tools Per Query', fontsize=12, fontweight='bold')
    ax.set_title('Average Tool Calls Per Query by Building', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(buildings, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Set y-axis limit to accommodate labels
    max_val = max(avg_tools)
    ax.set_ylim(0, max_val * 1.15)  # 15% margin above highest bar
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, avg_tools)):
        ax.text(bar.get_x() + bar.get_width()/2, val + (max_val * 0.02),
               f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Tools by building plot saved to: {output_path}")
    plt.close()


# ----------------------------------------------------------------------
# NEW: Visualization 3: Scatter plot with trend lines
# ----------------------------------------------------------------------
def plot_tool_count_vs_performance(query_data, output_path="tool_count_scatter.png"):
    """Create scatter plots showing correlation between tool count and F1 score per building."""
    
    # Group data by building
    by_building = defaultdict(list)
    for query in query_data:
        by_building[query['building']].append(query)
    
    buildings = sorted(by_building.keys())
    n_buildings = len(buildings)
    
    # Calculate grid dimensions
    n_cols = min(3, n_buildings)  # Max 3 columns
    n_rows = (n_buildings + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_buildings == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_buildings > 1 else axes
    
    for idx, building in enumerate(buildings):
        ax = axes[idx]
        building_queries = by_building[building]
        
        tool_counts = [q['tool_count'] for q in building_queries]
        f1_scores = [q['row_f1'] for q in building_queries]
        
        # Create scatter plot
        ax.scatter(tool_counts, f1_scores, alpha=0.6, s=50, 
                  color='#1f77b4', edgecolors='black', linewidth=0.5)
        
        # Add trend line if we have enough data points
        if len(tool_counts) > 1:
            z = np.polyfit(tool_counts, f1_scores, 1)
            p = np.poly1d(z)
            
            x_line = np.linspace(min(tool_counts), max(tool_counts), 100)
            ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8, label='Trend line')
            
            # Calculate correlation
            if len(set(tool_counts)) > 1:  # Need variation in x
                correlation, p_value = stats.pearsonr(tool_counts, f1_scores)
                
                # Add correlation text
                ax.text(0.05, 0.95, f'r = {correlation:.3f}\np = {p_value:.3f}',
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Tool Call Count', fontsize=11, fontweight='bold')
        ax.set_ylabel('Row Matching F1 Score', fontsize=11, fontweight='bold')
        ax.set_title(f'{building}\n(n={len(building_queries)})', 
                    fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    
    # Hide unused subplots
    for idx in range(n_buildings, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Tool count scatter plot saved to: {output_path}")
    plt.close()


# ----------------------------------------------------------------------
# NEW: Visualization 4: Correlation heatmap
# ----------------------------------------------------------------------
def plot_correlation_heatmap(query_data, output_path="correlation_heatmap.png"):
    """Create a heatmap showing correlation between tool count and F1 score for each building."""
    
    # Group data by building
    by_building = defaultdict(list)
    for query in query_data:
        by_building[query['building']].append(query)
    
    buildings = sorted(by_building.keys())
    
    # Calculate correlations
    correlations = []
    p_values = []
    sample_sizes = []
    
    for building in buildings:
        building_queries = by_building[building]
        tool_counts = [q['tool_count'] for q in building_queries]
        f1_scores = [q['row_f1'] for q in building_queries]
        
        sample_sizes.append(len(building_queries))
        
        # Calculate correlation if we have variation
        if len(set(tool_counts)) > 1 and len(tool_counts) > 1:
            corr, p_val = stats.pearsonr(tool_counts, f1_scores)
            correlations.append(corr)
            p_values.append(p_val)
        else:
            correlations.append(0.0)
            p_values.append(1.0)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(6, len(buildings) * 0.4)))
    
    # Subplot 1: Correlation values
    # Create 2D array for heatmap (single column)
    corr_array = np.array(correlations).reshape(-1, 1)
    
    im1 = ax1.imshow(corr_array, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    ax1.set_yticks(np.arange(len(buildings)))
    ax1.set_yticklabels(buildings)
    ax1.set_xticks([0])
    ax1.set_xticklabels(['Tool Count vs F1'])
    ax1.set_title('Pearson Correlation Coefficient', fontsize=12, fontweight='bold')
    
    # Add correlation values as text
    for i, (corr, p_val) in enumerate(zip(correlations, p_values)):
        color = 'white' if abs(corr) > 0.5 else 'black'
        significance = '*' if p_val < 0.05 else ''
        ax1.text(0, i, f'{corr:.3f}{significance}', 
                ha='center', va='center', color=color, fontsize=10)
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Correlation', rotation=270, labelpad=20, fontweight='bold')
    
    # Subplot 2: Sample sizes
    sample_array = np.array(sample_sizes).reshape(-1, 1)
    
    im2 = ax2.imshow(sample_array, cmap='Blues', aspect='auto')
    ax2.set_yticks(np.arange(len(buildings)))
    ax2.set_yticklabels(buildings)
    ax2.set_xticks([0])
    ax2.set_xticklabels(['Sample Size'])
    ax2.set_title('Number of Queries', fontsize=12, fontweight='bold')
    
    # Add sample size values as text
    for i, n in enumerate(sample_sizes):
        # Determine text color based on background
        bg_color = im2.cmap(im2.norm(n))
        luminance = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
        color = 'white' if luminance < 0.5 else 'black'
        ax2.text(0, i, f'{n}', ha='center', va='center', 
                color=color, fontsize=10)
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Count', rotation=270, labelpad=20, fontweight='bold')
    
    # Add note about significance
    fig.text(0.5, 0.02, '* indicates p < 0.05 (statistically significant)', 
            ha='center', fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Correlation heatmap saved to: {output_path}")
    plt.close()
    
    # Print summary statistics
    print("\nCorrelation Summary:")
    print("-" * 60)
    for building, corr, p_val, n in zip(buildings, correlations, p_values, sample_sizes):
        sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"{building:20s}: r={corr:+.3f} (p={p_val:.3f}{sig:2s}, n={n})")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize tool usage metrics"
    )
    parser.add_argument(
        "csv_file",
        help="CSV file to analyze"
    )
    parser.add_argument(
        "--min-tool-occurrences",
        type=int,
        default=5,
        help="Minimum occurrences for a tool to be included in analysis (default: 5)"
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to save output plots"
    )
    args = parser.parse_args()
    
    print(f"Loading data from {args.csv_file}...")
    data = load_csv_data(args.csv_file)
    print(f"  Loaded {len(data)} records")
    
    # Compute metrics
    print("\nComputing metrics...")
    building_stats = compute_tool_count_by_building(data)
    query_data = compute_tool_success_correlation(data)
    tool_impact = compute_tool_impact_stats(query_data, args.min_tool_occurrences)
    
    print(f"  Found {len(building_stats)} buildings")
    print(f"  Analyzed {len(tool_impact)} tools (min {args.min_tool_occurrences} occurrences)")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    plot_tool_impact(
        tool_impact,
        output_path=str(output_dir / "tool_impact.png")
    )
    
    plot_tools_by_building(
        building_stats,
        output_path=str(output_dir / "tools_by_building.png")
    )
    
    plot_tool_count_vs_performance(
        query_data,
        output_path=str(output_dir / "tool_count_scatter.png")
    )
    
    plot_correlation_heatmap(
        query_data,
        output_path=str(output_dir / "correlation_heatmap.png")
    )
    
    print("\nAll visualizations complete!")


if __name__ == "__main__":
    main()