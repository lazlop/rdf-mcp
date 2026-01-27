#!/usr/bin/env python3
"""
Visualize Tool Usage Metrics Across Multiple CSV Files

This script creates visualizations comparing tool usage patterns across multiple
CSV files, including:
- Bar charts comparing tool frequency by building
- Weighted average row F1 for each tool
- Kernel density plots for tool count distributions
- Average tool usage by building
"""

import csv
import re
import statistics
from collections import defaultdict, Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import sys
from scipy import stats

# Increase CSV field size limit
csv.field_size_limit(sys.maxsize)


# ----------------------------------------------------------------------
# Tool extraction (from your tool-counting code)
# ----------------------------------------------------------------------
def extract_tool_calls(message_history: str) -> List[str]:
    """Extract tool names from ToolCallPart entries in the message history."""
    tool_names = []
    pattern = r"ToolCallPart\(tool_name='([^']+)'"
    matches = re.findall(pattern, message_history)
    tool_names.extend(matches)
    return tool_names


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
# Tool usage metrics computation
# ----------------------------------------------------------------------
def compute_tool_metrics_by_building(data):
    """Compute tool usage metrics grouped by building."""
    by_building = defaultdict(lambda: {
        "tool_counter": Counter(),
        "query_count": 0,
        "total_tool_calls": 0,
        "queries_with_tools": 0
    })
    
    for row in data:
        query_id = row.get("query_id", "unknown")
        building = query_id.split("_")[0] if "_" in query_id else query_id
        message_history = row.get("message_history", "")
        
        tools = extract_tool_calls(message_history)
        # Filter out final_result
        tools = [t for t in tools if t != "final_result"]
        
        bd = by_building[building]
        bd["query_count"] += 1
        bd["tool_counter"].update(tools)
        bd["total_tool_calls"] += len(tools)
        if tools:
            bd["queries_with_tools"] += 1
    
    # Compute statistics
    building_stats = {}
    for building, metrics in by_building.items():
        building_stats[building] = {
            "tool_frequencies": dict(metrics["tool_counter"]),
            "query_count": metrics["query_count"],
            "total_tool_calls": metrics["total_tool_calls"],
            "queries_with_tools": metrics["queries_with_tools"],
            "avg_tools_per_query": metrics["total_tool_calls"] / metrics["query_count"] 
                                   if metrics["query_count"] > 0 else 0.0,
            "tool_usage_rate": metrics["queries_with_tools"] / metrics["query_count"] 
                              if metrics["query_count"] > 0 else 0.0
        }
    
    return building_stats


def compute_weighted_f1_by_tool(data):
    """
    Compute weighted average row F1 for each tool.
    Weight is based on how many times each tool appears.
    """
    tool_f1_data = defaultdict(lambda: {"f1_scores": [], "weights": []})
    
    for row in data:
        message_history = row.get("message_history", "")
        row_f1 = convert_to_numeric(row.get("row_matching_f1", 0))
        
        tools = extract_tool_calls(message_history)
        # Filter out final_result
        tools = [t for t in tools if t != "final_result"]
        
        # Count occurrences of each tool in this query
        tool_counts = Counter(tools)
        
        # Add F1 score for each tool with weight based on frequency
        for tool, count in tool_counts.items():
            tool_f1_data[tool]["f1_scores"].append(row_f1)
            tool_f1_data[tool]["weights"].append(count)
    
    # Compute weighted averages
    tool_weighted_f1 = {}
    for tool, data_dict in tool_f1_data.items():
        f1_scores = np.array(data_dict["f1_scores"])
        weights = np.array(data_dict["weights"])
        weighted_avg = np.average(f1_scores, weights=weights)
        tool_weighted_f1[tool] = {
            "weighted_avg_f1": weighted_avg,
            "total_occurrences": sum(weights),
            "num_queries": len(f1_scores)
        }
    
    return tool_weighted_f1


def compute_tool_count_distribution(data):
    """Compute distribution of tool counts per query."""
    tool_counts = []
    
    for row in data:
        message_history = row.get("message_history", "")
        tools = extract_tool_calls(message_history)
        # Filter out final_result
        tools = [t for t in tools if t != "final_result"]
        tool_counts.append(len(tools))
    
    return tool_counts


# ----------------------------------------------------------------------
# Visualization 1: Tool frequency by building
# ----------------------------------------------------------------------
def plot_tool_frequency_by_building(
    baseline_dict: Dict[str, dict],
    test_dict: Dict[str, dict],
    top_n_tools: int = 10,
    output_path: str = "tool_frequency_by_building.png"
):
    """Create bar chart comparing tool frequencies across buildings."""
    
    # Get all unique buildings
    all_buildings = set()
    for stats in list(baseline_dict.values()) + list(test_dict.values()):
        all_buildings.update(stats.keys())
    buildings = sorted(all_buildings)
    
    # Get top N most common tools across all data (excluding final_result)
    all_tools = Counter()
    for stats_dict in [baseline_dict, test_dict]:
        for building_stats in stats_dict.values():
            for building, metrics in building_stats.items():
                all_tools.update(metrics["tool_frequencies"])
    
    top_tools = [tool for tool, _ in all_tools.most_common(top_n_tools)]
    
    # Create subplots for each dataset
    n_datasets = len(baseline_dict) + len(test_dict)
    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 5), sharey=True)
    if n_datasets == 1:
        axes = [axes]
    
    ax_idx = 0
    
    # Baseline colors
    hatches = ['///', '\\\\\\', '|||', '---']
    
    # Color palette for tools (consistent across all plots)
    tool_colors = plt.cm.tab20(np.linspace(0, 1, len(top_tools)))
    
    # Plot baselines
    for i, (name, building_stats) in enumerate(baseline_dict.items()):
        ax = axes[ax_idx]
        
        # Prepare data matrix: buildings × tools
        data_matrix = []
        for building in buildings:
            if building in building_stats:
                frequencies = building_stats[building]["tool_frequencies"]
                row = [frequencies.get(tool, 0) for tool in top_tools]
            else:
                row = [0] * len(top_tools)
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix).T  # Transpose: tools × buildings
        
        # Stacked bar chart
        x = np.arange(len(buildings))
        bottom = np.zeros(len(buildings))
        
        hatch = hatches[i % len(hatches)]
        
        for tool_idx, tool in enumerate(top_tools):
            # Only add label if this is the first subplot
            label = tool if ax_idx == 0 else None
            ax.bar(x, data_matrix[tool_idx], width=0.6, bottom=bottom,
                   label=label,
                   color=tool_colors[tool_idx],
                   alpha=0.7, edgecolor='black', linewidth=0.5, hatch=hatch)
            bottom += data_matrix[tool_idx]
        
        ax.set_xlabel('Building', fontsize=11, fontweight='bold')
        # Only set y-label for the first (leftmost) subplot
        if ax_idx == 0:
            ax.set_ylabel('Tool Call Count', fontsize=11, fontweight='bold')
        ax.set_title(f'{name} (baseline)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(buildings, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        ax_idx += 1
    
    # Plot test runs
    for i, (name, building_stats) in enumerate(test_dict.items()):
        ax = axes[ax_idx]
        
        # Prepare data matrix
        data_matrix = []
        for building in buildings:
            if building in building_stats:
                frequencies = building_stats[building]["tool_frequencies"]
                row = [frequencies.get(tool, 0) for tool in top_tools]
            else:
                row = [0] * len(top_tools)
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix).T
        
        x = np.arange(len(buildings))
        bottom = np.zeros(len(buildings))
        
        for tool_idx, tool in enumerate(top_tools):
            # Only add label if this is the first subplot
            label = tool if ax_idx == 0 else None
            ax.bar(x, data_matrix[tool_idx], width=0.6, bottom=bottom,
                   label=label,
                   color=tool_colors[tool_idx],
                   alpha=0.8, edgecolor='black', linewidth=0.5)
            bottom += data_matrix[tool_idx]
        
        ax.set_xlabel('Building', fontsize=11, fontweight='bold')
        # Y-label not needed for subsequent subplots due to sharey=True
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(buildings, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        ax_idx += 1
    
    # Add a single legend for all subplots
    if n_datasets > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5),
                  fontsize=10, title='Tools')
    
    # fig.suptitle('Tool Usage Frequency by Building', 
    #              fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Tool frequency by building plot saved to: {output_path}")
    plt.close()

# ----------------------------------------------------------------------
# Visualization 2: Weighted average row F1 by tool
# ----------------------------------------------------------------------
def plot_weighted_f1_by_tool(
    baseline_f1_dict: Dict[str, dict],
    test_f1_dict: Dict[str, dict],
    output_path: str = "weighted_f1_by_tool.png"
):
    """Create bar chart comparing weighted average F1 scores by tool."""
    
    # Get all unique tools across all datasets
    all_tools = set()
    for f1_data in list(baseline_f1_dict.values()) + list(test_f1_dict.values()):
        all_tools.update(f1_data.keys())
    
    # Sort tools by average weighted F1 across all datasets
    tool_avg_f1 = {}
    for tool in all_tools:
        f1_values = []
        for f1_data in list(baseline_f1_dict.values()) + list(test_f1_dict.values()):
            if tool in f1_data:
                f1_values.append(f1_data[tool]["weighted_avg_f1"])
        tool_avg_f1[tool] = np.mean(f1_values) if f1_values else 0
    
    # Get top tools by F1
    top_tools = sorted(all_tools, key=lambda t: tool_avg_f1[t], reverse=True)[:15]
    
    # Create plot
    n_datasets = len(baseline_f1_dict) + len(test_f1_dict)
    x = np.arange(len(top_tools))
    width = 0.8 / n_datasets
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Baseline colors and patterns
    baseline_colors = ['#A9A9A9', '#808080', '#696969', '#5C5C5C']
    hatches = ['///', '\\\\\\', '|||', '---']
    
    # Test colors
    test_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    bar_idx = 0
    
    # Plot baselines
    for i, (name, f1_data) in enumerate(baseline_f1_dict.items()):
        f1_values = [f1_data[tool]["weighted_avg_f1"] if tool in f1_data else 0 
                     for tool in top_tools]
        
        offset = width * (bar_idx - n_datasets / 2 + 0.5)
        color = baseline_colors[i % len(baseline_colors)]
        hatch = hatches[i % len(hatches)]
        ax.bar(x + offset, f1_values, width, label=f"{name} (baseline)",
               alpha=0.7, color=color, hatch=hatch, edgecolor='black', linewidth=0.5)
        bar_idx += 1
    
    # Plot test runs
    for i, (name, f1_data) in enumerate(test_f1_dict.items()):
        f1_values = [f1_data[tool]["weighted_avg_f1"] if tool in f1_data else 0 
                     for tool in top_tools]
        
        offset = width * (bar_idx - n_datasets / 2 + 0.5)
        color = test_colors[i % len(test_colors)]
        ax.bar(x + offset, f1_values, width, label=name,
               alpha=0.8, color=color, edgecolor='black', linewidth=0.5)
        bar_idx += 1
    
    ax.set_xlabel('Tool', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weighted Average Row F1', fontsize=12, fontweight='bold')
    ax.set_title('Weighted Average Row F1 Score by Tool', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(top_tools, rotation=45, ha='right')
    ax.legend(fontsize=9, loc='best')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Weighted F1 by tool plot saved to: {output_path}")
    plt.close()


# ----------------------------------------------------------------------
# Visualization 3: Tool count distribution (KDE)
# ----------------------------------------------------------------------
def plot_tool_count_distribution(
    baseline_dist_dict: Dict[str, List[int]],
    test_dist_dict: Dict[str, List[int]],
    output_path: str = "tool_count_distribution.png"
):
    """Create kernel density plot for tool count distributions."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Baseline colors and styles
    baseline_colors = ['#A9A9A9', '#808080', '#696969', '#5C5C5C']
    baseline_styles = ['--', '-.', ':', (0, (3, 1, 1, 1))]
    
    # Test colors
    test_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Plot baselines
    for i, (name, tool_counts) in enumerate(baseline_dist_dict.items()):
        if len(tool_counts) > 1:
            # Create KDE
            kde = stats.gaussian_kde(tool_counts)
            x_range = np.linspace(0, max(tool_counts), 200)
            density = kde(x_range)
            
            color = baseline_colors[i % len(baseline_colors)]
            style = baseline_styles[i % len(baseline_styles)]
            ax.plot(x_range, density, label=f"{name} (baseline)", 
                   color=color, linestyle=style, linewidth=2.5, alpha=0.8)
    
    # Plot test runs
    for i, (name, tool_counts) in enumerate(test_dist_dict.items()):
        if len(tool_counts) > 1:
            # Create KDE
            kde = stats.gaussian_kde(tool_counts)
            x_range = np.linspace(0, max(tool_counts), 200)
            density = kde(x_range)
            
            color = test_colors[i % len(test_colors)]
            ax.plot(x_range, density, label=name, 
                   color=color, linewidth=2, alpha=0.9)
    
    ax.set_xlabel('Number of Tools per Query', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('Tool Count Distribution (Kernel Density Estimate)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(axis='both', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Tool count distribution plot saved to: {output_path}")
    plt.close()


# ----------------------------------------------------------------------
# Visualization 4: Average tools per query by building
# ----------------------------------------------------------------------
def plot_avg_tools_by_building(
    baseline_dict: Dict[str, dict],
    test_dict: Dict[str, dict],
    output_path: str = "avg_tools_by_building.png"
):
    """Create bar chart of average tool calls per query by building."""
    
    # Get all unique buildings
    all_buildings = set()
    for stats in list(baseline_dict.values()) + list(test_dict.values()):
        all_buildings.update(stats.keys())
    buildings = sorted(all_buildings)
    
    x = np.arange(len(buildings))
    total_bars = len(baseline_dict) + len(test_dict)
    width = 0.8 / total_bars
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Baseline colors and patterns
    baseline_colors = ['#A9A9A9', '#808080', '#696969', '#5C5C5C']
    hatches = ['///', '\\\\\\', '|||', '---']
    
    # Test colors
    test_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    bar_idx = 0
    
    # Plot baselines
    for i, (name, building_stats) in enumerate(baseline_dict.items()):
        avgs = []
        for building in buildings:
            if building in building_stats:
                avgs.append(building_stats[building]["avg_tools_per_query"])
            else:
                avgs.append(0.0)
        
        offset = width * (bar_idx - total_bars / 2 + 0.5)
        color = baseline_colors[i % len(baseline_colors)]
        hatch = hatches[i % len(hatches)]
        ax.bar(x + offset, avgs, width, label=f"{name} (baseline)",
               alpha=0.7, color=color, hatch=hatch, edgecolor='black', linewidth=0.5)
        bar_idx += 1
    
    # Plot test runs
    for i, (name, building_stats) in enumerate(test_dict.items()):
        avgs = []
        for building in buildings:
            if building in building_stats:
                avgs.append(building_stats[building]["avg_tools_per_query"])
            else:
                avgs.append(0.0)
        
        offset = width * (bar_idx - total_bars / 2 + 0.5)
        color = test_colors[i % len(test_colors)]
        ax.bar(x + offset, avgs, width, label=name,
               alpha=0.8, color=color, edgecolor='black', linewidth=0.5)
        bar_idx += 1
    
    ax.set_xlabel('Building', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Tools Per Query', fontsize=12, fontweight='bold')
    ax.set_title('Average Tool Calls Per Query by Building', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(buildings, rotation=45, ha='right')
    ax.legend(fontsize=9, loc='best')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Average tools by building plot saved to: {output_path}")
    plt.close()

# Add this new function for tool-specific success correlation

def plot_tool_success_correlation(
    baseline_corr_dict: Dict[str, List[dict]],
    test_corr_dict: Dict[str, List[dict]],
    min_occurrences: int = 5,
    output_path: str = "tool_success_correlation.png"
):
    """
    Create visualization showing correlation between specific tools and success rates.
    
    For each tool, shows:
    - Mean F1 score when tool is used vs when it's not used
    - Success rate (% queries with F1 >= 0.5) when tool is used
    """
    
    # Collect tool usage statistics across all datasets
    all_stats = {}
    
    for name, query_data in list(baseline_corr_dict.items()) + list(test_corr_dict.items()):
        # Track metrics for each tool
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
                
                # Success rate when tool is used
                success_count = sum(1 for f1 in stats['with_tool_f1'] if f1 >= 0.5)
                success_rate = success_count / len(stats['with_tool_f1']) if stats['with_tool_f1'] else 0
                
                tool_metrics[tool] = {
                    'with_f1': with_f1,
                    'without_f1': without_f1,
                    'delta_f1': with_f1 - without_f1,
                    'success_rate': success_rate,
                    'usage_count': stats['usage_count']
                }
        
        all_stats[name] = tool_metrics
    
    # Create visualization
    n_datasets = len(all_stats)
    fig, axes = plt.subplots(2, n_datasets, figsize=(7 * n_datasets, 10))
    if n_datasets == 1:
        axes = axes.reshape(2, 1)
    
    baseline_names = list(baseline_corr_dict.keys())
    
    for idx, (name, tool_metrics) in enumerate(all_stats.items()):
        # Sort tools by delta F1 (impact on performance)
        sorted_tools = sorted(tool_metrics.items(), 
                            key=lambda x: x[1]['delta_f1'], 
                            reverse=True)
        
        if not sorted_tools:
            continue
        
        tools = [t[0] for t in sorted_tools]
        delta_f1s = [t[1]['delta_f1'] for t in sorted_tools]
        success_rates = [t[1]['success_rate'] for t in sorted_tools]
        usage_counts = [t[1]['usage_count'] for t in sorted_tools]
        
        # Determine if baseline or test
        is_baseline = name in baseline_names
        
        # Plot 1: Delta F1 (impact on performance)
        ax1 = axes[0, idx]
        colors = ['#2ca02c' if d > 0 else '#d62728' for d in delta_f1s]
        bars = ax1.barh(tools, delta_f1s, color=colors, alpha=0.7, 
                       edgecolor='black', linewidth=0.5)
        
        if is_baseline:
            for bar in bars:
                bar.set_hatch('///')
        
        ax1.axvline(x=0, color='black', linewidth=1, linestyle='-', alpha=0.3)
        ax1.set_xlabel('F1 Score Impact\n(With Tool - Without Tool)', 
                      fontsize=10, fontweight='bold')
        ax1.set_ylabel('Tool', fontsize=10, fontweight='bold')
        title = f'{name} (baseline)' if is_baseline else name
        ax1.set_title(f'{title}\nTool Impact on Performance', 
                     fontsize=11, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (tool, delta) in enumerate(zip(tools, delta_f1s)):
            x_pos = delta + (0.01 if delta > 0 else -0.01)
            ha = 'left' if delta > 0 else 'right'
            ax1.text(x_pos, i, f'{delta:+.3f}', 
                    va='center', ha=ha, fontsize=8)
        
        # Plot 2: Success rate when tool is used
        ax2 = axes[1, idx]
        
        # Color by success rate
        colors = plt.cm.RdYlGn(np.array(success_rates))
        bars = ax2.barh(tools, success_rates, color=colors, alpha=0.7,
                       edgecolor='black', linewidth=0.5)
        
        if is_baseline:
            for bar in bars:
                bar.set_hatch('///')
        
        ax2.set_xlabel('Success Rate (F1 ≥ 0.5)', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Tool', fontsize=10, fontweight='bold')
        ax2.set_title(f'{title}\nSuccess Rate When Tool Used', 
                     fontsize=11, fontweight='bold')
        ax2.set_xlim(0, 1)
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels and usage count
        for i, (tool, rate, count) in enumerate(zip(tools, success_rates, usage_counts)):
            ax2.text(rate + 0.02, i, f'{rate:.1%} (n={count})', 
                    va='center', ha='left', fontsize=8)
    
    fig.suptitle('Tool Usage Correlation with Query Success', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Tool success correlation plot saved to: {output_path}")
    plt.close()


# Add this new function for building-level tool count vs success analysis

def plot_tool_count_success_by_building(
    baseline_corr_dict: Dict[str, List[dict]],
    test_corr_dict: Dict[str, List[dict]],
    output_path: str = "tool_count_success_by_building.png"
):
    """
    Analyze relationship between number of tool calls and success by building.
    
    Shows scatter plots with trend lines for each building, indicating whether
    more tool usage correlates with better or worse outcomes.
    """
    
    # Collect data by building for each dataset
    all_building_data = {}
    
    for name, query_data in list(baseline_corr_dict.items()) + list(test_corr_dict.items()):
        building_data = defaultdict(lambda: {
            'tool_counts': [],
            'f1_scores': [],
            'success_flags': []
        })
        
        for query in query_data:
            # Extract building from query (assuming it's embedded in the tools or we need to add it)
            # For now, we'll need to pass building info. Let's modify the correlation data structure
            # to include building information. We'll update this in the compute function.
            pass
        
        all_building_data[name] = building_data
    
    # Since we need building info, let's create a modified version that extracts it
    # This requires updating compute_tool_token_correlation
    
    print("Note: Building-level analysis requires building information in query data.")
    print("This will be implemented with updated data structure.")


# Update the compute_tool_token_correlation function to include building info:

def compute_tool_token_correlation(data):
    """Compute per-query tool counts and token counts for correlation analysis."""
    query_data = []
    
    for row in data:
        query_id = row.get("query_id", "unknown")
        building = query_id.split("_")[0] if "_" in query_id else "unknown"
        message_history = row.get("message_history", "")
        total_tokens = convert_to_numeric(row.get("total_tokens", 0))
        row_f1 = convert_to_numeric(row.get("row_matching_f1", 0))
        
        tools = extract_tool_calls(message_history)
        tool_count = len(tools)
        
        query_data.append({
            "building": building,
            "tool_count": tool_count,
            "total_tokens": total_tokens,
            "row_f1": row_f1,
            "tools": tools
        })
    
    return query_data


# Now add the complete building-level analysis function:

def plot_tool_count_success_by_building(
    baseline_corr_dict: Dict[str, List[dict]],
    test_corr_dict: Dict[str, List[dict]],
    output_path: str = "tool_count_success_by_building.png"
):
    """
    Analyze relationship between number of tool calls and success by building.
    
    For each building, shows:
    - Mean tool count for successful vs unsuccessful queries
    - Correlation between tool count and F1 score
    """
    
    # Collect data by building for each dataset
    all_dataset_stats = {}
    
    for name, query_data in list(baseline_corr_dict.items()) + list(test_corr_dict.items()):
        building_data = defaultdict(lambda: {
            'successful_tool_counts': [],
            'unsuccessful_tool_counts': [],
            'all_tool_counts': [],
            'all_f1_scores': []
        })
        
        for query in query_data:
            building = query['building']
            tool_count = query['tool_count']
            f1_score = query['row_f1']
            
            building_data[building]['all_tool_counts'].append(tool_count)
            building_data[building]['all_f1_scores'].append(f1_score)
            
            if f1_score >= 0.5:
                building_data[building]['successful_tool_counts'].append(tool_count)
            else:
                building_data[building]['unsuccessful_tool_counts'].append(tool_count)
        
        # Compute statistics
        building_stats = {}
        for building, data in building_data.items():
            if len(data['all_tool_counts']) > 0:
                # Compute correlation
                if len(data['all_tool_counts']) > 1:
                    correlation = np.corrcoef(data['all_tool_counts'], data['all_f1_scores'])[0, 1]
                else:
                    correlation = 0.0
                
                building_stats[building] = {
                    'successful_mean': statistics.mean(data['successful_tool_counts']) 
                                      if data['successful_tool_counts'] else 0,
                    'unsuccessful_mean': statistics.mean(data['unsuccessful_tool_counts']) 
                                        if data['unsuccessful_tool_counts'] else 0,
                    'correlation': correlation,
                    'n_successful': len(data['successful_tool_counts']),
                    'n_unsuccessful': len(data['unsuccessful_tool_counts'])
                }
        
        all_dataset_stats[name] = building_stats
    
    # Get all unique buildings
    all_buildings = set()
    for stats in all_dataset_stats.values():
        all_buildings.update(stats.keys())
    buildings = sorted(all_buildings)
    
    # Create visualization
    n_datasets = len(all_dataset_stats)
    fig, axes = plt.subplots(2, n_datasets, figsize=(7 * n_datasets, 10))
    if n_datasets == 1:
        axes = axes.reshape(2, 1)
    
    baseline_names = list(baseline_corr_dict.keys())
    
    for idx, (name, building_stats) in enumerate(all_dataset_stats.items()):
        is_baseline = name in baseline_names
        
        # Plot 1: Mean tool count for successful vs unsuccessful
        ax1 = axes[0, idx]
        
        successful_means = []
        unsuccessful_means = []
        for building in buildings:
            if building in building_stats:
                successful_means.append(building_stats[building]['successful_mean'])
                unsuccessful_means.append(building_stats[building]['unsuccessful_mean'])
            else:
                successful_means.append(0)
                unsuccessful_means.append(0)
        
        x = np.arange(len(buildings))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, successful_means, width, 
                       label='Successful (F1 ≥ 0.5)',
                       alpha=0.7, color='#2ca02c', edgecolor='black', linewidth=0.5)
        bars2 = ax1.bar(x + width/2, unsuccessful_means, width, 
                       label='Unsuccessful (F1 < 0.5)',
                       alpha=0.7, color='#d62728', edgecolor='black', linewidth=0.5)
        
        if is_baseline:
            for bar in bars1:
                bar.set_hatch('///')
            for bar in bars2:
                bar.set_hatch('///')
        
        ax1.set_xlabel('Building', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Mean Tool Calls', fontsize=10, fontweight='bold')
        title = f'{name} (baseline)' if is_baseline else name
        ax1.set_title(f'{title}\nTool Usage: Success vs Failure', 
                     fontsize=11, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(buildings, rotation=45, ha='right')
        ax1.legend(fontsize=9)
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Correlation between tool count and F1 score
        ax2 = axes[1, idx]
        
        correlations = []
        for building in buildings:
            if building in building_stats:
                correlations.append(building_stats[building]['correlation'])
            else:
                correlations.append(0)
        
        # Color by correlation strength
        colors = ['#2ca02c' if c > 0.1 else '#d62728' if c < -0.1 else '#808080' 
                 for c in correlations]
        bars = ax2.bar(x, correlations, color=colors, alpha=0.7,
                      edgecolor='black', linewidth=0.5)
        
        if is_baseline:
            for bar in bars:
                bar.set_hatch('///')
        
        ax2.axhline(y=0, color='black', linewidth=1, linestyle='-', alpha=0.3)
        ax2.axhline(y=0.1, color='green', linewidth=0.5, linestyle='--', alpha=0.3)
        ax2.axhline(y=-0.1, color='red', linewidth=0.5, linestyle='--', alpha=0.3)
        
        ax2.set_xlabel('Building', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Correlation (Tool Count vs F1)', fontsize=10, fontweight='bold')
        ax2.set_title(f'{title}\nCorrelation: More Tools = Better?', 
                     fontsize=11, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(buildings, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(-1, 1)
        
        # Add value labels
        for i, corr in enumerate(correlations):
            if abs(corr) > 0.05:
                y_pos = corr + (0.05 if corr > 0 else -0.05)
                ax2.text(i, y_pos, f'{corr:.2f}', 
                        ha='center', va='bottom' if corr > 0 else 'top', fontsize=8)
    
    fig.suptitle('Tool Usage vs Success by Building', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Tool count success by building plot saved to: {output_path}")
    plt.close()


# Update the main() function to include the new plots:

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize tool usage metrics across multiple CSV files"
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
        "--success-threshold",
        type=float,
        default=0.5,
        help="F1 score threshold for classifying queries as successful (default: 0.5)"
    )
    parser.add_argument(
        "--min-tool-occurrences",
        type=int,
        default=5,
        help="Minimum occurrences for a tool to be included in correlation analysis (default: 5)"
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
    baseline_building_dict = {}
    baseline_success_dict = {}
    baseline_corr_dict = {}
    
    for csv_arg in args.baseline:
        if ':' in csv_arg:
            name, path = csv_arg.split(':', 1)
        else:
            path = csv_arg
            name = Path(path).stem
        
        print(f"Loading baseline {name} from {path}...")
        data = load_csv_data(path)
        
        building_stats = compute_tool_metrics_by_building(data)
        success_stats = compute_tool_metrics_by_success(data, args.success_threshold)
        corr_data = compute_tool_token_correlation(data)
        
        baseline_building_dict[name] = building_stats
        baseline_success_dict[name] = success_stats
        baseline_corr_dict[name] = corr_data
        
        print(f"  Loaded {len(data)} records, {len(building_stats)} buildings")
    
    # Parse test CSV files
    test_building_dict = {}
    test_success_dict = {}
    test_corr_dict = {}
    
    for csv_arg in args.test:
        if ':' in csv_arg:
            name, path = csv_arg.split(':', 1)
        else:
            path = csv_arg
            name = Path(path).stem
        
        print(f"Loading test {name} from {path}...")
        data = load_csv_data(path)
        
        building_stats = compute_tool_metrics_by_building(data)
        corr_data = compute_tool_token_correlation(data)
        
        test_building_dict[name] = building_stats
        test_corr_dict[name] = corr_data
        
        print(f"  Loaded {len(data)} records, {len(building_stats)} buildings")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate visualizations
    print("\nGenerating tool usage visualizations...")
    
    plot_tool_frequency_by_building(
        baseline_building_dict,
        test_building_dict,
        output_path=str(output_dir / "tool_frequency_by_building.png")
    )
    
    plot_avg_tools_by_building(
        baseline_building_dict,
        test_building_dict,
        output_path=str(output_dir / "avg_tools_by_building.png")
    )
    
    plot_tool_success_correlation(
        baseline_corr_dict,
        test_corr_dict,
        min_occurrences=args.min_tool_occurrences,
        output_path=str(output_dir / "tool_success_correlation.png")
    )
    
    plot_tool_count_success_by_building(
        baseline_corr_dict,
        test_corr_dict,
        output_path=str(output_dir / "tool_count_success_by_building.png")
    )
    
    print("\nAll tool usage visualizations complete!")


if __name__ == "__main__":
    main()