#!/usr/bin/env python3
"""
Aggregate Performance Metrics Calculator

This script computes aggregate metrics for model performance based on:
- Token usage (prompt_tokens, completion_tokens, total_tokens)
- F1 scores (arity_matching_f1, exact_match_f1, entity_set_f1, row_matching_f1)
"""

import csv
import statistics
from collections import defaultdict
from pathlib import Path
import json


def load_csv_data(filepath):
    """Load CSV data and return list of dictionaries."""
    # Increase field size limit to handle large fields (like message_history)
    csv.field_size_limit(10 * 1024 * 1024)  # 10 MB
    
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def convert_to_numeric(value, default=0.0):
    """Convert string value to float, return default if conversion fails."""
    try:
        return float(value) if value and value.strip() else default
    except (ValueError, AttributeError):
        return default


def compute_aggregate_metrics(data):
    """Compute aggregate metrics from the data."""
    
    # Initialize metric collectors
    metrics = {
        'total_queries': len(data),
        'token_usage': {
            'prompt_tokens': [],
            'completion_tokens': [],
            'total_tokens': []
        },
        'f1_scores': {
            'arity_matching_f1': [],
            'exact_match_f1': [],
            'entity_set_f1': [],
            'row_matching_f1': []
        },
        'by_query_id': defaultdict(lambda: {
            'count': 0,
            'prompt_tokens': [],
            'completion_tokens': [],
            'total_tokens': [],
            'arity_matching_f1': [],
            'exact_match_f1': [],
            'entity_set_f1': [],
            'row_matching_f1': []
        }),
        'by_model': defaultdict(lambda: {
            'count': 0,
            'prompt_tokens': [],
            'completion_tokens': [],
            'total_tokens': [],
            'arity_matching_f1': [],
            'exact_match_f1': [],
            'entity_set_f1': [],
            'row_matching_f1': []
        })
    }
    
    # Collect data
    # Note: query_id is NOT unique - multiple rows can have the same query_id
    # We append all values to lists, so nothing is overwritten
    for row in data:
        query_id = row.get('query_id', 'unknown')
        model = row.get('model', 'unknown')
        
        # Token metrics
        prompt_tokens = convert_to_numeric(row.get('prompt_tokens'))
        completion_tokens = convert_to_numeric(row.get('completion_tokens'))
        total_tokens = convert_to_numeric(row.get('total_tokens'))
        
        # F1 scores
        arity_f1 = convert_to_numeric(row.get('arity_matching_f1'))
        exact_f1 = convert_to_numeric(row.get('exact_match_f1'))
        entity_f1 = convert_to_numeric(row.get('entity_set_f1'))
        row_f1 = convert_to_numeric(row.get('row_matching_f1'))
        
        # Overall metrics
        metrics['token_usage']['prompt_tokens'].append(prompt_tokens)
        metrics['token_usage']['completion_tokens'].append(completion_tokens)
        metrics['token_usage']['total_tokens'].append(total_tokens)
        
        metrics['f1_scores']['arity_matching_f1'].append(arity_f1)
        metrics['f1_scores']['exact_match_f1'].append(exact_f1)
        metrics['f1_scores']['entity_set_f1'].append(entity_f1)
        metrics['f1_scores']['row_matching_f1'].append(row_f1)
        
        # By query_id
        metrics['by_query_id'][query_id]['count'] += 1
        metrics['by_query_id'][query_id]['prompt_tokens'].append(prompt_tokens)
        metrics['by_query_id'][query_id]['completion_tokens'].append(completion_tokens)
        metrics['by_query_id'][query_id]['total_tokens'].append(total_tokens)
        metrics['by_query_id'][query_id]['arity_matching_f1'].append(arity_f1)
        metrics['by_query_id'][query_id]['exact_match_f1'].append(exact_f1)
        metrics['by_query_id'][query_id]['entity_set_f1'].append(entity_f1)
        metrics['by_query_id'][query_id]['row_matching_f1'].append(row_f1)
        
        # By model
        metrics['by_model'][model]['count'] += 1
        metrics['by_model'][model]['prompt_tokens'].append(prompt_tokens)
        metrics['by_model'][model]['completion_tokens'].append(completion_tokens)
        metrics['by_model'][model]['total_tokens'].append(total_tokens)
        metrics['by_model'][model]['arity_matching_f1'].append(arity_f1)
        metrics['by_model'][model]['exact_match_f1'].append(exact_f1)
        metrics['by_model'][model]['entity_set_f1'].append(entity_f1)
        metrics['by_model'][model]['row_matching_f1'].append(row_f1)
    
    return metrics


def compute_statistics(values):
    """Compute statistics for a list of values."""
    if not values:
        return {
            'mean': 0.0,
            'median': 0.0,
            'min': 0.0,
            'max': 0.0,
            'std_dev': 0.0,
            'sum': 0.0
        }
    
    return {
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'min': min(values),
        'max': max(values),
        'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
        'sum': sum(values)
    }


def format_report(metrics):
    """Format metrics into a readable report."""
    report = []
    
    report.append("=" * 80)
    report.append("AGGREGATE PERFORMANCE METRICS REPORT")
    report.append("=" * 80)
    report.append(f"\nTotal Queries Analyzed: {metrics['total_queries']}")
    
    # Overall Token Usage
    report.append("\n" + "=" * 80)
    report.append("OVERALL TOKEN USAGE")
    report.append("=" * 80)
    
    for token_type in ['prompt_tokens', 'completion_tokens', 'total_tokens']:
        stats = compute_statistics(metrics['token_usage'][token_type])
        report.append(f"\n{token_type.replace('_', ' ').title()}:")
        report.append(f"  Mean:     {stats['mean']:,.2f}")
        report.append(f"  Median:   {stats['median']:,.2f}")
        report.append(f"  Min:      {stats['min']:,.0f}")
        report.append(f"  Max:      {stats['max']:,.0f}")
        report.append(f"  Std Dev:  {stats['std_dev']:,.2f}")
        report.append(f"  Total:    {stats['sum']:,.0f}")
    
    # Overall F1 Scores
    report.append("\n" + "=" * 80)
    report.append("OVERALL F1 SCORES")
    report.append("=" * 80)
    
    for f1_type in ['arity_matching_f1', 'exact_match_f1', 'entity_set_f1', 'row_matching_f1']:
        stats = compute_statistics(metrics['f1_scores'][f1_type])
        report.append(f"\n{f1_type.replace('_', ' ').title()}:")
        report.append(f"  Mean:     {stats['mean']:.4f}")
        report.append(f"  Median:   {stats['median']:.4f}")
        report.append(f"  Min:      {stats['min']:.4f}")
        report.append(f"  Max:      {stats['max']:.4f}")
        report.append(f"  Std Dev:  {stats['std_dev']:.4f}")
    
    # By Query ID
    report.append("\n" + "=" * 80)
    report.append("METRICS BY QUERY ID")
    report.append("=" * 80)
    
    for query_id, data in sorted(metrics['by_query_id'].items()):
        report.append(f"\n{query_id} ({data['count']} queries):")
        
        # Token stats
        prompt_stats = compute_statistics(data['prompt_tokens'])
        completion_stats = compute_statistics(data['completion_tokens'])
        total_stats = compute_statistics(data['total_tokens'])
        
        report.append(f"  Tokens - Prompt (avg): {prompt_stats['mean']:,.1f}, "
                     f"Completion (avg): {completion_stats['mean']:,.1f}, "
                     f"Total (avg): {total_stats['mean']:,.1f}")
        
        # F1 stats
        arity_stats = compute_statistics(data['arity_matching_f1'])
        exact_stats = compute_statistics(data['exact_match_f1'])
        entity_stats = compute_statistics(data['entity_set_f1'])
        row_stats = compute_statistics(data['row_matching_f1'])
        
        report.append(f"  F1 Scores - Arity: {arity_stats['mean']:.3f}, "
                     f"Exact: {exact_stats['mean']:.3f}, "
                     f"Entity: {entity_stats['mean']:.3f}, "
                     f"Row: {row_stats['mean']:.3f}")
    
    # By Model
    report.append("\n" + "=" * 80)
    report.append("METRICS BY MODEL")
    report.append("=" * 80)
    
    for model, data in sorted(metrics['by_model'].items()):
        report.append(f"\n{model} ({data['count']} queries):")
        
        # Token stats
        prompt_stats = compute_statistics(data['prompt_tokens'])
        completion_stats = compute_statistics(data['completion_tokens'])
        total_stats = compute_statistics(data['total_tokens'])
        
        report.append(f"  Tokens - Prompt (avg): {prompt_stats['mean']:,.1f}, "
                     f"Completion (avg): {completion_stats['mean']:,.1f}, "
                     f"Total (avg): {total_stats['mean']:,.1f}")
        
        # F1 stats
        arity_stats = compute_statistics(data['arity_matching_f1'])
        exact_stats = compute_statistics(data['exact_match_f1'])
        entity_stats = compute_statistics(data['entity_set_f1'])
        row_stats = compute_statistics(data['row_matching_f1'])
        
        report.append(f"  F1 Scores - Arity: {arity_stats['mean']:.3f}, "
                     f"Exact: {exact_stats['mean']:.3f}, "
                     f"Entity: {entity_stats['mean']:.3f}, "
                     f"Row: {row_stats['mean']:.3f}")
    
    report.append("\n" + "=" * 80)
    
    return "\n".join(report)


def export_json(metrics, output_path):
    """Export metrics to JSON format."""
    json_data = {
        'total_queries': metrics['total_queries'],
        'overall': {
            'token_usage': {},
            'f1_scores': {}
        },
        'by_query_id': {},
        'by_model': {}
    }
    
    # Overall metrics
    for token_type in ['prompt_tokens', 'completion_tokens', 'total_tokens']:
        json_data['overall']['token_usage'][token_type] = compute_statistics(
            metrics['token_usage'][token_type]
        )
    
    for f1_type in ['arity_matching_f1', 'exact_match_f1', 'entity_set_f1', 'row_matching_f1']:
        json_data['overall']['f1_scores'][f1_type] = compute_statistics(
            metrics['f1_scores'][f1_type]
        )
    
    # By query ID
    for query_id, data in metrics['by_query_id'].items():
        json_data['by_query_id'][query_id] = {
            'count': data['count'],
            'token_usage': {
                'prompt_tokens': compute_statistics(data['prompt_tokens']),
                'completion_tokens': compute_statistics(data['completion_tokens']),
                'total_tokens': compute_statistics(data['total_tokens'])
            },
            'f1_scores': {
                'arity_matching_f1': compute_statistics(data['arity_matching_f1']),
                'exact_match_f1': compute_statistics(data['exact_match_f1']),
                'entity_set_f1': compute_statistics(data['entity_set_f1']),
                'row_matching_f1': compute_statistics(data['row_matching_f1'])
            }
        }
    
    # By model
    for model, data in metrics['by_model'].items():
        json_data['by_model'][model] = {
            'count': data['count'],
            'token_usage': {
                'prompt_tokens': compute_statistics(data['prompt_tokens']),
                'completion_tokens': compute_statistics(data['completion_tokens']),
                'total_tokens': compute_statistics(data['total_tokens'])
            },
            'f1_scores': {
                'arity_matching_f1': compute_statistics(data['arity_matching_f1']),
                'exact_match_f1': compute_statistics(data['exact_match_f1']),
                'entity_set_f1': compute_statistics(data['entity_set_f1']),
                'row_matching_f1': compute_statistics(data['row_matching_f1'])
            }
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    
    return output_path


def main():
    """Main function to run the analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compute aggregate performance metrics from CSV data'
    )
    parser.add_argument(
        'input_file',
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--output-json',
        help='Path to output JSON file (optional)',
        default=None
    )
    parser.add_argument(
        '--output-txt',
        help='Path to output text report file (optional)',
        default=None
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_csv_data(args.input_file)
    print(f"Loaded {len(data)} records")
    
    # Compute metrics
    print("Computing aggregate metrics...")
    metrics = compute_aggregate_metrics(data)
    
    # Generate report
    report = format_report(metrics)
    print("\n" + report)
    
    # Save text report if requested
    if args.output_txt:
        with open(args.output_txt, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nText report saved to: {args.output_txt}")
    
    # Export JSON if requested
    if args.output_json:
        json_path = export_json(metrics, args.output_json)
        print(f"JSON metrics saved to: {json_path}")


if __name__ == '__main__':
    main()
