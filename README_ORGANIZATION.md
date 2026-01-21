# Project Organization

This document describes the reorganized file structure for the RDF-MCP benchmark project.

## Directory Structure

```
rdf-mcp/
├── scripts/              # Executable scripts
│   └── run_benchmark_and_analyze.py  # Main workflow script
├── results/              # Benchmark results and analysis outputs
│   ├── *.csv            # Benchmark run results
│   ├── *_metrics.json   # Aggregate metrics
│   └── *_report.txt     # Text reports
├── configs/              # Configuration files
│   ├── benchmark-config.json
│   └── analysis-config.json
├── data/                 # Data files (if needed)
├── tests/                # Test files
├── archive/              # Archived old results
└── development/          # Development documentation

## Core Files (Root Directory)

- `benchmark.py` - Main benchmark runner
- `analyze_failures.py` - Failure analysis tool
- `aggregate_metrics.py` - Metrics aggregation tool
- `agent.py` - Agent implementation
- `metrics.py` - Metrics calculation
- `kgqa.py` - Knowledge graph QA functionality
- `utils.py` - Utility functions
- `namespaces.py` - RDF namespace definitions

## Usage

### Running Complete Benchmark Workflow

```bash
# Run benchmark only
python scripts/run_benchmark_and_analyze.py

# Run benchmark with failure analysis
python scripts/run_benchmark_and_analyze.py --analyze-failures

# Customize failure analysis
python scripts/run_benchmark_and_analyze.py \
    --analyze-failures \
    --max-message-chars 8000 \
    --threshold 0.9
```

### Running Individual Components

```bash
# Run benchmark only
python benchmark.py

# Analyze failures from existing results
python analyze_failures.py results/benchmark_run_YYYYMMDD_HHMMSS.csv

# Compute aggregate metrics
python aggregate_metrics.py results/benchmark_run_YYYYMMDD_HHMMSS.csv \
    --output-json results/metrics.json \
    --output-txt results/metrics_report.txt
```

## Output Files

All benchmark outputs are stored in the `results/` directory:

- `*_run_*.csv` - Raw benchmark results with all metrics
- `*_metrics.json` - Aggregated metrics in JSON format
- `*_metrics_report.txt` - Human-readable metrics report
- `*_failure_analysis.csv` - Failure classification results

## Configuration

Configuration files are stored in `configs/`:

- `benchmark-config.json` - Benchmark settings (model, API keys, data paths)
- `analysis-config.json` - Failure analysis settings (model, API keys)

## Development

See `development/` directory for:
- Example scripts
- Documentation
- Development notes
