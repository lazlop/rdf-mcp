"""
Script to run SPARQL query generation agent on all benchmark QA pairs.
Processes all JSON files in Benchmark_QA_pairs directory against their corresponding TTL files.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Import agent and utilities
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.agent import SimpleSparqlAgentMCP
# from agents.react_agent import SimpleSparqlAgentMCP

from scripts.utils import CsvLogger

# Load benchmark configuration
config_path = Path("../configs/benchmark-config.json")
if not config_path.is_file():
    print("Configuration file 'benchmark-config.json' not found.", file=sys.stderr)
    sys.exit(1)

config = json.load(open(config_path))
EXCLUDE_BUILDINGS = config.get("exclude-buildings", [])

# Configuration - update these with your actual values
MODEL_NAME = config.get("models", [None])[0]
API_KEY = config.get("api-key")
BASE_URL = config.get("base-url")
RESULTS_DIR = Path(config.get("results-dir", "../results"))

# Directory paths
BENCHMARK_DIR = Path(config.get("buildingqa-dir")) / "Benchmark_QA_pairs"
# EVAL_BUILDINGS_DIR = Path(config.get("buildingqa-dir")) / "eval_buildings"
EVAL_BUILDINGS_DIR = Path(config.get("buildingqa-dir")) / "bschema" / "without-ontology"

# Standard prefixes for SPARQL queries
STANDARD_PREFIXES = """
PREFIX brick: <https://brickschema.org/schema/Brick#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2001/XMLSchema#>
"""

def create_timestamped_logger(base_name: str = "sparql_agent_run") -> tuple[CsvLogger, str]:
    """Create a logger with timestamp in filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{base_name}_{timestamp}.csv"
    logger = CsvLogger(RESULTS_DIR / log_filename)
    print(f"Created logger: {log_filename}")
    return logger, log_filename


def load_json_file(filepath: Path) -> Dict:
    """Load and parse JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def get_ttl_path(building_id: str) -> Path:
    """Get the corresponding TTL file path for a building ID."""
    ttl_path = EVAL_BUILDINGS_DIR / building_id
    if not ttl_path.exists():
        raise FileNotFoundError(f"TTL file not found: {ttl_path}")
    return ttl_path


def load_kg_content(ttl_path: Path) -> str:
    """Load the knowledge graph content from TTL file."""
    with open(ttl_path, "r") as f:
        return f.read()


async def process_building_queries(
    json_data: List[Dict],
    logger: CsvLogger,
    max_tool_calls: int = 20,
) -> None:
    """
    Process all queries for a single building.

    Args:
        json_data: List containing building data with queries.
        logger: CSV logger instance.
        max_tool_calls: Maximum number of tool calls for the agent.
    """
    if not json_data:
        print("Warning: Empty JSON data")
        return

    building_data = json_data[0]
    building_id = building_data.get("building_id")
    if not building_id:
        print("Warning: No building_id found in JSON")
        return

    print("\n" + "=" * 60)
    print(f"Processing building: {building_id}")
    print("=" * 60)

    # Load TTL graph
    ttl_path = get_ttl_path(building_id)
    kg_content = load_kg_content(ttl_path)

    # Initialise agent
    agent = SimpleSparqlAgentMCP(
        sparql_endpoint=str(ttl_path),
        model_name=MODEL_NAME,
        max_tool_calls=max_tool_calls,
        api_key=API_KEY,
        base_url=BASE_URL,
    )

    queries_list = building_data.get("queries", [])
    total_questions = sum(len(q.get("questions", [])) for q in queries_list)

    print(f"Total query groups: {len(queries_list)}")
    print(f"Total questions to process: {total_questions}")

    question_counter = 0

    for query_group in queries_list:
        query_id = query_group.get("query_id", "unknown")
        ground_truth_sparql = query_group.get("sparql_query", "")
        description = query_group.get("description", "")

        print(f"\n--- Query Group: {query_id} ---")
        print(f"Description: {description}")

        for question_data in query_group.get("questions", []):
            question_counter += 1
            q_num = question_data.get("question_number", question_counter)
            q_text = question_data.get("text", "")
            q_source = question_data.get("source", "unknown")

            print(f"\nQuestion {question_counter}/{total_questions} (Q{q_num}):")
            print(f"  Text: {q_text}")
            print(f"  Source: {q_source}")

            eval_data = {
                "query_id": query_id,
                "question": q_text,
                "ground_truth_sparql": ground_truth_sparql,
            }

            try:
                await agent.generate_query(
                    eval_data=eval_data,
                    logger=logger,
                    prefixes=STANDARD_PREFIXES,
                    knowledge_graph_content=kg_content,
                )
                print("  ✅ Completed")
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
                import traceback

                print(f"  Traceback: {traceback.format_exc()}")


async def main_async() -> None:
    """Async entry point for the benchmark run."""
    print("Starting SPARQL Agent Benchmark Run")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    logger, log_filename = create_timestamped_logger()

    if not BENCHMARK_DIR.is_dir():
        print(f"Benchmark directory not found: {BENCHMARK_DIR}", file=sys.stderr)
        return

    json_files = list(BENCHMARK_DIR.glob("*_combined.json"))
    if not json_files:
        print(f"No JSON files found in {BENCHMARK_DIR}", file=sys.stderr)
        return

    print(f"\nFound {len(json_files)} building files to process:")
    for idx, json_path in enumerate(json_files, start=1):
        print("\n" + "#" * 60)
        print(f"Building {idx}/{len(json_files)}: {json_path.name}")
        print("#" * 60)
        if any(exc in json_path.name for exc in EXCLUDE_BUILDINGS):
            print(f"Skipping excluded building {json_path.name}")
            continue
        try:
            json_data = load_json_file(json_path)
            await process_building_queries(json_data, logger)
        except Exception as e:
            print(f"❌ Unexpected error while processing {json_path.name}: {e}")
            import traceback

            print(traceback.format_exc())
            continue

    print("\n" + "=" * 60)
    print("Benchmark run completed!")
    print(f"Results logged to: {log_filename}")
    print("=" * 60)
    for jf in json_files:
        print(f"  - {jf.name}")

# Loop moved inside main_async


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main_async())
