"""
Script to run SPARQL query generation agent on all benchmark QA pairs.
Processes all JSON files in Benchmark_QA_pairs directory against their corresponding TTL files.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import glob

# Assuming these are your imports from the agent code
from agent import SimpleSparqlAgentMCP, CsvLogger  # Update with actual module name

config = json.load(open("benchmark-config.json"))
# Configuration - update these with your actual values
MODEL_NAME = config.get("models")[0]
API_KEY = config.get("api-key")
BASE_URL = config.get("base-url")

# Directory paths
BENCHMARK_DIR = Path(config.get("buildingqa-dir")) / "Benchmark_QA_pairs"
EVAL_BUILDINGS_DIR = Path(config.get("buildingqa-dir")) / "eval_buildings"

# Standard prefixes for SPARQL queries
STANDARD_PREFIXES = """
PREFIX brick: <https://brickschema.org/schema/Brick#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2001/XMLSchema#>
"""


def create_timestamped_logger(base_name: str = "sparql_agent_run") -> CsvLogger:
    """Create a logger with timestamp in filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{base_name}_{timestamp}.csv"
    logger = CsvLogger(log_filename)
    print(f"Created logger: {log_filename}")
    return logger


def load_json_file(filepath: Path) -> Dict:
    """Load and parse JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_ttl_path(building_id: str) -> Path:
    """Get the corresponding TTL file path for a building ID."""
    ttl_path = EVAL_BUILDINGS_DIR / building_id
    if not ttl_path.exists():
        raise FileNotFoundError(f"TTL file not found: {ttl_path}")
    return ttl_path


def load_kg_content(ttl_path: Path) -> str:
    """Load the knowledge graph content from TTL file."""
    with open(ttl_path, 'r') as f:
        return f.read()


def process_building_queries(
    json_data: List[Dict],
    logger: CsvLogger,
    max_tool_calls: int = 5
) -> None:
    """
    Process all queries for a single building.
    
    Args:
        json_data: List containing building data with queries
        logger: CSV logger instance
        max_tool_calls: Maximum number of tool calls for agent
    """
    if not json_data or len(json_data) == 0:
        print("Warning: Empty JSON data")
        return
    
    building_data = json_data[0]
    building_id = building_data.get('building_id')
    
    if not building_id:
        print("Warning: No building_id found in JSON")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing building: {building_id}")
    print(f"{'='*60}")
    
    # Get TTL file path and load content
    ttl_path = get_ttl_path(building_id)
    kg_content = load_kg_content(ttl_path)
    
    # Initialize agent for this building
    agent = SimpleSparqlAgentMCP(
        sparql_endpoint=str(ttl_path),
        model_name=MODEL_NAME,
        max_tool_calls=max_tool_calls,
        api_key=API_KEY,
        base_url=BASE_URL,
    )
    
    # Process each query group
    queries_list = building_data.get('queries', [])
    total_questions = sum(len(q.get('questions', [])) for q in queries_list)
    
    print(f"Total query groups: {len(queries_list)}")
    print(f"Total questions to process: {total_questions}")
    
    question_count = 0
    
    for query_group in queries_list:
        query_id = query_group.get('query_id', 'unknown')
        ground_truth_sparql = query_group.get('sparql_query', '')
        description = query_group.get('description', '')
        
        print(f"\n--- Query Group: {query_id} ---")
        print(f"Description: {description}")
        
        questions = query_group.get('questions', [])
        
        for question_data in questions:
            question_count += 1
            question_num = question_data.get('question_number', question_count)
            question_text = question_data.get('text', '')
            question_source = question_data.get('source', 'unknown')
            
            print(f"\nQuestion {question_count}/{total_questions} (Q{question_num}):")
            print(f"  Text: {question_text}")
            print(f"  Source: {question_source}")
            
            # Prepare evaluation data
            eval_data = {
                'building_id': building_id,
                'query_id': query_id,
                'question_number': question_num,
                'question': question_text,
                'question_source': question_source,
                'ground_truth_sparql': ground_truth_sparql,
                'description': description
            }
            
            try:
                # Run agent query generation
                agent.generate_query(
                    eval_data=eval_data,
                    logger=logger,
                    prefixes=STANDARD_PREFIXES,
                    kg_content=kg_content
                )
                print("  ✅ Completed")
                
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
                # Log error but continue processing
                logger.log_error(eval_data, str(e))


def main():
    """Main execution function."""
    print("Starting SPARQL Agent Benchmark Run")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create timestamped logger
    logger = create_timestamped_logger()
    
    # Find all JSON files in benchmark directory
    json_files = list(BENCHMARK_DIR.glob("*_combined.json"))
    
    if not json_files:
        print(f"No JSON files found in {BENCHMARK_DIR}")
        return
    
    print(f"\nFound {len(json_files)} building files to process:")
    for jf in json_files:
        print(f"  - {jf.name}")
    
    # Process each building
    for idx, json_file in enumerate(json_files, 1):
        print(f"\n{'#'*60}")
        print(f"Building {idx}/{len(json_files)}: {json_file.name}")
        print(f"{'#'*60}")
        
        try:
            # Load JSON data
            json_data = load_json_file(json_file)
            
            # Process all queries for this building
            process_building_queries(json_data, logger)
            
        except FileNotFoundError as e:
            print(f"❌ Error: {str(e)}")
            continue
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing JSON: {str(e)}")
            continue
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print("Benchmark run completed!")
    print(f"Results logged to: {logger.log_file if hasattr(logger, 'log_file') else 'CSV file'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()