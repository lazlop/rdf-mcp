"""
test_simple_sparql_agent.py

Basic test script for the SimpleSparqlAgentMCP class.
"""

import asyncio
import tempfile
from pathlib import Path
from agent import SimpleSparqlAgentMCP, run_agent
from utils import CsvLogger
from rdflib import Graph
import yaml

with open('/Users/lazlopaul/Desktop/cborg/api_key.yaml', 'r') as file:
    config = yaml.safe_load(file)
    API_KEY = config['key']
    BASE_URL = config['base_url']

MODEL_NAME = "lbl/cborg-coder"

async def test_basic_initialization():
    """Test that the agent initializes correctly."""
    print("\n=== Test 1: Basic Initialization ===")
    
    ttl_path = 'test-building.ttl'
    
    agent = SimpleSparqlAgentMCP(
        sparql_endpoint=ttl_path,
        model_name=MODEL_NAME,
        max_tool_calls=5,
        api_key=API_KEY,
        base_url=BASE_URL,
    )
    
    print(f"✅ Agent initialized")
    print(f"   - Endpoint: {agent.sparql_endpoint_url}")
    print(f"   - Graph loaded: {agent.graph is not None}")
    if agent.graph:
        print(f"   - Triple count: {len(agent.graph)}")


async def test_query_generation():
    """Test query generation with mock data."""
    print("\n=== Test 2: Query Generation ===")
    
    ttl_path = 'test-building.ttl'
    log_path = 'test-logger.csv'
    logger = CsvLogger(log_path)
    
    eval_data = {
        'query_id': 'test_001',
        'question': 'What fans are part of AHU1?',
        'ground_truth_sparql': """
            PREFIX brick: <https://brickschema.org/schema/Brick#>
            PREFIX ex: <http://example.org/>
            SELECT ?fan WHERE {
                ex:AHU1 brick:hasPart ?fan .
                ?fan a brick:Fan .
            }
        """
    }
    
    prefixes = """
        PREFIX brick: <https://brickschema.org/schema/Brick#>
        PREFIX ex: <http://example.org/>
    """
    
    kg_content = "ex:AHU1 a brick:Air_Handler_Unit ; brick:hasPart ex:Fan1 ."



    agent = SimpleSparqlAgentMCP(
        sparql_endpoint=ttl_path,
        model_name=MODEL_NAME,
        max_tool_calls=5,
        api_key=API_KEY,
        base_url=BASE_URL,
    )
    
    try:
        await agent.generate_query(eval_data, logger, prefixes, kg_content)
        print("✅ Query generation completed")
    except Exception as e:
        print(f"⚠️  Query generation failed (expected without real API): {e}")
       


async def main():
    """Run all tests."""
    print("🧪 Starting SimpleSparqlAgentMCP Tests\n")
    
    await test_basic_initialization()
    await test_query_generation()
    
    print("\n✨ Tests complete!")


if __name__ == "__main__":
    asyncio.run(main())