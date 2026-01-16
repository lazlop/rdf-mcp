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
    
    # Create a minimal TTL file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttl', delete=False) as f:
        f.write("""
@prefix brick: <https://brickschema.org/schema/Brick#> .
@prefix ex: <http://example.org/> .

ex:AHU1 a brick:Air_Handler_Unit .
ex:Room1 a brick:Room .
""")
        ttl_path = f.name
        g = Graph()
        g.parse(ttl_path, format='turtle')
    
    try:
        agent = SimpleSparqlAgentMCP(
            sparql_endpoint=ttl_path,
            model_name=MODEL_NAME,
            max_tool_calls=5,
            api_key=API_KEY,
            base_url=BASE_URL,
            graph=g
        )
        
        print(f"✅ Agent initialized")
        print(f"   - Endpoint: {agent.sparql_endpoint_url}")
        print(f"   - Graph loaded: {agent.graph is not None}")
        if agent.graph:
            print(f"   - Triple count: {len(agent.graph)}")
        
    finally:
        Path(ttl_path).unlink()


async def test_query_generation():
    """Test query generation with mock data."""
    print("\n=== Test 2: Query Generation ===")
    
    # Create test TTL file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttl', delete=False) as f:
        f.write("""
@prefix brick: <https://brickschema.org/schema/Brick#> .
@prefix ex: <http://example.org/> .

ex:AHU1 a brick:Air_Handler_Unit ;
    brick:hasPart ex:Fan1 .
ex:Fan1 a brick:Fan .
""")
        ttl_path = f.name
        g = Graph()
        g.parse(ttl_path, format='turtle')
    
    # Create test CSV logger
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as log_f:
        log_path = log_f.name
    
    try:
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
            graph=g
        )
        
        
        # Note: This will fail without real API credentials
        # It's mainly to test the flow
        try:
            await agent.generate_query(eval_data, logger, prefixes, kg_content)
            print("✅ Query generation completed")
        except Exception as e:
            print(f"⚠️  Query generation failed (expected without real API): {e}")
        
    finally:
        Path(ttl_path).unlink()
        Path(log_path).unlink()


def test_run_agent_function():
    """Test the convenience run_agent function."""
    print("\n=== Test 3: run_agent Function ===")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttl', delete=False) as f:
        f.write("@prefix brick: <https://brickschema.org/schema/Brick#> .")
        ttl_path = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as log_f:
        log_path = log_f.name
    
    try:
        logger = CsvLogger(log_path)
        eval_data = {
            'query_id': 'test_002',
            'question': 'Test question',
            'ground_truth_sparql': 'SELECT ?s WHERE { ?s a ?type }'
        }
        
        try:
            run_agent(
                sparql_endpoint=ttl_path,
                eval_data=eval_data,
                logger=logger,
                prefixes="",
                knowledge_graph_content="",
                model_name="gpt-4",
                api_key="test-key"
            )
            print("✅ run_agent function executed")
        except Exception as e:
            print(f"⚠️  run_agent failed (expected without real API): {e}")
            
    finally:
        Path(ttl_path).unlink()
        Path(log_path).unlink()


async def main():
    """Run all tests."""
    print("🧪 Starting SimpleSparqlAgentMCP Tests\n")
    
    await test_basic_initialization()
    await test_query_generation()
    # test_run_agent_function()
    
    print("\n✨ Tests complete!")


if __name__ == "__main__":
    asyncio.run(main())