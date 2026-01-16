"""
test_kgqa_agent.py

Basic test script for kgqa_agent.py
Tests the SparqlRefinementAgentMCP class with a simple example.
"""

import asyncio
import csv
import os
from pathlib import Path
from typing import Any, Dict


# Simple CsvLogger mock for testing
class CsvLogger:
    """Simple CSV logger for testing purposes."""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.fieldnames = None
        
        # Create the file with headers if it doesn't exist
        if not os.path.exists(filename):
            # We'll set fieldnames on first log
            pass
    
    def log(self, entry: Dict[str, Any]):
        """Log an entry to the CSV file."""
        file_exists = os.path.exists(self.filename)
        
        # Set fieldnames from first entry
        if self.fieldnames is None:
            self.fieldnames = list(entry.keys())
        
        with open(self.filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(entry)
        
        print(f"✅ Logged entry to {self.filename}")


# Import the agent
from kgqa_agent import SparqlRefinementAgentMCP


async def test_basic_query():
    """Test 1: Basic query with a simple TTL file."""
    print("\n" + "=" * 70)
    print("TEST 1: Basic Temperature Sensor Query")
    print("=" * 70)
    
    # Create a simple test TTL file
    test_ttl_content = """
@prefix brick: <https://brickschema.org/schema/Brick#> .
@prefix ex: <http://example.org/building#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

ex:sensor1 a brick:Temperature_Sensor ;
    brick:hasLocation ex:room101 .

ex:sensor2 a brick:Temperature_Sensor ;
    brick:hasLocation ex:room102 .

ex:sensor3 a brick:Temperature_Sensor ;
    brick:hasLocation ex:room103 .
"""
    
    # Write test TTL file
    test_ttl_path = "test_simple.ttl"
    with open(test_ttl_path, 'w') as f:
        f.write(test_ttl_content)
    
    print(f"✅ Created test TTL file: {test_ttl_path}")
    
    # Initialize the agent
    agent = SparqlRefinementAgentMCP(
        sparql_endpoint=test_ttl_path,
        model_name="lbl/llama",
        max_iterations=3,
        api_key_file="/Users/lazlopaul/Desktop/cborg/api_key.yaml",
        mcp_server_script="brick.py",
    )
    
    # Prepare evaluation data
    eval_data = {
        'query_id': 'test_001',
        'question': 'What are all the temperature sensors in the building?',
        'ground_truth_sparql': """
PREFIX brick: <https://brickschema.org/schema/Brick#>
PREFIX ex: <http://example.org/building#>

SELECT ?sensor WHERE {
    ?sensor a brick:Temperature_Sensor .
}
""",
    }
    
    # Knowledge graph content (same as TTL for this simple test)
    knowledge_graph_content = test_ttl_content
    
    prefixes = """
PREFIX brick: <https://brickschema.org/schema/Brick#>
PREFIX ex: <http://example.org/building#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
"""
    
    # Create logger
    logger = CsvLogger("test_results_basic.csv")
    
    # Run the agent
    try:
        await agent.refine_and_evaluate_query(
            eval_data=eval_data,
            logger=logger,
            prefixes=prefixes,
            knowledge_graph_content=knowledge_graph_content
        )
        print("\n✅ Test 1 completed successfully!")
    except Exception as e:
        print(f"\n❌ Test 1 failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    if os.path.exists(test_ttl_path):
        os.remove(test_ttl_path)
        print(f"🧹 Cleaned up test file: {test_ttl_path}")


async def test_no_results_query():
    """Test 2: Query that should return no results."""
    print("\n" + "=" * 70)
    print("TEST 2: Query with No Results")
    print("=" * 70)
    
    # Create a simple test TTL file
    test_ttl_content = """
@prefix brick: <https://brickschema.org/schema/Brick#> .
@prefix ex: <http://example.org/building#> .

ex:sensor1 a brick:Temperature_Sensor ;
    brick:hasLocation ex:room101 .
"""
    
    # Write test TTL file
    test_ttl_path = "test_no_results.ttl"
    with open(test_ttl_path, 'w') as f:
        f.write(test_ttl_content)
    
    print(f"✅ Created test TTL file: {test_ttl_path}")
    
    # Initialize the agent
    agent = SparqlRefinementAgentMCP(
        sparql_endpoint=test_ttl_path,
        model_name="lbl/llama",
        max_iterations=2,
        api_key_file="/Users/lazlopaul/Desktop/cborg/api_key.yaml",
        mcp_server_script="brick.py",
    )
    
    # Prepare evaluation data - asking for something that doesn't exist
    eval_data = {
        'query_id': 'test_002',
        'question': 'What are all the humidity sensors in the building?',
        'ground_truth_sparql': None,  # No ground truth for this test
    }
    
    knowledge_graph_content = test_ttl_content
    
    prefixes = """
PREFIX brick: <https://brickschema.org/schema/Brick#>
PREFIX ex: <http://example.org/building#>
"""
    
    # Create logger
    logger = CsvLogger("test_results_no_results.csv")
    
    # Run the agent
    try:
        await agent.refine_and_evaluate_query(
            eval_data=eval_data,
            logger=logger,
            prefixes=prefixes,
            knowledge_graph_content=knowledge_graph_content
        )
        print("\n✅ Test 2 completed successfully!")
    except Exception as e:
        print(f"\n❌ Test 2 failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    if os.path.exists(test_ttl_path):
        os.remove(test_ttl_path)
        print(f"🧹 Cleaned up test file: {test_ttl_path}")


async def test_relationship_query():
    """Test 3: Query involving relationships between entities."""
    print("\n" + "=" * 70)
    print("TEST 3: Relationship Query (AHU and Fans)")
    print("=" * 70)
    
    # Create a test TTL file with relationships
    test_ttl_content = """
@prefix brick: <https://brickschema.org/schema/Brick#> .
@prefix ex: <http://example.org/building#> .

ex:ahu1 a brick:Air_Handler_Unit ;
    brick:hasPart ex:sf1 .

ex:ahu2 a brick:Air_Handler_Unit ;
    brick:hasPart ex:sf2 .

ex:sf1 a brick:Supply_Fan .
ex:sf2 a brick:Supply_Fan .
"""
    
    # Write test TTL file
    test_ttl_path = "test_relationships.ttl"
    with open(test_ttl_path, 'w') as f:
        f.write(test_ttl_content)
    
    print(f"✅ Created test TTL file: {test_ttl_path}")
    
    # Initialize the agent
    agent = SparqlRefinementAgentMCP(
        sparql_endpoint=test_ttl_path,
        model_name="lbl/llama",
        max_iterations=3,
        api_key_file="/Users/lazlopaul/Desktop/cborg/api_key.yaml",
        mcp_server_script="brick.py",
    )
    
    # Prepare evaluation data
    eval_data = {
        'query_id': 'test_003',
        'question': 'List all air handling units and their supply fans.',
        'ground_truth_sparql': """
PREFIX brick: <https://brickschema.org/schema/Brick#>
PREFIX ex: <http://example.org/building#>

SELECT ?ahu ?fan WHERE {
    ?ahu a brick:Air_Handler_Unit ;
         brick:hasPart ?fan .
    ?fan a brick:Supply_Fan .
}
""",
    }
    
    knowledge_graph_content = test_ttl_content
    
    prefixes = """
PREFIX brick: <https://brickschema.org/schema/Brick#>
PREFIX ex: <http://example.org/building#>
"""
    
    # Create logger
    logger = CsvLogger("test_results_relationships.csv")
    
    # Run the agent
    try:
        await agent.refine_and_evaluate_query(
            eval_data=eval_data,
            logger=logger,
            prefixes=prefixes,
            knowledge_graph_content=knowledge_graph_content
        )
        print("\n✅ Test 3 completed successfully!")
    except Exception as e:
        print(f"\n❌ Test 3 failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    if os.path.exists(test_ttl_path):
        os.remove(test_ttl_path)
        print(f"🧹 Cleaned up test file: {test_ttl_path}")


async def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("KGQA Agent MCP - Test Suite")
    print("=" * 70)
    
    # Run tests
    await test_basic_query()
    await test_no_results_query()
    await test_relationship_query()
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
    
    # Show generated log files
    print("\n📊 Generated log files:")
    for log_file in ["test_results_basic.csv", "test_results_no_results.csv", "test_results_relationships.csv"]:
        if os.path.exists(log_file):
            print(f"  - {log_file}")


if __name__ == "__main__":
    asyncio.run(main())
