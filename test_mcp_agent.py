"""
test_simple_sparql_agent.py

Test script for the SimpleSparqlAgentMCP class.
Tests a specific question against a test-building.ttl semantic model.
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_agent import SimpleSparqlAgentMCP, run_agent
from ReAct_agent.utils import CsvLogger, extract_prefixes_from_ttl, get_kg_subset_content


def setup_test_environment():
    """Set up the test environment and paths."""
    # Paths
    current_dir = Path(__file__).parent
    test_building_path = current_dir / "test-building.ttl"
    
    # Check if test file exists
    if not test_building_path.exists():
        print(f"❌ Error: Test file not found at {test_building_path}")
        print("Please ensure test-building.ttl is in the same directory as this script.")
        sys.exit(1)
    
    return test_building_path


def create_test_data():
    """Create test evaluation data."""
    eval_data = {
        'query_id': 'test_001',
        'question': 'For each air terminal equipped with a temperature sensor, which HVAC zones does it supply?',
        'ground_truth_sparql': """
PREFIX brick: <https://brickschema.org/schema/Brick#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?airTerminal ?zone
WHERE {
    ?airTerminal a brick:Air_Terminal_Unit ;
                 brick:hasPoint ?sensor ;
                 brick:feeds ?zone .
    ?sensor a brick:Temperature_Sensor .
    ?zone a brick:HVAC_Zone .
}
        """.strip(),
        'domain': 'brick',
        'complexity': 'medium',
        'query_type': 'multi_hop'
    }
    
    return eval_data


def setup_logger():
    """Set up CSV logger for test results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(__file__).parent / "test_results"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"test_simple_agent_{timestamp}.csv"
    
    logger = CsvLogger(
        filename=str(log_file),
        fieldnames=[
            'query_id', 'question', 'domain', 'complexity', 'query_type',
            'ground_truth_sparql', 'model', 'generated_sparql',
            'syntax_ok', 'returns_results', 'perfect_match',
            'gt_num_rows', 'gt_num_cols', 'gen_num_rows', 'gen_num_cols',
            'arity_matching_f1', 'entity_set_f1', 'row_matching_f1', 'exact_match_f1',
            'less_columns_flag', 'prompt_tokens', 'completion_tokens', 'total_tokens'
        ]
    )
    
    print(f"📝 Results will be logged to: {log_file}")
    return logger


async def test_agent_basic():
    """Basic test of the SimpleSparqlAgentMCP."""
    print("\n" + "="*80)
    print("TEST 1: Basic Agent Initialization and Query Generation")
    print("="*80)
    
    test_building_path = setup_test_environment()
    eval_data = create_test_data()
    logger = setup_logger()
    
    # Extract prefixes from the TTL file
    print("\n📖 Extracting prefixes from test-building.ttl...")
    prefixes = extract_prefixes_from_ttl(str(test_building_path))
    print(f"✅ Extracted prefixes:\n{prefixes}")
    
    # Get knowledge graph content (you may want to extract a subset in practice)
    print("\n📖 Loading knowledge graph content...")
    with open(test_building_path, 'r') as f:
        kg_content = f.read()
    print(f"✅ Loaded {len(kg_content)} characters of KG content")
    
    # Initialize agent
    print("\n🤖 Initializing SimpleSparqlAgentMCP...")
    agent = SimpleSparqlAgentMCP(
        sparql_endpoint=str(test_building_path),
        model_name="gpt-4",  # Change to your preferred model
        max_tool_calls=5,
        mcp_server_script="brick.py"
    )
    
    # Run the agent
    print("\n🚀 Running agent to generate SPARQL query...")
    await agent.generate_query(
        eval_data=eval_data,
        logger=logger,
        prefixes=prefixes,
        knowledge_graph_content=kg_content[:5000]  # Limiting to first 5000 chars for context
    )
    
    print("\n✅ Test 1 completed!")


async def test_agent_with_different_models():
    """Test agent with different model configurations."""
    print("\n" + "="*80)
    print("TEST 2: Testing Different Model Configurations")
    print("="*80)
    
    test_building_path = setup_test_environment()
    eval_data = create_test_data()
    
    models_to_test = [
        {"model_name": "gpt-4", "max_tool_calls": 5},
        {"model_name": "gpt-3.5-turbo", "max_tool_calls": 3},
        {"model_name": "gpt-4", "max_tool_calls": 10},
    ]
    
    prefixes = extract_prefixes_from_ttl(str(test_building_path))
    with open(test_building_path, 'r') as f:
        kg_content = f.read()
    
    for i, config in enumerate(models_to_test, 1):
        print(f"\n--- Configuration {i}/{len(models_to_test)} ---")
        print(f"Model: {config['model_name']}, Max Tool Calls: {config['max_tool_calls']}")
        
        logger = setup_logger()
        
        agent = SimpleSparqlAgentMCP(
            sparql_endpoint=str(test_building_path),
            **config,
            mcp_server_script="brick.py"
        )
        
        # Update query_id to track different configurations
        test_data = eval_data.copy()
        test_data['query_id'] = f"test_config_{i:03d}"
        
        await agent.generate_query(
            eval_data=test_data,
            logger=logger,
            prefixes=prefixes,
            knowledge_graph_content=kg_content[:5000]
        )
    
    print("\n✅ Test 2 completed!")


async def test_convenience_function():
    """Test the convenience run_agent function."""
    print("\n" + "="*80)
    print("TEST 3: Testing Convenience Function")
    print("="*80)
    
    test_building_path = setup_test_environment()
    eval_data = create_test_data()
    logger = setup_logger()
    
    prefixes = extract_prefixes_from_ttl(str(test_building_path))
    with open(test_building_path, 'r') as f:
        kg_content = f.read()
    
    print("\n🚀 Running via convenience function...")
    run_agent(
        sparql_endpoint=str(test_building_path),
        eval_data=eval_data,
        logger=logger,
        prefixes=prefixes,
        knowledge_graph_content=kg_content[:5000],
        model_name="gpt-4",
        max_tool_calls=5,
        mcp_server_script="brick.py"
    )
    
    print("\n✅ Test 3 completed!")


async def test_error_handling():
    """Test error handling with invalid inputs."""
    print("\n" + "="*80)
    print("TEST 4: Error Handling")
    print("="*80)
    
    logger = setup_logger()
    
    # Test with non-existent file
    print("\n🧪 Testing with non-existent TTL file...")
    agent = SimpleSparqlAgentMCP(
        sparql_endpoint="non_existent_file.ttl",
        model_name="gpt-4",
        max_tool_calls=3
    )
    
    eval_data = create_test_data()
    eval_data['query_id'] = 'test_error_001'
    
    await agent.generate_query(
        eval_data=eval_data,
        logger=logger,
        prefixes="",
        knowledge_graph_content=""
    )
    
    print("\n✅ Test 4 completed!")


def print_test_summary():
    """Print a summary of all tests."""
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print("""
Tests completed:
1. ✅ Basic agent initialization and query generation
2. ✅ Multiple model configurations
3. ✅ Convenience function
4. ✅ Error handling

Check the test_results/ directory for detailed CSV logs.
    """)


async def run_all_tests():
    """Run all test cases."""
    print("\n" + "#"*80)
    print("# SIMPLE SPARQL AGENT MCP - TEST SUITE")
    print("#"*80)
    
    try:
        await test_agent_basic()
        await test_agent_with_different_models()
        await test_convenience_function()
        await test_error_handling()
        
        print_test_summary()
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point for the test script."""
    # Check for required environment variables
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables.")
        print("Please set it or provide an api_key_file parameter.")
    
    # Run all tests
    asyncio.run(run_all_tests())


if __name__ == "__main__":
    main()