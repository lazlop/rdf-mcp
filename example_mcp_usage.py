"""
example_mcp_usage.py

Example script demonstrating how to use the SparqlRefinementAgentMCP class.
"""

import asyncio
from kgqa_agent import SparqlRefinementAgentMCP, run_agent
from ReAct_agent.utils import CsvLogger

# Example 1: Using the agent directly with async/await
async def example_direct_usage():
    """Example of using the agent directly with async/await."""
    
    # Initialize the agent
    agent = SparqlRefinementAgentMCP(
        sparql_endpoint="path/to/your/file.ttl",  # or "http://your-sparql-endpoint.com/sparql"
        model_name="lbl/llama",
        max_iterations=5,
        api_key_file="/Users/lazlopaul/Desktop/cborg/api_key.yaml",  # or use api_key and base_url directly
        mcp_server_script="brick.py",  # or "s223.py" for ASHRAE 223p
    )
    
    # Prepare evaluation data
    eval_data = {
        'query_id': 'example_001',
        'question': 'What are all the temperature sensors in the building?',
        'ground_truth_sparql': None,  # Optional: provide ground truth for evaluation
    }
    
    # Prepare knowledge graph content (subset relevant to the question)
    knowledge_graph_content = """
    @prefix brick: <https://brickschema.org/schema/Brick#> .
    @prefix ex: <http://example.org/building#> .
    
    ex:sensor1 a brick:Temperature_Sensor ;
        brick:hasLocation ex:room101 .
    
    ex:sensor2 a brick:Temperature_Sensor ;
        brick:hasLocation ex:room102 .
    """
    
    prefixes = """
    PREFIX brick: <https://brickschema.org/schema/Brick#>
    PREFIX ex: <http://example.org/building#>
    """
    
    # Create logger
    logger = CsvLogger("example_results.csv")
    
    # Run the agent
    await agent.refine_and_evaluate_query(
        eval_data=eval_data,
        logger=logger,
        prefixes=prefixes,
        knowledge_graph_content=knowledge_graph_content
    )
    
    print("\n✅ Example completed!")


# Example 2: Using the convenience function
def example_convenience_function():
    """Example of using the convenience function (handles async internally)."""
    
    eval_data = {
        'query_id': 'example_002',
        'question': 'List all air handling units and their supply fans.',
        'ground_truth_sparql': None,
    }
    
    knowledge_graph_content = """
    @prefix brick: <https://brickschema.org/schema/Brick#> .
    @prefix ex: <http://example.org/building#> .
    
    ex:ahu1 a brick:Air_Handler_Unit ;
        brick:hasPart ex:sf1 .
    
    ex:sf1 a brick:Supply_Fan .
    """
    
    prefixes = """
    PREFIX brick: <https://brickschema.org/schema/Brick#>
    PREFIX ex: <http://example.org/building#>
    """
    
    logger = CsvLogger("example_results_2.csv")
    
    # Use the convenience function
    run_agent(
        sparql_endpoint="path/to/your/file.ttl",
        eval_data=eval_data,
        logger=logger,
        prefixes=prefixes,
        knowledge_graph_content=knowledge_graph_content,
        model_name="lbl/llama",
        max_iterations=5,
        api_key_file="/Users/lazlopaul/Desktop/cborg/api_key.yaml",
        mcp_server_script="brick.py"
    )
    
    print("\n✅ Example completed!")


# Example 3: Using with ASHRAE 223p ontology
async def example_s223_usage():
    """Example of using the agent with ASHRAE 223p ontology."""
    
    agent = SparqlRefinementAgentMCP(
        sparql_endpoint="path/to/s223/file.ttl",
        model_name="lbl/llama",
        max_iterations=5,
        api_key_file="/Users/lazlopaul/Desktop/cborg/api_key.yaml",
        mcp_server_script="s223.py",  # Use s223.py for ASHRAE 223p
    )
    
    eval_data = {
        'query_id': 's223_example_001',
        'question': 'What are all the connection points in the system?',
        'ground_truth_sparql': None,
    }
    
    knowledge_graph_content = """
    @prefix s223: <http://data.ashrae.org/standard223#> .
    @prefix ex: <http://example.org/system#> .
    
    ex:cp1 a s223:ConnectionPoint ;
        s223:cnx ex:equipment1 .
    """
    
    prefixes = """
    PREFIX s223: <http://data.ashrae.org/standard223#>
    PREFIX ex: <http://example.org/system#>
    """
    
    logger = CsvLogger("s223_results.csv")
    
    await agent.refine_and_evaluate_query(
        eval_data=eval_data,
        logger=logger,
        prefixes=prefixes,
        knowledge_graph_content=knowledge_graph_content
    )
    
    print("\n✅ S223 example completed!")


# Example 4: Using with custom MCP server arguments
async def example_custom_mcp_args():
    """Example of using custom MCP server arguments."""
    
    custom_mcp_args = [
        "run",
        "--with", "mcp[cli]",
        "--with", "rdflib",
        "--with", "oxrdflib",
        "--with", "custom-package",  # Add custom packages
        "mcp",
        "run",
        "brick.py"
    ]
    
    agent = SparqlRefinementAgentMCP(
        sparql_endpoint="path/to/your/file.ttl",
        model_name="lbl/llama",
        max_iterations=5,
        api_key="your-api-key",
        base_url="http://localhost:1234/v1",
        mcp_server_script="brick.py",
        mcp_server_args=custom_mcp_args
    )
    
    # ... rest of the code similar to previous examples


if __name__ == "__main__":
    print("=" * 60)
    print("SPARQL Refinement Agent MCP - Examples")
    print("=" * 60)
    
    # Run example 1 (async)
    print("\n--- Example 1: Direct async usage ---")
    asyncio.run(example_direct_usage())
    
    # Run example 2 (convenience function)
    print("\n--- Example 2: Convenience function ---")
    example_convenience_function()
    
    # Run example 3 (S223)
    print("\n--- Example 3: ASHRAE 223p usage ---")
    asyncio.run(example_s223_usage())
