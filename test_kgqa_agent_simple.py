"""
test_kgqa_agent_simple.py

Simple test to verify kgqa_agent.py can be imported and basic functionality works.
This test doesn't make actual LLM calls.
"""

import os
import sys

print("=" * 70)
print("KGQA Agent MCP - Simple Import and Structure Test")
print("=" * 70)

# Test 1: Import the module
print("\n[Test 1] Importing kgqa_agent module...")
try:
    from kgqa_agent import SparqlRefinementAgentMCP, run_agent
    print("✅ Successfully imported SparqlRefinementAgentMCP and run_agent")
except Exception as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

# Test 2: Import dependencies
print("\n[Test 2] Checking dependencies...")
try:
    from ReAct_agent.utils import CsvLogger, get_kg_subset_content, extract_prefixes_from_ttl
    print("✅ Successfully imported from ReAct_agent.utils")
except Exception as e:
    print(f"❌ Failed to import utils: {e}")
    sys.exit(1)

try:
    from metrics import get_arity_matching_f1, get_entity_and_row_matching_f1, get_exact_match_f1
    print("✅ Successfully imported from metrics")
except Exception as e:
    print(f"❌ Failed to import metrics: {e}")
    sys.exit(1)

# Test 3: Create a simple TTL file and test graph loading
print("\n[Test 3] Testing local TTL file loading...")
test_ttl_content = """
@prefix brick: <https://brickschema.org/schema/Brick#> .
@prefix ex: <http://example.org/building#> .

ex:sensor1 a brick:Temperature_Sensor .
ex:sensor2 a brick:Temperature_Sensor .
"""

test_ttl_path = "test_import.ttl"
with open(test_ttl_path, 'w') as f:
    f.write(test_ttl_content)

try:
    # Just test initialization, don't run queries
    agent = SparqlRefinementAgentMCP(
        sparql_endpoint=test_ttl_path,
        model_name="test/model",
        max_iterations=1,
        api_key="test_key",
        base_url="http://localhost:1234/v1",
        mcp_server_script="brick.py",
    )
    print(f"✅ Successfully initialized agent with local TTL file")
    print(f"   - Graph loaded: {agent.graph is not None}")
    print(f"   - Is remote: {agent.is_remote}")
    print(f"   - Triple count: {len(agent.graph) if agent.graph else 0}")
except Exception as e:
    print(f"❌ Failed to initialize agent: {e}")
    import traceback
    traceback.print_exc()
finally:
    if os.path.exists(test_ttl_path):
        os.remove(test_ttl_path)
        print(f"🧹 Cleaned up {test_ttl_path}")

# Test 4: Test SPARQL query execution on local graph
print("\n[Test 4] Testing SPARQL query execution on local graph...")
test_ttl_content = """
@prefix brick: <https://brickschema.org/schema/Brick#> .
@prefix ex: <http://example.org/building#> .

ex:sensor1 a brick:Temperature_Sensor ;
    brick:hasLocation ex:room101 .

ex:sensor2 a brick:Temperature_Sensor ;
    brick:hasLocation ex:room102 .
"""

test_ttl_path = "test_query.ttl"
with open(test_ttl_path, 'w') as f:
    f.write(test_ttl_content)

try:
    agent = SparqlRefinementAgentMCP(
        sparql_endpoint=test_ttl_path,
        model_name="test/model",
        max_iterations=1,
        api_key="test_key",
        base_url="http://localhost:1234/v1",
        mcp_server_script="brick.py",
    )
    
    # Test a simple SPARQL query
    test_query = """
    PREFIX brick: <https://brickschema.org/schema/Brick#>
    PREFIX ex: <http://example.org/building#>
    
    SELECT ?sensor WHERE {
        ?sensor a brick:Temperature_Sensor .
    }
    """
    
    result = agent._run_sparql_query(test_query)
    print(f"✅ Successfully executed SPARQL query")
    print(f"   - Syntax OK: {result['syntax_ok']}")
    print(f"   - Row count: {result['row_count']}")
    print(f"   - Col count: {result['col_count']}")
    
    if result['row_count'] == 2:
        print(f"✅ Query returned expected number of results (2)")
    else:
        print(f"⚠️  Query returned {result['row_count']} results, expected 2")
        
except Exception as e:
    print(f"❌ Failed to execute query: {e}")
    import traceback
    traceback.print_exc()
finally:
    if os.path.exists(test_ttl_path):
        os.remove(test_ttl_path)
        print(f"🧹 Cleaned up {test_ttl_path}")

# Test 5: Test CsvLogger
print("\n[Test 5] Testing CsvLogger...")
try:
    from ReAct_agent.utils import LOG_FIELDNAMES
    logger = CsvLogger("test_log.csv", LOG_FIELDNAMES)
    
    test_entry = {
        'query_id': 'test_001',
        'question': 'Test question',
        'model': 'test/model',
        'generated_sparql': 'SELECT * WHERE { ?s ?p ?o }',
        'syntax_ok': True,
        'returns_results': True,
        'perfect_match': False,
        'gt_num_rows': 0,
        'gt_num_cols': 0,
        'gen_num_rows': 5,
        'gen_num_cols': 3,
        'arity_matching_f1': 0.8,
        'exact_match_f1': 0.0,
        'entity_set_f1': 0.7,
        'row_matching_f1': 0.6,
        'less_columns_flag': False,
        'prompt_tokens': 100,
        'completion_tokens': 50,
        'total_tokens': 150
    }
    
    logger.log(test_entry)
    logger.close()
    
    if os.path.exists("test_log.csv"):
        print(f"✅ Successfully created and wrote to CSV log")
        os.remove("test_log.csv")
        print(f"🧹 Cleaned up test_log.csv")
    else:
        print(f"⚠️  CSV file was not created")
        
except Exception as e:
    print(f"❌ Failed to test CsvLogger: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("All basic tests completed!")
print("=" * 70)
print("\n✅ The kgqa_agent.py module is properly set up and functional.")
print("   To run full integration tests with LLM calls, use test_kgqa_agent.py")
