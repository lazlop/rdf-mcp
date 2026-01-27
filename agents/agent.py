"""
simple_sparql_agent_mcp.py

Simplified SPARQL agent that uses tool calls up to a maximum limit.
"""
from pprint import pprint
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from jsonschema import ValidationError
import yaml
from pydantic import BaseModel, Field
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits, UsageLimitExceeded
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.toolsets.fastmcp import FastMCPToolset
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after
from httpx import AsyncClient, HTTPStatusError
from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential


from pyparsing import ParseException
from rdflib import BNode, Graph, Literal, URIRef

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.metrics import (
    get_arity_matching_f1,
    get_entity_set_f1,
    get_row_matching_f1,
    get_exact_match_f1,
    get_best_subset_column_f1
)

from scripts.utils import CsvLogger

from agents.kgqa import sparql_query, TOOLSETS


RUN_UNTIL_RESULTS = True  # If True, run until results are found; else, single pass
# RUN_UNTIL_RESULTS = False # Just a single pass 

def create_retrying_client():
    """Create a client with smart retry handling for multiple error types."""

    def should_retry_status(response):
        """Raise exceptions for retryable HTTP status codes."""
        if response.status_code in (429, 502, 503, 504):
            response.raise_for_status()  # This will raise HTTPStatusError

    transport = AsyncTenacityTransport(
        config=RetryConfig(
            # Retry on HTTP errors and connection issues
            retry=retry_if_exception_type((HTTPStatusError, ConnectionError)),
            # Smart waiting: respects Retry-After headers, falls back to exponential backoff
            wait=wait_retry_after(
                fallback_strategy=wait_exponential(multiplier=1, max=60),
                max_wait=300
            ),
            # Stop after 5 attempts
            stop=stop_after_attempt(5),
            # Re-raise the last exception if all retries fail
            reraise=True
        ),
        validate_response=should_retry_status
    )
    return AsyncClient(transport=transport)


class SparqlQuery(BaseModel):
    """Model for the agent's output."""
    sparql_query: str = Field(..., description="The generated SPARQL query.")


class QueryCritique(BaseModel):
    """Model for the Critique Agent's structured feedback."""
    decision: str = Field(..., description="The decision, either 'IMPROVE' or 'FINAL'.")
    feedback: str = Field(..., description="Natural language feedback explaining the decision.")


class SimpleSparqlAgentMCP:
    """
    A simplified SPARQL agent that uses MCP tools to generate queries.
    Uses a single agent with a maximum number of tool calls.
    """

    def __init__(
        self, 
        sparql_endpoint: str, # just a graph file for now
        parsed_graph_file: str,
        model_name: str = "lbl/cborg-coder",
        max_tool_calls: int = 100,
        max_iterations: int = 1,
        total_tokens_limit: int = 200000,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        config_file: Optional[str] = None,
        toolset: Optional[str] = 'mcp',
        mcp_server_script: str = "../agents/kgqa.py",
        reasoning_model: bool = False,
    ):
        """
        Initialize the Simple SPARQL Agent with MCP support.
        
        Args:
            sparql_endpoint: URL of SPARQL endpoint or path to local TTL file
            model_name: Name of the model to use
            max_tool_calls: Maximum number of tool calls allowed
            api_key: API key for the model provider
            base_url: Base URL for the model provider
            api_key_file: Path to YAML file containing 'key' and 'base_url'
            mcp_server_script: Name of the MCP server script
            mcp_server_args: Custom arguments for MCP server
        """
        self.sparql_endpoint_url = sparql_endpoint
        self.parsed_graph_file = parsed_graph_file
        self.model_name = model_name
        self.max_tool_calls = max_tool_calls
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.messages = []
        self.max_iterations = max_iterations
        self.total_tokens_limit = total_tokens_limit

        # Load API credentials
        if api_key:
            self.api_key = api_key
            self.base_url = base_url
            self.mcp_server_script = mcp_server_script
        elif config_file:
            with open(config_file, 'r') as file:
                config = json.load(file)
                self.api_key = config.get('api-key', api_key)
                self.base_url = config.get('base-url', base_url)
                self.model_name = config.get('models', model_name)[0]
                self.total_tokens_limit = config.get('total_tokens_limit', total_tokens_limit)
                self.max_iterations = config.get('max_iterations', max_iterations)
                self.max_tool_calls = config.get('max_tool_calls', max_tool_calls)
                self.mcp_server_script = config.get('mcp_server_script', mcp_server_script)
                self.reasoning_model = config.get('reasoning_model', reasoning_model)
                toolset_name = config.get('toolset', toolset)
                self.toolset = FastMCPToolset(TOOLSETS.get(toolset_name))
                print(config)
        
        self.is_remote = sparql_endpoint.lower().startswith("http")

        if self.is_remote:
            raise NotImplementedError("Remote SPARQL endpoints are not supported in this simplified agent.")
        else:
            print(f"🗂️ Local TTL file mode activated. Loading graph from: {self.sparql_endpoint_url}")
            if not os.path.exists(self.sparql_endpoint_url):
                print(f"   -> ❌ ERROR: File not found at {self.sparql_endpoint_url}. Queries will fail.")
                return
            try:
                self.graph = Graph(store = "Oxigraph")
                self.graph.parse(self.sparql_endpoint_url, format="turtle")
                print(f"   -> ✅ Graph loaded successfully with {len(self.graph)} triples.")
            except Exception as e:
                print(f"   -> ❌ ERROR: Failed to load or parse the TTL file: {e}")
                self.graph = None

        # Pass graph_file as environment variable to MCP server
        os.environ['GRAPH_FILE'] = self.sparql_endpoint_url
        os.environ['PARSED_GRAPH_FILE'] = self.parsed_graph_file
        # Set up the model
        client = create_retrying_client()

        self.model = OpenAIModel(
            model_name=self.model_name,
            provider=OpenAIProvider(base_url=self.base_url, api_key=self.api_key, http_client = client),
        )

        
        self.limits = UsageLimits(total_tokens_limit = self.total_tokens_limit, request_limit = self.max_tool_calls)

        recommended_tool_calls = self.max_tool_calls // 5

        system_prompt = (
            f"You are an expert SPARQL developer specializing in Brick Schema and ASHRAE 223p.\n"
            f"Generate complete, validated SPARQL queries to answer user questions.\n\n"
            
            f"MANDATORY WORKFLOW:\n"
            f"Step 1: Call get_building_summary to understand the building model and namespaces\n"
            f"Step 2: Call get_relationship_between_classes to identify valid property paths\n"
            f"Step 3: Call sparql_snapshots to review existing query patterns\n"
            f"Step 4: Construct and return your final SPARQL query\n"
            f"Step 5: If execution fails, call describe_entity on problematic URIs to debug\n\n"
            
            f"QUERY CONSTRUCTION RULES:\n"
            f"1. Prefixes: Always define standard prefixes (brick:, rdf:, rdfs:, unit:, s223:)\n"
            f"2. Subclass handling: Use 'rdf:type/rdfs:subClassOf*' to capture class hierarchies\n"
            f"3. Projections: When writing SPARQL queries return more columns rather than fewer\n"
            f"4. Verification: Never guess entity names or relationships - use tools to verify\n\n"
            
            f"REASONING APPROACH:\n"
            f"- Before each tool call, briefly state: (1) what you know, (2) what you need to verify to answer the user request, (3) which tool to use\n"
            f"- Keep reasoning concise - 2-3 bullet points maximum\n"
            f"- After gathering information, write your query directly without redundant verification\n"
        )

        self.agent = Agent(
            self.model,
            output_type=SparqlQuery,
            toolsets = [self.toolset],
            system_prompt=system_prompt,
            retries=10
        )
        print('✅ SimpleSparqlAgentMCP initialized successfully.')
   
    async def generate_query(
        self,
        eval_data: Dict[str, Any],
        logger: CsvLogger,
        prefixes: str,
    ) -> None:
        """Generate and evaluate a SPARQL query using ReAct-like approach."""
        self.prompt_tokens = self.completion_tokens = self.total_tokens = 0
        
        generated_query = ""
        tool_calls_exceeded = False
        actual_tool_calls = 0
        critique_feedback = None  # Store critique feedback for next iteration
        query_results = None  # Store results for retry context
        
        nl_question = eval_data['question']
        ground_truth_sparql = eval_data.get('ground_truth_sparql')

        print(f"\n🚀 Generating query for: '{nl_question}'")

        # =========================================================================
        # PHASE 1: PLANNING - Get the model to think about what tools it needs
        # =========================================================================
        planning_prompt = (
            f"Question: {nl_question}\n\n"
            f"Available tools:\n"
            f"- get_building_summary: Get overview of building structure and entities\n"
            f"- get_relationship_between_classes: Find predicate paths between entity classes\n"
            f"- sparql_snapshots: See example SPARQL query patterns\n"
            f"- describe_entity: Get detailed information about a specific entity\n\n"
            f"Think step-by-step:\n"
            f"1. What information do I need to answer this question?\n"
            f"2. Which tools should I call and in what order?\n"
            f"3. What am I looking for from each tool?\n\n"
            f"Provide your reasoning and planned approach."
        )
        self.all_previous_messages = []
        
        try:
            # Create a planning agent without structured output
            planning_agent = Agent(
                self.model,
                toolsets=[self.toolset],
                system_prompt="You are a helpful assistant that plans approaches to solving SPARQL query generation tasks.",
            )
            if self.reasoning_model:
                print("🤖 Skipping planning phase for reasoning models.")
            else:
                print("📋 Phase 1: Planning approach...")
                with capture_run_messages() as planning_messages:
                    planning_result = await planning_agent.run(
                        planning_prompt,
                        usage_limits=self.limits,
                    )
                    
                    # Track tokens from planning
                    if hasattr(planning_result, 'usage'):
                        usage = planning_result.usage()
                        if usage:
                            self.prompt_tokens += usage.input_tokens
                            self.completion_tokens += usage.output_tokens
                            self.total_tokens += usage.total_tokens
                    
                    self.all_previous_messages += [str(msg) for msg in planning_messages]
                
                plan = planning_result.data if hasattr(planning_result, 'data') else str(planning_result)
                print(f"📝 Plan: {plan}\n")
            
            # =========================================================================
            # PHASE 2: EXECUTION - Execute the plan and generate query
            # =========================================================================
            for i in range(self.max_iterations):
                iteration_tool_calls = 0
                
                if i == 0:
                    # First iteration: use the plan
                    if self.reasoning_model:
                        execution_prompt = (
                            f"User Question: {nl_question}\n"
                            # f"Generate a SPARQL query to answer the user question. \n"
                            # f"Use the tools available to gather necessary information before constructing the query. \n"
                            # f"Return the complete SPARQL query. \n"
                            # f"IMPORTANT - when you have gathered enough information, return the SPARQL query. \n"                            
                        )
                    else:
                        execution_prompt = (
                            f"Question: {nl_question}\n\n"
                            f"Your planned approach:\n{plan}\n\n"
                            f"Now execute this plan:\n"
                            f"1. Call the tools you identified to gather necessary information\n"
                            f"2. Use the tool results to construct an accurate SPARQL query\n"
                            f"3. Return the complete SPARQL query\n\n"
                            f"Remember: Use actual entity names and relationships discovered through the tools."
                        )
                else:
                    # Retry iterations: include previous failure context with critique feedback
                    execution_prompt = (
                        f"Question: {nl_question}\n\n"
                        f"Previous attempt failed:\n"
                        f"Query: {generated_query}\n"
                        f"Result: {json.dumps(query_results)}\n"
                        f"Critique Feedback: {critique_feedback}\n\n"
                        f"The critique agent has identified issues with your query.\n"
                        # f"1. Use describe_entity on entities that might not exist\n"
                        # f"2. Verify relationship paths with get_relationship_between_classes\n"
                        # f"3. Check sparql_snapshots for correct query patterns\n"
                        # f"4. Generate a corrected SPARQL query that addresses the critique\n\n"
                        f"Think carefully about the feedback and how you can improve the query. Use sparql_snapshots to ensure you use correct query patterns.\n"
                    )
                
                print(f"🔧 Phase 2: Execution (Iteration {i+1}/{self.max_iterations})...")
                
                with capture_run_messages() as messages:
                    async def _run_agent(message_history=[]):
                        return await self.agent.run(
                            execution_prompt,
                            message_history=message_history,
                            usage_limits=self.limits, 
                        )
                    
                    result = await _run_agent()
                    
                    # Track tokens
                    if hasattr(result, 'usage'):
                        usage = result.usage()
                        if usage:
                            self.prompt_tokens += usage.input_tokens
                            self.completion_tokens += usage.output_tokens
                            self.total_tokens += usage.total_tokens
                    
                    # Count tool calls
                    for msg in messages:
                        if hasattr(msg, 'parts'):
                            for part in msg.parts:
                                if hasattr(part, 'part_kind') and part.part_kind == 'tool-call':
                                    iteration_tool_calls += 1
                                elif type(part).__name__ == 'ToolCallPart':
                                    iteration_tool_calls += 1
                    
                    actual_tool_calls += iteration_tool_calls
                    self.all_previous_messages += [str(msg) for msg in messages]
                    
                    # Check if no tools were called on first execution
                    if i == 0 and iteration_tool_calls == 0:
                        print("⚠️ No tools called in first iteration")
                    
                    # Check tool call limit
                    if actual_tool_calls > self.max_tool_calls:
                        tool_calls_exceeded = True
                        print(f"⚠️ Tool call limit exceeded: {actual_tool_calls}/{self.max_tool_calls}")
                        generated_query = ""
                        break
                    else:
                        generated_query = result.output.sparql_query
                        print(f"✅ Generated query (used {iteration_tool_calls} tools this iteration, {actual_tool_calls}/{self.max_tool_calls} total):\n{generated_query}")
                    
                # =========================================================================
                # PHASE 3: OBSERVATION - Test the query and decide next action
                # =========================================================================
                
                if i < self.max_iterations - 1:
                    print("🔍 Phase 3: Testing and critiquing query...")

                    # Execute the query
                    query_results = sparql_query(generated_query, result_length=100)

                    # Prepare results summary for critique
                    if query_results and query_results['row_count'] > 0:
                        results_summary = f"Query returned {query_results['row_count']} results successfully."
                        print(f"   ✅ {results_summary}")
                    else:
                        error_msg = query_results.get('error', 'No results') if query_results else 'Query execution failed'
                        results_summary = f"Query failed with error: {error_msg}"
                        print(f"   ❌ {results_summary}")

                    # Call Critique Agent to evaluate the query
                    print("🧐 Calling Critique Agent...")
                    
                    # Create critique agent with structured output
                    critique_agent = Agent(
                        self.model,
                        output_type=QueryCritique,
                        toolsets = [self.toolset],
                        system_prompt=(
                            "You are an expert in SPARQL, especially for Brick Schema and ASHRAE 223p. "
                            "Your job is to review a SPARQL query and its results based on an original question. "
                            "Think carefully about whether the query answers the question accurately. Does it retrieve enough results? "
                            "Provide brief constructive feedback to improve the query if needed."
                            "Use 'FINAL' if the query correctly answers the question, even with zero results if that's the accurate answer. "
                            "Use 'IMPROVE' if the query has errors, incorrect logic, or doesn't address the question properly."
                        ),
                    )
                    
                    # Build critique prompt
                    critique_prompt = (
                        f"Original Question: \"{nl_question}\"\n\n"
                        f"SPARQL Query Attempt:\n```sparql\n{generated_query}\n```\n\n"
                        f"Execution Results Summary:\n{results_summary}\n\n"
                        f"Sample Results (if any):\n"
                        f"{json.dumps(query_results.get('results', [])[:3], indent=2) if query_results and query_results.get('results') else 'None'}\n\n"
                        f"Provide your decision (FINAL or IMPROVE) and detailed feedback."
                    )
                    
                    try:
                        with capture_run_messages() as critique_messages:
                            critique_result = await critique_agent.run(
                                critique_prompt,
                                usage_limits=self.limits,
                            )
                            
                            # Track tokens from critique
                            if hasattr(critique_result, 'usage'):
                                usage = critique_result.usage()
                                if usage:
                                    self.prompt_tokens += usage.input_tokens
                                    self.completion_tokens += usage.output_tokens
                                    self.total_tokens += usage.total_tokens
                            
                            self.all_previous_messages += [str(msg) for msg in critique_messages]
                        
                        # Extract critique data
                        critique = critique_result.output
                        
                        print(f"   -> Critique Decision: {critique.decision}")
                        print(f"   -> Critique Feedback: {critique.feedback}")

                        # Act on critique decision
                        if critique.decision == "FINAL":
                            print("\n✅ Critique Agent approved the query. Refinement complete.")
                            break
                        else:
                            # Query needs improvement - store feedback for next iteration
                            print(f"   ⚠️  Query needs improvement: {critique.feedback}")
                            critique_feedback = critique.feedback
                            
                            if not RUN_UNTIL_RESULTS:
                                print("   Stopping after single iteration (RUN_UNTIL_RESULTS=False)")
                                break
                            
                            print(f"   🔄 Retrying with feedback ({i+1}/{self.max_iterations})...")
                            # feedback will be used in next iteration's execution_prompt
                    
                    except Exception as e:
                        print(f"❌ Critique Agent failed: {e}")
                        if not RUN_UNTIL_RESULTS:
                            print("   Stopping after single iteration (RUN_UNTIL_RESULTS=False)")
                            break
                        print(f"   🔄 Retrying ({i+1}/{self.max_iterations})...")
                        critique_feedback = f"Critique agent failed with error: {str(e)}. Please review and improve your query."
                        continue
        except UsageLimitExceeded as e:
            # Capture the usage information from the exception
            print(f"❌ Token limit exceeded: {e}")
            token_limit_exceeded = True
            
            # Parse token count from exception message
            # Format: "Exceeded the total_tokens_limit of X (total_tokens=Y)"
            import re
            match = re.search(r'total_tokens=(\d+)', str(e))
            if match:
                tokens_at_limit = int(match.group(1))
                self.total_tokens = tokens_at_limit
                print(f"📊 Token usage at limit: {tokens_at_limit}/{self.total_tokens_limit}")
                
                # Optionally try to estimate prompt vs completion tokens
                # This is a rough estimate assuming some ratio, or you could track running totals
                # For now, we'll just set total_tokens and leave the breakdown unknown
            else:
                print(f"⚠️ Could not parse token count from exception: {e}")
            
            # Try to extract any partial query that was generated
            last_sparql_query = os.getenv('LAST_SPARQL_QUERY', '')
            if last_sparql_query:
                generated_query = last_sparql_query
                print("♻️ Using LAST_SPARQL_QUERY from environment.")

        except Exception as e:
            print(f"❌ Error during query generation: {e}")
            import traceback
            traceback.print_exc()
            generated_query = ""
            tool_calls_exceeded = False

        if not generated_query:
            last_sparql_query = os.getenv('LAST_SPARQL_QUERY', '')
            if last_sparql_query != '':
                generated_query = last_sparql_query
                print("♻️ Using LAST_SPARQL_QUERY from environment.")
            else:
                print("💔 Could not generate a query")

        # -----------------------------------------------------------------
        # Evaluate with exponential backoff
        # unnecessary, but also don't need to remove. 
        # -----------------------------------------------------------------
        
        print("Evaluating generated query...")
        gen_results_obj = sparql_query(generated_query)

        print("Evaluating ground truth query...")
        if ground_truth_sparql:
            gt_results_obj = sparql_query(ground_truth_sparql)        
        # Calculate metrics
        print("Calculating evaluation metrics...")
        arity_f1, entity_set_f1, row_matching_f1, exact_match_f1, best_subset_column_f1 = 0.0, 0.0, 0.0, 0.0, 0.0
        less_columns_flag = False
        
        if gt_results_obj and gt_results_obj["syntax_ok"] and gen_results_obj["syntax_ok"]:
            gold_rows = gt_results_obj["results"]
            pred_rows = gen_results_obj["results"]
            
            arity_f1 = get_arity_matching_f1(generated_query, ground_truth_sparql)
            print('calculated arity f1:', arity_f1)
            entity_set_f1 = get_entity_set_f1(gold_rows=gold_rows, pred_rows=pred_rows)
            print('calculated entity set f1:', entity_set_f1)
            row_matching_f1 = get_row_matching_f1(gold_rows=gold_rows, pred_rows=pred_rows)
            print('calculated row matching f1:', row_matching_f1)
            exact_match_f1 = get_exact_match_f1(gold_rows=gold_rows, pred_rows=pred_rows)
            print('calculated exact match f1:', exact_match_f1)
            best_subset_column_f1 = get_best_subset_column_f1(gold_rows=gold_rows, pred_rows=pred_rows)
            print('calculated best subset column f1:', best_subset_column_f1)
            less_columns_flag = gen_results_obj['col_count'] < gt_results_obj['col_count']
        
        log_entry = {
            **eval_data,
            'model': self.model_name,
            'generated_sparql': generated_query,
            'message_history': "\n".join(self.all_previous_messages),
            'syntax_ok': gen_results_obj['syntax_ok'],
            'returns_results': gen_results_obj['row_count'] > 0,
            'perfect_match': row_matching_f1 == 1.0,
            'gt_num_rows': gt_results_obj['row_count'] if gt_results_obj else 0,
            'gt_num_cols': gt_results_obj['col_count'] if gt_results_obj else 0,
            'gen_num_rows': gen_results_obj['row_count'],
            'gen_num_cols': gen_results_obj['col_count'],
            'arity_matching_f1': arity_f1,
            'entity_set_f1': entity_set_f1,
            'row_matching_f1': row_matching_f1,
            'exact_match_f1': exact_match_f1,
            'best_subset_column_f1': best_subset_column_f1,
            'less_columns_flag': less_columns_flag,
            'tool_calls_exceeded': tool_calls_exceeded,
            'actual_tool_calls': actual_tool_calls,
            'max_tool_calls': self.max_tool_calls,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens
        }

        logger.log(log_entry)
        print(f"📊 Logged results for query_id: {eval_data['query_id']}")
        pprint({'entity_set_f1': entity_set_f1,
            'row_matching_f1': row_matching_f1,
            'exact_match_f1': exact_match_f1,
            'best_subset_column_f1': best_subset_column_f1,
            'total_tokens': self.total_tokens}
            )


def run_agent(
    sparql_endpoint: str,
    eval_data: Dict[str, Any],
    logger: CsvLogger,
    prefixes: str,
    **kwargs
):
    """Convenience function to run the agent."""
    agent = SimpleSparqlAgentMCP(sparql_endpoint, **kwargs)
    asyncio.run(agent.generate_query(eval_data, logger, prefixes))
