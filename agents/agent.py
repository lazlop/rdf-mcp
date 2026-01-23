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

import yaml
from pydantic import BaseModel, Field
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits, UsageLimitExceeded
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.toolsets.fastmcp import FastMCPToolset
from pyparsing import ParseException
from rdflib import BNode, Graph, Literal, URIRef
from SPARQLWrapper import JSON, SPARQLWrapper

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

from agents.kgqa import sparql_query, mcp, toolset1_mcp


RUN_UNTIL_RESULTS = True  # If True, run until results are found; else, single pass
# RUN_UNTIL_RESULTS = False # Just a single pass 

class SparqlQuery(BaseModel):
    """Model for the agent's output."""
    sparql_query: str = Field(..., description="The generated SPARQL query.")


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
        max_tool_calls: int = 30,
        max_iterations: int = 3,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key_file: Optional[str] = None,
        mcp_server_script: str = "../agents/kgqa.py",
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
        # self.toolset = FastMCPToolset(mcp)
        self.toolset = FastMCPToolset(toolset1_mcp)

        # Load API credentials
        if api_key:
            self.api_key = api_key
            self.base_url = base_url
            self.mcp_server_script = mcp_server_script
        elif api_key_file:
            with open(api_key_file, 'r') as file:
                config = yaml.safe_load(file)
                self.api_key = config.get('key', api_key)
                self.base_url = config.get('base_url', base_url)
                # self.mcp_server_script = config.get('mcp_server_script', mcp_server_script)
        else:
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            self.base_url = base_url or os.getenv('OPENAI_BASE_URL')
            self.mcp_server_script = mcp_server_script

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
        self.model = OpenAIModel(
            model_name=self.model_name,
            provider=OpenAIProvider(base_url=self.base_url, api_key=self.api_key),
        )

        self.limits = UsageLimits(total_tokens_limit = 100000, request_limit = 20)

        recommended_tool_calls = self.max_tool_calls // 3
        system_prompt = (
            f"You are an expert SPARQL developer for Brick Schema and ASHRAE S223. \n"
            f"Your job is to write a single, complete SPARQL query to answer the user's request. "
            f"An example workflow using the available tools may be:\n"
            f"1) use get_building_summary to understand the building model,\n"
            f"2) use get_relationship_between_classes to find predicate paths between classes,\n"
            f"3) look at sparql_snapshots to construct a final query.\n"
            f"If the query is incorrect, you may use describe_entity to understand entities better.\n"
            f"If you are unsure about how many projections to return, return more rather than fewer. "
            # f"ONCE YOU HAVE GENERATED A SUCCESSFUL QUERY THAT ANSWERS THE USER REQUEST, PROVIDE YOUR FINAL ANSWER.\n"
            f"Use up to {recommended_tool_calls} tool calls if needed.\n\n"
        )

        self.agent = Agent(
            self.model,
            output_type=SparqlQuery,
            toolsets = [self.toolset],
            system_prompt=system_prompt,
            retries=10
        )
        print('✅ SimpleSparqlAgentMCP initialized successfully.')

    # -------------------------------------------------------------------------
    # Exponential backoff helper
    # -------------------------------------------------------------------------
    async def _exponential_backoff(self, coro, *args, delays: List[int] = [0, 30, 60, 5*60, 15*60, 30*60]):
        """
        Retry an async callable with exponential backoff.

        Parameters
        ----------
        coro : Callable
            The coroutine function to execute.
        *args :
            Arguments to pass to the coroutine.
        delays : List[int]
            List of delays (in seconds) between retries. Defaults to 5, 15, 30 minutes.

        Returns
        -------
        Any
            The result of the successful coroutine call.

        Raises
        ------
        Exception
            Propagates the last exception if all retries fail.
        """
        last_exc = None
        for i, delay in enumerate(delays):
            try:
                return await coro(*args)
            except Exception as exc:
                last_exc = exc
                if i == len(delays) - 1:
                    break
                await asyncio.sleep(delay)
        raise last_exc

    async def generate_query(
        self,
        eval_data: Dict[str, Any],
        logger: CsvLogger,
        prefixes: str,
    ) -> None:
        """Generate and evaluate a SPARQL query."""
        self.prompt_tokens = self.completion_tokens = self.total_tokens = 0
        
        generated_query = ""
        tool_calls_exceeded = False
        actual_tool_calls = 0
        
        nl_question = eval_data['question']
        ground_truth_sparql = eval_data.get('ground_truth_sparql')

        print(f"\n🚀 Generating query for: '{nl_question}'")

        user_message = f"Question: {nl_question}"
        try:
            self.all_previous_messages = []
            all_messages = []
            message_history = []
            for i in range(self.max_iterations):
                iteration_tool_calls = 0
                with capture_run_messages() as messages:
                    async def _run_agent(message_history=[]):
                        return await self.agent.run(
                            user_message,
                            message_history=message_history,
                            usage_limits=self.limits, 
                        )
                    # result = await self._exponential_backoff(_run_agent)
                    result = await _run_agent()
                    # track tokens
                    if hasattr(result, 'usage'):
                        usage = result.usage()
                        if usage:
                            self.prompt_tokens += usage.input_tokens
                            self.completion_tokens += usage.output_tokens
                            self.total_tokens += usage.total_tokens
                    
                    for msg in messages:
                        if hasattr(msg, 'parts'):
                            for part in msg.parts:
                                if hasattr(part, 'part_kind') and part.part_kind == 'tool-call':
                                    iteration_tool_calls += 1
                                elif type(part).__name__ == 'ToolCallPart':
                                    iteration_tool_calls += 1
                    actual_tool_calls += iteration_tool_calls
                    
                    if actual_tool_calls > self.max_tool_calls:
                        tool_calls_exceeded = True
                        print(f"⚠️ Tool call limit exceeded: {actual_tool_calls}/{self.max_tool_calls}")
                        generated_query = ""
                    else:
                        generated_query = result.output.sparql_query
                        print(f"✅ Generated query (used {actual_tool_calls}/{self.max_tool_calls} tool calls):\n{generated_query}")
                    
                    query_results = sparql_query(generated_query, result_length=10)
                    self.all_previous_messages += [str(msg) for msg in messages]
                    
                    if query_results and query_results['row_count'] > 0:
                        print(f"   -> Query returned {query_results['row_count']} results.")
                        break
                    else:
                        print(f"   -> Query returned no results. Retrying ({i+1}/{self.max_iterations})...")
                        # Update user_message for next iteration with failed query context
                        user_message = (
                            f"Question: {nl_question}\n\n"
                            f"Previous attempt returned no results\n"
                            f"Query: {generated_query}\n"
                            f"Result: {json.dumps(query_results)}\n\n"
                            f"Please try again, and use the available tools to ensure that the query returns the correct data."
                        )
                if not RUN_UNTIL_RESULTS:
                    break
        except Exception as e:
            print(f"❌ Query generation failed: {e}")
            self.all_previous_messages += [str(msg) for msg in messages]
            traceback.print_exc()
            generated_query = os.getenv("LAST_SPARQL_QUERY")

        if not generated_query:
            if tool_calls_exceeded:
                print(f"💔 Could not generate a query - tool call limit exceeded ({actual_tool_calls}/{self.max_tool_calls})")
            else:
                print("💔 Could not generate a query")
            log_entry = {
            **eval_data,
            'model': self.model_name,
            'generated_sparql': generated_query,
            'message_history': "\n".join(self.all_previous_messages),
            'syntax_ok': False,
            'returns_results': False,
            'perfect_match': False,
            'gt_num_rows': 0,
            'gt_num_cols': 0,
            'gen_num_rows': 0,
            'gen_num_cols': 0,
            'arity_matching_f1': 0.0,
            'entity_set_f1': 0.0,
            'row_matching_f1': 0.0,
            'exact_match_f1': 0.0,
            'best_subset_column_f1': 0.0,
            'less_columns_flag': True,
            'tool_calls_exceeded': tool_calls_exceeded,
            'actual_tool_calls': actual_tool_calls,
            'max_tool_calls': self.max_tool_calls,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens
        }
            logger.log(log_entry)
            return

        # -----------------------------------------------------------------
        # Evaluate with exponential backoff
        # unnecessary, but also don't need to remove. 
        # -----------------------------------------------------------------
        
        print("Evaluating generated query...")
        print(generated_query)
        gen_results_obj = sparql_query(generated_query)

        print("Evaluating ground truth query...")
        print(ground_truth_sparql)
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
            entity_set_f1 = get_entity_set_f1(gold_rows=gold_rows, pred_rows=pred_rows)
            row_matching_f1 = get_row_matching_f1(gold_rows=gold_rows, pred_rows=pred_rows)
            exact_match_f1 = get_exact_match_f1(gold_rows=gold_rows, pred_rows=pred_rows)
            best_subset_column_f1 = get_best_subset_column_f1(gold_rows=gold_rows, pred_rows=pred_rows)
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
