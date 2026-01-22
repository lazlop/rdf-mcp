"""
simple_sparql_agent_mcp.py

Simplified SPARQL agent with query review that uses tool calls up to a maximum limit.
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
from pydantic_ai.mcp import MCPServerStdio
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

from agents.kgqa import sparql_query

class SparqlQuery(BaseModel):
    """Model for the query writer agent's output."""
    sparql_query: str = Field(..., description="The generated or revised SPARQL query.")


class QueryCritique(BaseModel):
    """Model for the critique agent's structured feedback."""
    feedback: str = Field(..., description="Natural language feedback explaining the decision and what to improve.")


class SimpleSparqlAgentMCP:
    """
    A SPARQL agent with iterative review that uses MCP tools to generate queries.
    Uses two agents: one to generate queries and one to critique them.
    """

    def __init__(
        self, 
        sparql_endpoint: str,
        model_name: str = "lbl/cborg-coder",
        max_tool_calls: int = 10,
        max_iterations: int = 3,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key_file: Optional[str] = None,
        mcp_server_script: str = "../agents/kgqa.py",
    ):
        """
        Initialize the SPARQL Agent with MCP support and query review.
        
        Args:
            sparql_endpoint: URL of SPARQL endpoint or path to local TTL file
            model_name: Name of the model to use
            max_tool_calls: Maximum number of tool calls allowed per iteration
            max_iterations: Maximum number of query refinement iterations
            api_key: API key for the model provider
            base_url: Base URL for the model provider
            api_key_file: Path to YAML file containing 'key' and 'base_url'
            mcp_server_script: Name of the MCP server script
        """
        self.sparql_endpoint_url = sparql_endpoint
        self.model_name = model_name
        self.max_tool_calls = max_tool_calls
        self.max_iterations = max_iterations
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.messages = []

        # Load API credentials
        if api_key:
            self.api_key = api_key
            self.base_url = base_url
        elif api_key_file:
            with open(api_key_file, 'r') as file:
                config = yaml.safe_load(file)
                self.api_key = config.get('key', api_key)
                self.base_url = config.get('base_url', base_url)
        else:
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            self.base_url = base_url or os.getenv('OPENAI_BASE_URL')

        self.is_remote = sparql_endpoint.lower().startswith("http")

        if self.is_remote:
            raise NotImplementedError("Remote SPARQL endpoints are not supported in this simplified agent.")
        else:
            print(f"🗂️ Local TTL file mode activated. Loading graph from: {self.sparql_endpoint_url}")
            if not os.path.exists(self.sparql_endpoint_url):
                print(f"   -> ❌ ERROR: File not found at {self.sparql_endpoint_url}. Queries will fail.")
                return
            try:
                self.graph = Graph(store="Oxigraph")
                self.graph.parse(self.sparql_endpoint_url, format="turtle")
                print(f"   -> ✅ Graph loaded successfully with {len(self.graph)} triples.")
            except Exception as e:
                print(f"   -> ❌ ERROR: Failed to load or parse the TTL file: {e}")
                self.graph = None

        # Pass graph_file as environment variable to MCP server
        os.environ['GRAPH_FILE'] = self.sparql_endpoint_url
        mcp_env = os.environ.copy()          
        mcp_server_args = [
            "run", "--with", "mcp[cli]", "--with", "rdflib", 
            "--with", "oxrdflib", "mcp", "run", mcp_server_script
        ]
        
        self.mcp_server = MCPServerStdio(
            "uv", 
            args=mcp_server_args,
            env=mcp_env 
        )

        # Set up the model
        self.model = OpenAIModel(
            model_name=self.model_name,
            provider=OpenAIProvider(base_url=self.base_url, api_key=self.api_key),
        )

        # Create query writer agent (uses MCP tools)
        self.query_writer_agent = Agent(
            self.model,
            result_type=SparqlQuery,
            mcp_servers=[self.mcp_server],
            retries=3
        )

        # Create critique agent (no MCP tools needed)
        self.critique_agent = Agent(
            self.model,
            result_type=QueryCritique,
            retries=3
        )

    async def _exponential_backoff(self, coro, *args, delays: List[int] = [5*60, 15*60, 30*60]):
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
        knowledge_graph_content: str
    ) -> None:
        """Generate, review, and evaluate a SPARQL query with iterative refinement."""
        self.prompt_tokens = self.completion_tokens = self.total_tokens = 0
        self.messages = []
        
        nl_question = eval_data['question']
        ground_truth_sparql = eval_data.get('ground_truth_sparql')

        print(f"\n🚀 Starting query generation with review for: '{nl_question}'")
        
        # Initialize tool call tracking
        tool_calls_exceeded = False
        actual_tool_calls = 0
        
        # System prompt for query writer
        query_writer_system_prompt = (
            f"You are a SPARQL developer for Brick Schema and ASHRAE 223p. "
            f"Generate a complete SPARQL query to answer the user's question. "
            f"CALL THE AVAILABLE TOOLS TO GENERATE OR IMPROVE THE QUERY. "
            f"IF THE USER QUESTION MENTIONS SOMETHING IN THE SEMANTIC MODEL, THEN IT MUST BE RETRIEVED IN THE QUERY."
            # f"You may use up to {self.max_tool_calls // 2} tool calls per iteration.\n\n"
            # f"If you receive feedback on a previous attempt, use it to improve your query."
        )

        # System prompt for critique agent
        critique_system_prompt = (
            f"You are an expert in SPARQL, especially for Brick Schema and ASHRAE 223p. "
            f"Your job is to review a SPARQL query and its results based on the original question. "
            f"Decide if the query correctly answers the question or if it needs improvement.\n\n"
            f"Make sure the results include every piece of information mentioned in the question"
            f"Respond with feedback explaining what needs to be improved\n\n"
            # f"Respond with:\n"
            # f"- feedback: Explanation of what needs to be improved\n\n"
            # f"- decision: if you improved the SPARQL query, respond with 'IMPROVE'. If the input SPARQL query doesn't need to be improved respond with 'FINAL'. \n"
            # f"DO NOT RESPOND 'FINAL' IF THE QUERY RETURNS NO RESULTS OR IF YOU SUGGEST ANY CHANGES.\n"
        )

        final_generated_query = ""
        query_writer_history = []
        
        try:
            async with self.query_writer_agent.run_mcp_servers():
                for iteration in range(self.max_iterations):
                    print(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")
                    
                    # Prepare user message for query writer
                    if iteration == 0:
                        user_message = f"Question: {nl_question}"
                    else:
                        user_message = (
                            f"Your previous query received the following feedback: '{last_feedback}'\n\n"
                            f"Please provide an improved query that addresses this feedback.\n"
                            f"Original question: {nl_question}"
                        )
                    
                    print("✍️  Calling Query Writer Agent...")
                    with capture_run_messages() as writer_messages:
                        async def _run_writer():
                            return await self.query_writer_agent.run(
                                user_message,
                                message_history=query_writer_history,
                                system_prompt=query_writer_system_prompt
                            )
                        
                        writer_result = await self._exponential_backoff(_run_writer)
                        
                        # Track tokens
                        if hasattr(writer_result, 'usage'):
                            usage = writer_result.usage()
                            if usage:
                                self.prompt_tokens += usage.request_tokens
                                self.completion_tokens += usage.response_tokens
                                self.total_tokens += usage.total_tokens
                        
                        # Count tool calls from messages
                        # In pydantic-ai, tool calls are in ModelRequest messages with ToolCallPart parts
                        iteration_tool_calls = 0
                        for msg in writer_messages:
                            if hasattr(msg, 'parts'):
                                for part in msg.parts:
                                    # Check if this part is a tool call
                                    if hasattr(part, 'part_kind') and part.part_kind == 'tool-call':
                                        iteration_tool_calls += 1
                                    # Alternative: check the type name
                                    elif type(part).__name__ == 'ToolCallPart':
                                        iteration_tool_calls += 1
                        actual_tool_calls += iteration_tool_calls
                        
                        # Check if tool calls exceeded limit
                        if actual_tool_calls > self.max_tool_calls:
                            tool_calls_exceeded = True
                            print(f"⚠️ Tool call limit exceeded: {actual_tool_calls}/{self.max_tool_calls}")
                            final_generated_query = ""
                        else:
                            final_generated_query = writer_result.data.sparql_query
                            print(f"   -> Generated query (used {iteration_tool_calls} tool calls this iteration, {actual_tool_calls}/{self.max_tool_calls} total):\n{final_generated_query}")
                        
                        if writer_messages:
                            self.messages.extend([str(msg) for msg in writer_messages])
                    
                    # Execute the query to get results for critique
                    print("🔎 Executing query for review...")
                    results_obj = sparql_query(final_generated_query, result_length=-1)
                    
                    results_summary = (
                        f"Syntax OK: {results_obj['syntax_ok']}\n"
                        f"Row Count: {results_obj['row_count']}\n"
                        f"Column Count: {results_obj['col_count']}\n"
                    )
                    
                    if results_obj['syntax_ok']:
                        results_summary += f"First few results: {json.dumps(results_obj['results'][:5], indent=2)}"
                    else:
                        results_summary += f"Error: {results_obj.get('error_message', 'Unknown error')}"
                    
                    print(f"   -> Results summary: {results_summary[:200]}...")
                    
                    # Get critique
                    print("🧐 Calling Critique Agent...")
                    critique_message = (
                        f"Original Question: \"{nl_question}\"\n\n"
                        f"SPARQL Query Attempt:\n```sparql\n{final_generated_query}\n```\n\n"
                        f"Execution Results:\n{results_summary}"
                    )
                    
                    with capture_run_messages() as critique_messages:
                        async def _run_critique():
                            return await self.critique_agent.run(
                                critique_message,
                                system_prompt=critique_system_prompt
                            )
                        
                        critique_result = await self._exponential_backoff(_run_critique)
                        
                        # Track tokens
                        if hasattr(critique_result, 'usage'):
                            usage = critique_result.usage()
                            if usage:
                                self.prompt_tokens += usage.request_tokens
                                self.completion_tokens += usage.response_tokens
                                self.total_tokens += usage.total_tokens
                        
                        critique = critique_result.data
                        # print(f"   -> Critique Decision: {critique.decision}")
                        print(f"   -> Critique Feedback: {critique.feedback}")
                        
                        if critique_messages:
                            self.messages.extend([str(msg) for msg in critique_messages])
                    
                    # # Check if we should stop iterating
                    # if critique.decision == "FINAL":
                    #     print("\n✅ Critique Agent approved the query. Refinement complete.")
                    #     break
                    
                    # Prepare feedback for next iteration
                    last_feedback = critique.feedback
                    
                    # Update query writer history to maintain conversation context
                    query_writer_history = writer_result.new_messages()
                    
                    # If this is the last iteration, use the query anyway
                    if iteration == self.max_iterations - 1:
                        print(f"\n⚠️  Reached maximum iterations ({self.max_iterations}). Using last generated query.")
                        
        except Exception as e:
            print(f"❌ Query generation/review failed: {e}")
            traceback.print_exc()
            final_generated_query = os.getenv("LAST_GENERATED_QUERY")

        if not final_generated_query:
            if tool_calls_exceeded:
                print(f"💔 Could not generate a query - tool call limit exceeded ({actual_tool_calls}/{self.max_tool_calls})")
            else:
                print("💔 Could not generate a query")
            log_entry = {
                **eval_data,
                'model': self.model_name,
                'generated_sparql': '',
                'message_history': "\n".join(self.messages),
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

        # Final evaluation
        print("\n--- Final Evaluation ---")
        gen_results_obj = sparql_query(final_generated_query, result_length=-1)
        if ground_truth_sparql:
            gt_results_obj = sparql_query(ground_truth_sparql, result_length=-1)
        else:
            gt_results_obj = None
            
        # Calculate metrics
        arity_f1, entity_set_f1, row_matching_f1, exact_match_f1, best_subset_column_f1 = 0.0, 0.0, 0.0, 0.0, 0.0
        less_columns_flag = False
        
        if gt_results_obj and gt_results_obj["syntax_ok"] and gen_results_obj["syntax_ok"]:
            gold_rows = gt_results_obj["results"]
            pred_rows = gen_results_obj["results"]
            
            arity_f1 = get_arity_matching_f1(final_generated_query, ground_truth_sparql)
            entity_set_f1 = get_entity_set_f1(gold_rows=gold_rows, pred_rows=pred_rows)
            row_matching_f1 = get_row_matching_f1(gold_rows=gold_rows, pred_rows=pred_rows)
            exact_match_f1 = get_exact_match_f1(gold_rows=gold_rows, pred_rows=pred_rows)
            best_subset_column_f1 = get_best_subset_column_f1(gold_rows=gold_rows, pred_rows=pred_rows)
            less_columns_flag = gen_results_obj['col_count'] < gt_results_obj['col_count']
        
        log_entry = {
            **eval_data,
            'model': self.model_name,
            'generated_sparql': final_generated_query,
            'message_history': "\n".join(self.messages),
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


def run_agent(
    sparql_endpoint: str,
    eval_data: Dict[str, Any],
    logger: CsvLogger,
    prefixes: str,
    knowledge_graph_content: str,
    **kwargs
):
    """Convenience function to run the agent."""
    agent = SimpleSparqlAgentMCP(sparql_endpoint, **kwargs)
    asyncio.run(agent.generate_query(eval_data, logger, prefixes, knowledge_graph_content))
