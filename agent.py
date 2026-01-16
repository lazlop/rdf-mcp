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
from pydantic_ai.mcp import MCPServerStdio
from pyparsing import ParseException
from rdflib import BNode, Graph, Literal, URIRef
from SPARQLWrapper import JSON, SPARQLWrapper

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics import (
    get_arity_matching_f1,
    get_entity_and_row_matching_f1,
    get_exact_match_f1
)

from utils import CsvLogger

from kgqa import sparql_query

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
        sparql_endpoint: str, 
        model_name: str = "lbl/cborg-coder",
        max_tool_calls: int = 10,
        max_iterations: int = 2,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key_file: Optional[str] = None,
        mcp_server_script: str = "kgqa.py",
        graph_file: Optional[str] = None,
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
        self.model_name = model_name
        self.max_tool_calls = max_tool_calls
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

        # Load API credentials
        if api_key_file:
            with open(api_key_file, 'r') as file:
                config = yaml.safe_load(file)
                self.api_key = config.get('key', api_key)
                self.base_url = config.get('base_url', base_url)
        else:
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            self.base_url = base_url or os.getenv('OPENAI_BASE_URL')
        
        self.graph_file = graph_file 
        
        # Pass graph_file as environment variable to MCP server
        os.environ['GRAPH_FILE'] = graph_file
        mcp_env = os.environ.copy()          
        mcp_server_args = [
            "run", "--with", "mcp[cli]", "--with", "rdflib", 
            "--with", "oxrdflib", "mcp", "run", mcp_server_script
        ]
        
        # ADD env parameter
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

        # Create single agent
        self.agent = Agent(
            self.model,
            result_type=SparqlQuery,
            mcp_servers=[self.mcp_server],
            retries=3
        )

        self.graph = Graph()
        self.graph.parse(graph_file, format='turtle')
    async def generate_query(
        self,
        eval_data: Dict[str, Any],
        logger: CsvLogger,
        prefixes: str,
        knowledge_graph_content: str
    ) -> None:
        """Generate and evaluate a SPARQL query."""
        self.prompt_tokens = self.completion_tokens = self.total_tokens = 0
        nl_question = eval_data['question']
        ground_truth_sparql = eval_data.get('ground_truth_sparql')

        print(f"\n🚀 Generating query for: '{nl_question}'")
        
        system_prompt = (
            f"You are an expert SPARQL developer for Brick Schema and ASHRAE 223p. "
            f"Generate a complete SPARQL query to answer the user's question. "
            f"You can use the provided MCP tools to generate the query."
            f"Use the sparql_query tool to ensure the final query is correct before returning the final result."
            f"Use up to {self.max_tool_calls} tool calls if needed.\n\n"
        )

        user_message = f"Question: {nl_question}"
        
        try:
            with capture_run_messages() as messages:
                async with self.agent.run_mcp_servers():
                    result = await self.agent.run(
                        user_message,
                        message_history=[],
                        system_prompt=system_prompt
                    )
                    
                    # Track tokens
                    if hasattr(result, '_usage') and result._usage:
                        self.prompt_tokens += getattr(result._usage, 'request_tokens', 0)
                        self.completion_tokens += getattr(result._usage, 'response_tokens', 0)
                        self.total_tokens += getattr(result._usage, 'total_tokens', 0)
                    
                    generated_query = result.data.sparql_query
                    print(f"✅ Generated query:\n{generated_query}")
                    pprint(messages)
                
        except Exception as e:
            print(f"❌ Query generation failed: {e}")
            traceback.print_exc()
            generated_query = ""

        if not generated_query:
            print("💔 Could not generate a query")
            return

        # Evaluate
        print("\n--- Evaluation ---")
        gen_results_obj = sparql_query(generated_query, result_length = -1)
        gt_results_obj = sparql_query(ground_truth_sparql) if ground_truth_sparql else None
        
        # Calculate metrics
        arity_f1, entity_set_f1, row_matching_f1, exact_match_f1 = 0.0, 0.0, 0.0, 0.0
        less_columns_flag = False
        
        if gt_results_obj and gt_results_obj["syntax_ok"] and gen_results_obj["syntax_ok"]:
            gold_rows = gt_results_obj["results"]
            pred_rows = gen_results_obj["results"]
            
            arity_f1 = get_arity_matching_f1(generated_query, ground_truth_sparql)
            entity_and_row_f1 = get_entity_and_row_matching_f1(gold_rows=gold_rows, pred_rows=pred_rows)
            entity_set_f1 = entity_and_row_f1['entity_set_f1']
            row_matching_f1 = entity_and_row_f1['row_matching_f1']
            exact_match_f1 = get_exact_match_f1(gold_rows=gold_rows, pred_rows=pred_rows)
            less_columns_flag = gen_results_obj['col_count'] < gt_results_obj['col_count']
        
        log_entry = {
            **eval_data,
            'model': self.model_name,
            'generated_sparql': generated_query,
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
            'less_columns_flag': less_columns_flag,
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