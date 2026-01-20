"""
simple_sparql_agent_mcp.py

Simplified SPARQL agent that uses tool calls up to a maximum limit.
"""

import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_ai import Agent
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

from ReAct_agent.utils import CsvLogger


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
        model_name: str = "lbl/llama",
        max_tool_calls: int = 10,
        max_iterations: int = 2,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key_file: Optional[str] = None,
        mcp_server_script: str = "brick.py",
        mcp_server_args: Optional[List[str]] = None
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

        # Set up MCP server
        if mcp_server_args is None:
            mcp_server_args = [
                "run", "--with", "mcp[cli]", "--with", "rdflib", 
                "--with", "oxrdflib", "mcp", "run", mcp_server_script
            ]
        
        self.mcp_server = MCPServerStdio("uv", args=mcp_server_args)

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

        # Determine if remote or local
        self.graph = None
        self.is_remote = sparql_endpoint.lower().startswith("http")

        if self.is_remote:
            print(f"🌐 Remote SPARQL endpoint: {self.sparql_endpoint_url}")
        else:
            print(f"🗂️ Local TTL file mode: {self.sparql_endpoint_url}")
            if not os.path.exists(self.sparql_endpoint_url):
                print(f"❌ File not found: {self.sparql_endpoint_url}")
                return
            try:
                self.graph = Graph()
                self.graph.parse(self.sparql_endpoint_url, format="turtle")
                print(f"✅ Graph loaded: {len(self.graph)} triples")
            except Exception as e:
                print(f"❌ Failed to load graph: {e}")
                self.graph = None

    def _format_rdflib_results(self, qres) -> Dict[str, Any]:
        """Convert rdflib QueryResult to SPARQLWrapper format."""
        variables = [str(v) for v in qres.vars]
        bindings = []
        for row in qres:
            binding_row = {}
            for var_name in variables:
                term = row[var_name]
                if term is None:
                    continue
                
                term_dict = {}
                if isinstance(term, URIRef):
                    term_dict = {'type': 'uri', 'value': str(term)}
                elif isinstance(term, Literal):
                    term_dict = {'type': 'literal', 'value': str(term)}
                    if term.datatype:
                        term_dict['datatype'] = str(term.datatype)
                    if term.language:
                        term_dict['xml:lang'] = term.language
                elif isinstance(term, BNode):
                    term_dict = {'type': 'bnode', 'value': str(term)}
                
                binding_row[var_name] = term_dict
            bindings.append(binding_row)
        
        return {"results": bindings, "variables": variables}

    def _run_sparql_query(self, query: str) -> Dict[str, Any]:
        """Execute SPARQL query on local graph or remote endpoint."""
        print(f"\n🔎 Running query: {query[:80]}...")
        
        if not self.is_remote:
            if self.graph is None:
                return {
                    "summary_string": "Graph not loaded",
                    "results": [],
                    "row_count": 0,
                    "col_count": 0,
                    "syntax_ok": False,
                    "error_message": "Graph not loaded"
                }
            
            try:
                qres = self.graph.query(query)
                formatted_results = self._format_rdflib_results(qres)
                bindings = formatted_results["results"]
                summary = f"Found {len(bindings)} results"
                
                return {
                    "summary_string": summary,
                    "results": bindings,
                    "row_count": len(bindings),
                    "col_count": len(formatted_results["variables"]),
                    "syntax_ok": True,
                    "error_message": None
                }
            except Exception as e:
                return {
                    "summary_string": f"Query failed: {str(e)}",
                    "results": [],
                    "row_count": 0,
                    "col_count": 0,
                    "syntax_ok": False,
                    "error_message": str(e)
                }
        else:
            try:
                sparql = SPARQLWrapper(self.sparql_endpoint_url)
                sparql.setQuery(query)
                sparql.setReturnFormat(JSON)
                results_json = sparql.query().convert()
                
                bindings = results_json.get("results", {}).get("bindings", [])
                variables = results_json.get("head", {}).get("vars", [])
                summary = f"Found {len(bindings)} results"
                
                return {
                    "summary_string": summary,
                    "results": bindings,
                    "row_count": len(bindings),
                    "col_count": len(variables),
                    "syntax_ok": True,
                    "error_message": None
                }
            except Exception as e:
                return {
                    "summary_string": f"Query failed: {str(e)}",
                    "results": [],
                    "row_count": 0,
                    "col_count": 0,
                    "syntax_ok": False,
                    "error_message": str(e)
                }

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
            f"You can use the provided MCP tools to look up ontology definitions. "
            f"Use up to {self.max_tool_calls} tool calls if needed.\n\n"
            f"Knowledge graph context:\n```turtle\n{knowledge_graph_content}\n```"
        )

        user_message = f"Question: {nl_question}"
        
        try:
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
                
        except Exception as e:
            print(f"❌ Query generation failed: {e}")
            traceback.print_exc()
            generated_query = ""

        if not generated_query:
            print("💔 Could not generate a query")
            return

        # Evaluate
        print("\n--- Evaluation ---")
        gen_results_obj = self._run_sparql_query(generated_query)
        gt_results_obj = self._run_sparql_query(ground_truth_sparql) if ground_truth_sparql else None
        
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