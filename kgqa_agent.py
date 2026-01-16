"""
sparql_refinement_agent_mcp.py

Implements the SparqlRefinementAgent class for iterative SPARQL query generation and critique.
This version uses pydantic_ai with MCP server support.
"""

import asyncio
import csv
import itertools
import json
import os
import re
import sys
import time
import traceback
import uuid
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Third-party imports
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerStdio
from pyparsing import ParseException
from rdflib import BNode, Graph, Literal, URIRef
from SPARQLWrapper import JSON, SPARQLWrapper

# Local imports
from metrics import (
    get_arity_matching_f1,
    get_entity_and_row_matching_f1,
    get_exact_match_f1
)

from ReAct_agent.utils import (
    get_kg_subset_content, 
    extract_prefixes_from_ttl, 
    check_if_question_exists, 
    CsvLogger
)

# --- Pydantic Models for the Two-Agent Workflow ---

class SparqlQuery(BaseModel):
    """Model for the Query Writer Agent's output."""
    sparql_query: str = Field(..., description="The generated or revised SPARQL query.")


class QueryCritique(BaseModel):
    """Model for the Critique Agent's structured feedback."""
    decision: str = Field(..., description="The decision, either 'IMPROVE' or 'FINAL'.")
    feedback: str = Field(..., description="Natural language feedback explaining the decision.")


# --- The Orchestrating Agent Class ---

class SparqlRefinementAgentMCP:
    """
    An agent that orchestrates a conversation between a Query Writer and a Critique Agent
    to iteratively develop, evaluate, and log a SPARQL query.
    Can query a remote SPARQL endpoint or a local TTL file.
    Uses pydantic_ai with MCP server support.
    """

    def __init__(
        self, 
        sparql_endpoint: str, 
        model_name: str = "lbl/llama",
        max_iterations: int = 2,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key_file: Optional[str] = None,
        mcp_server_script: str = "brick.py",
        mcp_server_args: Optional[List[str]] = None
    ):
        """
        Initialize the SPARQL Refinement Agent with MCP support.
        
        Args:
            sparql_endpoint: URL of SPARQL endpoint or path to local TTL file
            model_name: Name of the model to use
            max_iterations: Maximum number of refinement iterations
            api_key: API key for the model provider
            base_url: Base URL for the model provider
            api_key_file: Path to YAML file containing 'key' and 'base_url'
            mcp_server_script: Name of the MCP server script (e.g., 'brick.py', 's223.py')
            mcp_server_args: Custom arguments for MCP server (if None, uses default)
        """
        self.sparql_endpoint_url = sparql_endpoint
        self.model_name = model_name
        self.max_iterations = max_iterations
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
                "run",
                "--with",
                "mcp[cli]",
                "--with",
                "rdflib",
                "--with",
                "oxrdflib",
                "mcp",
                "run",
                mcp_server_script
            ]
        
        self.mcp_server = MCPServerStdio("uv", args=mcp_server_args)

        # Set up the model
        self.model = OpenAIModel(
            model_name=self.model_name,
            provider=OpenAIProvider(base_url=self.base_url, api_key=self.api_key),
        )

        # Create agents for query writing and critique
        self.query_writer_agent = Agent(
            self.model,
            result_type=SparqlQuery,
            mcp_servers=[self.mcp_server],
            retries=3
        )

        self.critique_agent = Agent(
            self.model,
            result_type=QueryCritique,
            mcp_servers=[self.mcp_server],
        )

        # --- Differentiate between remote endpoint and local file ---
        self.graph = None
        self.is_remote = sparql_endpoint.lower().startswith("http")

        if self.is_remote:
            print(f"🌐 Remote SPARQL endpoint mode activated: {self.sparql_endpoint_url}")
        else:
            print(f"🗂️ Local TTL file mode activated. Loading graph from: {self.sparql_endpoint_url}")
            if not os.path.exists(self.sparql_endpoint_url):
                print(f"   -> ❌ ERROR: File not found at {self.sparql_endpoint_url}. Queries will fail.")
                return
            try:
                self.graph = Graph()
                self.graph.parse(self.sparql_endpoint_url, format="turtle")
                print(f"   -> ✅ Graph loaded successfully with {len(self.graph)} triples.")
            except Exception as e:
                print(f"   -> ❌ ERROR: Failed to load or parse the TTL file: {e}")
                self.graph = None

    def _format_rdflib_results(self, qres) -> Dict[str, Any]:
        """Converts rdflib QueryResult to the same dict format as SPARQLWrapper."""
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
        """
        Executes a SPARQL query, dispatching to rdflib (local) or SPARQLWrapper (remote).
        Returns a structured dictionary of results.
        """
        print(f"\n🔎 Running SPARQL query... (first 80 chars: {query[:80].replace(chr(10), ' ')}...)")
        
        # --- Branch for local RDF file (rdflib) ---
        if not self.is_remote:
            if self.graph is None:
                return {
                    "summary_string": "SPARQL query failed: The local RDF graph is not loaded.",
                    "results": [],
                    "row_count": 0,
                    "col_count": 0,
                    "syntax_ok": False,
                    "error_message": "Graph not loaded."
                }
            
            try:
                qres = self.graph.query(query)
                formatted_results = self._format_rdflib_results(qres)
                bindings = formatted_results["results"]
                summary = f"Query executed successfully on local graph. Found {len(bindings)} results."
                if not bindings:
                    summary = "The query executed successfully on the local graph but returned no results."
                
                return {
                    "summary_string": summary,
                    "results": bindings,
                    "row_count": len(bindings),
                    "col_count": len(formatted_results["variables"]),
                    "syntax_ok": True,
                    "error_message": None
                }
            except (ParseException, Exception) as e:
                print(f"   -> SPARQL Query (local) Failed: {e}")
                error_msg = f"The query failed to parse with the following error: {str(e)}"
                return {
                    "summary_string": error_msg,
                    "results": [],
                    "row_count": 0,
                    "col_count": 0,
                    "syntax_ok": False,
                    "error_message": str(e)
                }

        # --- Original logic for remote SPARQL endpoint ---
        else:
            try:
                sparql = SPARQLWrapper(self.sparql_endpoint_url)
                sparql.setQuery(query)
                sparql.setReturnFormat(JSON)
                results_json = sparql.query().convert()
                
                bindings = results_json.get("results", {}).get("bindings", [])
                variables = results_json.get("head", {}).get("vars", [])
                
                summary = "Query executed successfully. Here are the first 10 results:\n" + json.dumps(bindings[:10], indent=2)
                if not bindings:
                    summary = "The query executed successfully but returned no results."

                return {
                    "summary_string": summary,
                    "results": bindings,
                    "row_count": len(bindings),
                    "col_count": len(variables),
                    "syntax_ok": True,
                    "error_message": None
                }
            except Exception as e:
                print(f"   -> SPARQL Query (remote) Failed: {e}")
                return {
                    "summary_string": f"The query failed to execute with the following error: {str(e)}",
                    "results": [],
                    "row_count": 0,
                    "col_count": 0,
                    "syntax_ok": False,
                    "error_message": str(e)
                }

    async def _get_query_writer_response(self, system_prompt: str, user_message: str) -> Optional[SparqlQuery]:
        """Call the Query Writer Agent using pydantic_ai."""
        try:
            async with self.query_writer_agent.run_mcp_servers():
                result = await self.query_writer_agent.run(
                    user_message,
                    message_history=[],
                    system_prompt=system_prompt
                )
                
                # Track token usage if available
                if hasattr(result, '_usage') and result._usage:
                    self.prompt_tokens += getattr(result._usage, 'request_tokens', 0)
                    self.completion_tokens += getattr(result._usage, 'response_tokens', 0)
                    self.total_tokens += getattr(result._usage, 'total_tokens', 0)
                
                return result.data
        except Exception as e:
            print(f"Query Writer Agent failed: {e}")
            traceback.print_exc()
            return None

    async def _get_critique_response(self, system_prompt: str, user_message: str) -> Optional[QueryCritique]:
        """Call the Critique Agent using pydantic_ai."""
        try:
            async with self.critique_agent.run_mcp_servers():
                result = await self.critique_agent.run(
                    user_message,
                    message_history=[],
                    system_prompt=system_prompt
                )
                
                # Track token usage if available
                if hasattr(result, '_usage') and result._usage:
                    self.prompt_tokens += getattr(result._usage, 'request_tokens', 0)
                    self.completion_tokens += getattr(result._usage, 'response_tokens', 0)
                    self.total_tokens += getattr(result._usage, 'total_tokens', 0)
                
                return result.data
        except Exception as e:
            print(f"Critique Agent failed: {e}")
            traceback.print_exc()
            return None

    async def refine_and_evaluate_query(
        self,
        eval_data: Dict[str, Any],
        logger: CsvLogger,
        prefixes: str,
        knowledge_graph_content: str
    ) -> None:
        """Main loop to refine, evaluate, and log a query."""
        self.prompt_tokens = self.completion_tokens = self.total_tokens = 0
        nl_question = eval_data['question']
        ground_truth_sparql = eval_data.get('ground_truth_sparql')

        print(f"\n🚀 Starting refinement workflow for question: '{nl_question}'")
        
        query_writer_system_prompt = (
            f"You are an expert SPARQL developer for Brick Schema and ASHRAE 223p. "
            f"Your job is to write a single, complete SPARQL query to answer the user's request. "
            f"Here is a relevant subgraph for your context:\n\n"
            f"```turtle\n{knowledge_graph_content}\n```\n\n"
            f"If you are unsure about how many projections to return, return more rather than fewer. "
            f"If you are given feedback on a prior attempt, use it to revise and improve your query. "
            f"You have access to MCP tools that can help you look up ontology definitions and relationships."
        )

        critique_system_prompt = (
            "You are an expert in SPARQL especially for Brick Schema and ASHRAE 223p. "
            "Your job is to review a SPARQL query and its results based on an original question. "
            "Decide if the query is correct or needs improvement. "
            "You have access to MCP tools that can help you verify ontology concepts."
        )

        log_file_name = os.path.basename(logger.filename).replace('.csv', '')
        print(f"📝 Logging to file: {log_file_name}")

        final_generated_query = ""
        conversation_history = []

        for i in range(self.max_iterations):
            print(f"\n--- Iteration {i + 1} ---")
            print("✍️  Calling Query Writer Agent...")
            
            # Build user message with conversation history
            if i == 0:
                user_message = f"User Question: {nl_question}"
            else:
                user_message = conversation_history[-1]  # Last feedback
            
            query_response = await self._get_query_writer_response(
                query_writer_system_prompt,
                user_message
            )
            
            if not query_response or not query_response.sparql_query:
                print("❌ Query Writer failed to produce a valid query. Aborting iteration.")
                break
            
            final_generated_query = query_response.sparql_query
            print(f"   -> Query received:\n{final_generated_query}")

            results_obj = self._run_sparql_query(final_generated_query)
            print(f"   -> Results Summary: {results_obj['summary_string'][:250]}...")

            print("🧐 Calling Critique Agent...")
            critique_user_message = (
                f"Original Question: \"{nl_question}\"\n\n"
                f"SPARQL Query Attempt:\n```sparql\n{final_generated_query}\n```\n\n"
                f"Execution Results Summary:\n{results_obj['summary_string']}"
            )
            
            critique = await self._get_critique_response(
                critique_system_prompt,
                critique_user_message
            )

            if not critique:
                print("❌ Critique Agent failed. Ending refinement loop.")
                break
            
            print(f"   -> Critique Decision: {critique.decision}")
            print(f"   -> Critique Feedback: {critique.feedback}")

            if critique.decision == "FINAL":
                print("\n✅ Critique Agent approved the query. Refinement complete.")
                break
            
            feedback_for_writer = (
                f"Your last query attempt received the following feedback: '{critique.feedback}'. "
                f"Please provide a new, improved query that addresses this feedback."
            )
            conversation_history.append(feedback_for_writer)
        
        if not final_generated_query:
            print("💔 Agentic workflow could not produce a final query.")
            return

        print("\n--- Final Evaluation and Logging ---")
        gen_results_obj = self._run_sparql_query(final_generated_query)
        gt_results_obj = self._run_sparql_query(ground_truth_sparql) if ground_truth_sparql else None
        
        # Initialize metrics to default values
        arity_f1, entity_set_f1, row_matching_f1, exact_match_f1 = 0.0, 0.0, 0.0, 0.0
        less_columns_flag = False
        
        # Calculate metrics only if both ground truth and generated queries are valid
        if gt_results_obj and gt_results_obj["syntax_ok"] and gen_results_obj["syntax_ok"]:
            gold_rows = gt_results_obj["results"]
            pred_rows = gen_results_obj["results"]
            
            arity_f1 = get_arity_matching_f1(final_generated_query, ground_truth_sparql)
            entity_and_row_f1 = get_entity_and_row_matching_f1(gold_rows=gold_rows, pred_rows=pred_rows)
            entity_set_f1 = entity_and_row_f1['entity_set_f1']
            row_matching_f1 = entity_and_row_f1['row_matching_f1']
            exact_match_f1 = get_exact_match_f1(gold_rows=gold_rows, pred_rows=pred_rows)
            
            # Determine if the generated query returned fewer columns than the ground truth
            less_columns_flag = gen_results_obj['col_count'] < gt_results_obj['col_count']
        
        log_entry = {
            **eval_data,
            'model': self.model_name,
            'generated_sparql': final_generated_query,
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
        print(f"📊 Log entry saved for query_id: {eval_data['query_id']}")


# Convenience function to run the agent
def run_agent(
    sparql_endpoint: str,
    eval_data: Dict[str, Any],
    logger: CsvLogger,
    prefixes: str,
    knowledge_graph_content: str,
    **kwargs
):
    """
    Convenience function to run the SPARQL refinement agent.
    
    Args:
        sparql_endpoint: URL of SPARQL endpoint or path to local TTL file
        eval_data: Evaluation data dictionary
        logger: CSV logger instance
        prefixes: SPARQL prefixes
        knowledge_graph_content: Knowledge graph content in Turtle format
        **kwargs: Additional arguments to pass to SparqlRefinementAgentMCP
    """
    agent = SparqlRefinementAgentMCP(sparql_endpoint, **kwargs)
    asyncio.run(agent.refine_and_evaluate_query(eval_data, logger, prefixes, knowledge_graph_content))
