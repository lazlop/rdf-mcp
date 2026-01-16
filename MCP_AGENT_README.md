# SPARQL Refinement Agent with MCP Support

This document describes the MCP-enabled version of the SPARQL Refinement Agent (`kgqa_agent.py`), which integrates Model Context Protocol (MCP) servers to enhance SPARQL query generation with ontology-aware tools.

## Overview

The `SparqlRefinementAgentMCP` class is an enhanced version of the original `SparqlRefinementAgent` that uses:
- **pydantic_ai** for agent orchestration
- **MCP servers** to provide ontology lookup tools (Brick Schema, ASHRAE 223p, etc.)
- **Async/await** pattern for better performance
- Same evaluation metrics and logging as the original

## Key Differences from Original Agent

| Feature | Original (`kgqa_agent.py`) | MCP Version (`kgqa_agent.py`) |
|---------|---------------------------|----------------------------------|
| LLM Client | OpenAI client | pydantic_ai Agent |
| Tool Access | None | MCP server tools |
| Execution | Synchronous | Asynchronous |
| Response Handling | Manual JSON parsing | Pydantic models via pydantic_ai |
| Ontology Lookup | Static context only | Dynamic via MCP tools |

## Installation

Ensure you have the required dependencies:

```bash
pip install pydantic-ai pydantic rdflib oxrdflib SPARQLWrapper pyyaml
```

Or if using `uv`:

```bash
uv pip install pydantic-ai pydantic rdflib oxrdflib SPARQLWrapper pyyaml
```

## Basic Usage

### Method 1: Direct Async Usage

```python
import asyncio
from kgqa_agent import SparqlRefinementAgentMCP
from ReAct_agent.utils import CsvLogger

async def main():
    # Initialize the agent
    agent = SparqlRefinementAgentMCP(
        sparql_endpoint="path/to/your/file.ttl",
        model_name="lbl/llama",
        max_iterations=5,
        api_key_file="/path/to/api_key.yaml",
        mcp_server_script="brick.py"
    )
    
    # Prepare evaluation data
    eval_data = {
        'query_id': 'example_001',
        'question': 'What are all the temperature sensors?',
        'ground_truth_sparql': None
    }
    
    # Knowledge graph content
    kg_content = """
    @prefix brick: <https://brickschema.org/schema/Brick#> .
    ex:sensor1 a brick:Temperature_Sensor .
    """
    
    prefixes = "PREFIX brick: <https://brickschema.org/schema/Brick#>"
    logger = CsvLogger("results.csv")
    
    # Run the agent
    await agent.refine_and_evaluate_query(
        eval_data=eval_data,
        logger=logger,
        prefixes=prefixes,
        knowledge_graph_content=kg_content
    )

asyncio.run(main())
```

### Method 2: Convenience Function

```python
from kgqa_agent import run_agent
from ReAct_agent.utils import CsvLogger

# Prepare data
eval_data = {'query_id': '001', 'question': 'List all sensors'}
kg_content = "..."
prefixes = "..."
logger = CsvLogger("results.csv")

# Run (handles async internally)
run_agent(
    sparql_endpoint="path/to/file.ttl",
    eval_data=eval_data,
    logger=logger,
    prefixes=prefixes,
    knowledge_graph_content=kg_content,
    model_name="lbl/llama",
    api_key_file="/path/to/api_key.yaml",
    mcp_server_script="brick.py"
)
```

## Configuration Options

### Constructor Parameters

```python
SparqlRefinementAgentMCP(
    sparql_endpoint: str,           # SPARQL endpoint URL or local TTL file path
    model_name: str = "lbl/llama",  # Model name
    max_iterations: int = 5,        # Max refinement iterations
    api_key: Optional[str] = None,  # API key (or use api_key_file)
    base_url: Optional[str] = None, # Base URL (or use api_key_file)
    api_key_file: Optional[str] = None,  # YAML file with 'key' and 'base_url'
    mcp_server_script: str = "brick.py",  # MCP server script name
    mcp_server_args: Optional[List[str]] = None  # Custom MCP args
)
```

### API Key Configuration

**Option 1: YAML file**
```yaml
# api_key.yaml
key: "your-api-key"
base_url: "http://localhost:1234/v1"
```

```python
agent = SparqlRefinementAgentMCP(
    sparql_endpoint="...",
    api_key_file="/path/to/api_key.yaml"
)
```

**Option 2: Direct parameters**
```python
agent = SparqlRefinementAgentMCP(
    sparql_endpoint="...",
    api_key="your-api-key",
    base_url="http://localhost:1234/v1"
)
```

**Option 3: Environment variables**
```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="http://localhost:1234/v1"
```

```python
agent = SparqlRefinementAgentMCP(sparql_endpoint="...")
```

### MCP Server Configuration

**Using Brick Schema (default):**
```python
agent = SparqlRefinementAgentMCP(
    sparql_endpoint="...",
    mcp_server_script="brick.py"
)
```

**Using ASHRAE 223p:**
```python
agent = SparqlRefinementAgentMCP(
    sparql_endpoint="...",
    mcp_server_script="s223.py"
)
```

**Custom MCP arguments:**
```python
custom_args = [
    "run",
    "--with", "mcp[cli]",
    "--with", "rdflib",
    "--with", "custom-package",
    "mcp",
    "run",
    "brick.py"
]

agent = SparqlRefinementAgentMCP(
    sparql_endpoint="...",
    mcp_server_args=custom_args
)
```

## How It Works

### Architecture

```
┌─────────────────────────────────────┐
│  SparqlRefinementAgentMCP           │
│                                     │
│  ┌───────────────────────────────┐ │
│  │  Query Writer Agent           │ │
│  │  (pydantic_ai + MCP tools)    │ │
│  └───────────────────────────────┘ │
│                                     │
│  ┌───────────────────────────────┐ │
│  │  Critique Agent               │ │
│  │  (pydantic_ai + MCP tools)    │ │
│  └───────────────────────────────┘ │
│                                     │
│  ┌───────────────────────────────┐ │
│  │  SPARQL Executor              │ │
│  │  (rdflib or SPARQLWrapper)    │ │
│  └───────────────────────────────┘ │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  MCP Server (brick.py / s223.py)    │
│  - Ontology lookup tools            │
│  - Class/property definitions       │
│  - Relationship queries             │
└─────────────────────────────────────┘
```

### Workflow

1. **Initialization**: Agent loads the knowledge graph and sets up MCP server
2. **Query Generation**: Query Writer Agent generates SPARQL query
   - Can use MCP tools to look up ontology definitions
   - Receives knowledge graph context
3. **Query Execution**: Query is executed against the endpoint/file
4. **Critique**: Critique Agent evaluates the query and results
   - Can use MCP tools to verify ontology concepts
   - Decides: FINAL or IMPROVE
5. **Iteration**: If IMPROVE, feedback is sent back to Query Writer
6. **Evaluation**: Final query is evaluated against ground truth (if provided)
7. **Logging**: Results and metrics are logged to CSV

## MCP Tools Available

The agent has access to MCP tools provided by the server script:

### Brick Schema (brick.py)
- Look up class definitions
- Query property relationships
- Explore class hierarchy
- Validate Brick concepts

### ASHRAE 223p (s223.py)
- Look up 223p class definitions
- Query connection point types
- Explore equipment relationships
- Validate 223p concepts

## Evaluation Metrics

The agent computes the same metrics as the original:

- **syntax_ok**: Whether the query is syntactically valid
- **returns_results**: Whether the query returns any results
- **perfect_match**: Whether results exactly match ground truth
- **arity_matching_f1**: F1 score for number of columns
- **entity_set_f1**: F1 score for entity sets
- **row_matching_f1**: F1 score for row matching
- **exact_match_f1**: F1 score for exact matches
- **Token usage**: Prompt, completion, and total tokens

## Migration from Original Agent

To migrate from `SparqlRefinementAgent` to `SparqlRefinementAgentMCP`:

1. **Update imports:**
   ```python
   # Old
   from kgqa_agent import SparqlRefinementAgent
   
   # New
   from kgqa_agent import SparqlRefinementAgentMCP
   ```

2. **Update initialization:**
   ```python
   # Old
   agent = SparqlRefinementAgent(
       sparql_endpoint="...",
       model_name="...",
       client=openai_client
   )
   
   # New
   agent = SparqlRefinementAgentMCP(
       sparql_endpoint="...",
       model_name="...",
       api_key_file="..."  # or api_key/base_url
   )
   ```

3. **Update method calls to async:**
   ```python
   # Old
   agent.refine_and_evaluate_query(eval_data, logger, prefixes, kg_content)
   
   # New
   await agent.refine_and_evaluate_query(eval_data, logger, prefixes, kg_content)
   ```

4. **Wrap in async function:**
   ```python
   async def main():
       agent = SparqlRefinementAgentMCP(...)
       await agent.refine_and_evaluate_query(...)
   
   asyncio.run(main())
   ```

   Or use the convenience function:
   ```python
   run_agent(sparql_endpoint, eval_data, logger, prefixes, kg_content, ...)
   ```

## Advantages of MCP Version

1. **Ontology-Aware**: Agent can dynamically look up ontology definitions
2. **Better Accuracy**: Access to tools improves query correctness
3. **Extensible**: Easy to add new MCP servers for different ontologies
4. **Modern Stack**: Uses pydantic_ai for better type safety and validation
5. **Async Support**: Better performance for I/O-bound operations

## Troubleshooting

### MCP Server Not Starting
- Ensure `uv` is installed: `pip install uv`
- Check MCP server script exists (brick.py, s223.py)
- Verify dependencies are installed

### Token Usage Not Tracked
- Token tracking depends on the model provider
- Some providers may not return usage information
- Check if `result._usage` is available

### Query Execution Fails
- Verify SPARQL endpoint is accessible (if remote)
- Check TTL file exists and is valid (if local)
- Ensure prefixes are correctly defined

## Examples

See `example_mcp_usage.py` for complete working examples including:
- Basic usage with Brick Schema
- ASHRAE 223p usage
- Custom MCP server configuration
- Convenience function usage

## License

Same as the original project.
