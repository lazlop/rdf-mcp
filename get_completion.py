import asyncio
from devtools import pprint
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerStdio
import yaml

server = MCPServerStdio(
    "uv",
    args=[
        "run",
        "--with",
        "mcp[cli]",
        "--with",
        "rdflib",
        "--with",
        "oxrdflib",
        "mcp",
        "run",
        "brick.py"
    ],
)

with open('/Users/lazlopaul/Desktop/cborg/api_key.yaml', 'r') as file:
    config = yaml.safe_load(file)
    API_KEY = config['key']
    BASE_URL = config['base_url']

model = OpenAIModel(
        model_name="lbl/llama",
        # i'm using LM Studio here, but you could use any other provider that exposes
        # an OpenAI-like API
        provider=OpenAIProvider(base_url=BASE_URL, api_key=API_KEY),
    )

agent = Agent(
    model,
    mcp_servers=[server],
)

prompt = """Create a simple Brick model of a AHU box with 3 sensors: RAT, SAT and OAT. Also include a SF with a SF command

Look up definitions of concepts and their relationships to ensure you are building a valid Brick model.
Use the tool to determine what properties a term can have. Only use the predicates defined by the ontology.
Output a turtle file with the Brick model.
"""

prompt = """what is the brick class that represents an air handling unit. Provide a final response with only the brick class name
"""
async def main():
    with capture_run_messages() as messages:
        async with agent.run_mcp_servers():
            result = await agent.run(prompt)
    pprint(messages)
    print(result.output)

asyncio.run(main())