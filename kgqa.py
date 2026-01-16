from mcp.server.fastmcp import FastMCP
from rdflib import Graph, URIRef, Literal, Namespace, BRICK, RDFS, RDF, BNode
from rdflib.term import Variable
from typing import List, Optional, Dict, Any
from smash import smash_distance
import sys

mcp = FastMCP("GraphDemo", dependencies=["rdflib", "oxrdflib"])

# ontology can be brick or 223 
ontology = Graph().parse("https://brickschema.org/schema/1.4/Brick.ttl")
@mcp.tool()
def get_subgraph_with_hops(
    central_node: str | URIRef,
    graph: Graph,
    num_hops: int = 1,
    get_classes: bool = False,
) -> str:
    """
    Extract a subgraph containing all triples within num_hops from central_node,
    optionally including rdf:type information for all entities.
    
    Args:
        central_node: The central node URI as string (will be converted to Brick namespace)
        num_hops: Number of hops to traverse from the central node (default: 1)
        get_classes: Whether to include rdf:type information for all entities (default: False)
    
    Returns:
        Serialized Turtle representation of the subgraph
    """
    # Convert string to URIRef in Brick namespace
    if isinstance(central_node, str):
        if central_node.startswith("http://") or central_node.startswith("https://"):
            central_uri = URIRef(central_node)
        else:
            central_uri = BRICK[central_node]
    else:
        central_uri = URIRef(central_node)
    
    subgraph = Graph(store='Oxigraph')
    visited_nodes = set()
    current_layer = {central_uri}
    
    # Add class information for central node
    for class_uri in ontology.objects(central_uri, RDF.type):
        subgraph.add((central_uri, RDF.type, class_uri))
    
    for hop in range(num_hops):
        next_layer = set()
        
        for node in current_layer:
            if node in visited_nodes:
                continue
            visited_nodes.add(node)
            
            # Get triples where node is subject
            for p, o in ontology.predicate_objects(node):
                subgraph.add((node, p, o))
                if isinstance(o, URIRef):
                    next_layer.add(o)
                    # Add class information for object
                    for class_uri in ontology.objects(o, RDF.type):
                        subgraph.add((o, RDF.type, class_uri))
            
            # Get triples where node is object
            for s, p in ontology.subject_predicates(node):
                subgraph.add((s, p, node))
                if isinstance(s, URIRef):
                    next_layer.add(s)
                    # Add class information for subject
                    for class_uri in ontology.objects(s, RDF.type):
                        subgraph.add((s, RDF.type, class_uri))
        
        current_layer = next_layer
    
    if get_classes:
        for s, p, o in subgraph:
            for class_uri in ontology.objects(s, RDF.type):
                subgraph.add((s, RDF.type, class_uri))
            for class_uri in ontology.objects(o, RDF.type):
                subgraph.add((o, RDF.type, class_uri))
    
    return subgraph.serialize(format="turtle")

def _format_rdflib_results(qres) -> Dict[str, Any]:
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
@mcp.tool()
def run_sparql_query(query: str, graph: Graph) -> Dict[str, Any]:
    """
    Executes a SPARQL query, dispatching to rdflib 
    """
    print(f"\n🔎 Running SPARQL query... (first 80 chars: {query[:80].replace(chr(10), ' ')}...)")

    try:
        qres = graph.query(query)
        formatted_results = _format_rdflib_results(qres)
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
    except Exception as e:
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