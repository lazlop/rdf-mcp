from mcp.server.fastmcp import FastMCP
from rdflib import Graph, URIRef, Literal, Namespace, BRICK, RDFS
from rdflib.term import Variable
from typing import List, Optional
from smash import smash_distance
import sys

mcp = FastMCP("GraphDemo", dependencies=["rdflib", "oxrdflib"])
@mcp.tool()
def get_subgraph_with_hops(
    central_node: str,
    num_hops: int = 1,
    get_classes: bool = False
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