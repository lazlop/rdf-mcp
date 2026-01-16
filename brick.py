from mcp.server.fastmcp import FastMCP
from rdflib import Graph, URIRef, Literal, Namespace, BRICK, RDFS
from rdflib.term import Variable
from typing import List, Optional
from smash import smash_distance
import sys

mcp = FastMCP("GraphDemo", dependencies=["rdflib", "oxrdflib"])

S223 = Namespace("http://data.ashrae.org/standard223#")
ontology = Graph().parse("https://brickschema.org/schema/1.4/Brick.ttl")


# @mcp.tool()
# def expand_abbreviation(abbreviation: str) -> list[str]:
#     """Expand an abbreviation to its full form using the Brick ontology"""
#     # return the top 5 matches from the class dictionary
#     closest_matches = sorted(CLASS_DICT.keys(), key=lambda x: smash_distance(abbreviation, x))[:5]
#     print(f"closest match to {abbreviation} is {closest_matches}", file=sys.stderr)
#     return closest_matches


# @mcp.tool()
# def get_terms() -> list[str]:
#     """Get all terms in the Brick ontology graph"""
#     query = """
#     PREFIX owl: <http://www.w3.org/2002/07/owl#>
#     PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
#     PREFIX brick: <https://brickschema.org/schema/Brick#>
#     PREFIX s223: <http://data.ashrae.org/standard223#>
#     SELECT ?class WHERE {
#         { ?class a owl:Class }
#         UNION
#         { ?class a rdfs:Class }
#         FILTER NOT EXISTS { ?class owl:deprecated true }
#         FILTER NOT EXISTS { ?class brick:aliasOf ?alias }
#     }"""
#     results = ontology.query(query)
#     # return [str(row[0]).split('#')[-1] for row in results]
#     r = [str(row[0]).split("#")[-1] for row in results]
#     return r


# @mcp.tool()
# def get_properties() -> list[str]:
#     """Get all properties in the Brick ontology graph"""
#     query = """
#     PREFIX owl: <http://www.w3.org/2002/07/owl#>
#     PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
#     PREFIX s223: <http://data.ashrae.org/standard223#>
#     PREFIX brick: <https://brickschema.org/schema/Brick#>
#     PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
#     SELECT ?prop WHERE {
#         { ?prop rdfs:subPropertyOf ?property }
#         UNION
#         { ?prop a owl:ObjectProperty }
#         UNION
#         { ?prop a owl:DataProperty }
#     }"""
#     results = ontology.query(query)
#     # return [str(row[0]).split('#')[-1] for row in results]
#     r = [str(row[0]).split("#")[-1] for row in results]
#     return r


# @mcp.tool()
# def get_possible_properties(class_: str) -> list[tuple[str, str]]:
#     """Returns pairs of possible (property, object type) for a given brick class"""
#     query = """
#     PREFIX owl: <http://www.w3.org/2002/07/owl#>
#     PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
#     PREFIX s223: <http://data.ashrae.org/standard223#>
#     PREFIX brick: <https://brickschema.org/schema/Brick#>
#     PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
#     SELECT ?path ?type WHERE {
#         ?from brick:aliasOf?/rdfs:subClassOf* ?fromp .
#         { ?shape brick:aliasOf?/sh:targetClass ?fromp }
#         UNION
#         { ?fromp a sh:NodeShape . BIND(?fromp as ?shape) }
#         ?shape sh:property ?prop .
#         ?prop sh:path ?path .
#          FILTER (!isBlank(?path))
#         OPTIONAL { { ?prop sh:node ?type } UNION { ?prop sh:class ?type } }
#     }
#     """
#     res = list(ontology.query(query, initBindings={"from": BRICK[class_]}).bindings)
#     print(res, file=sys.stderr)
#     path_object_pairs = set([(r[Variable("path")], r[Variable("type")]) for r in res])
#     return list(path_object_pairs)


# @mcp.tool()
# def get_definition_brick(class_: str) -> str:
#     """Get the definition of cyber-physical concepts from the Brick ontology."""
#     return ontology.cbd(BRICK[class_]).serialize(format="turtle")

# @mcp.resource("rdf://describe/{term}")
# def get_definition(term: str) -> str:
#     """Get the definition of cyber-physical concepts like sensors from the Brick ontology."""
#     return ontology.cbd(BRICK[term]).serialize(format="turtle")



# # build a dictionary of all classes in the Brick ontology
# def build_class_dict() -> dict[str, str]:
#     """Build a dictionary of all classes in the Brick ontology"""
#     class_dict = {}
#     for term in get_terms():
#         class_uri = BRICK[term]
#         label = ontology.value(subject=class_uri, predicate=RDFS.label)
#         if label:
#             label = str(label)
#         else:
#             label = str(term).split("#")[-1]
#         class_dict[label] = term
#     return class_dict


# CLASS_DICT = build_class_dict()
# print(f"{CLASS_DICT}", file=sys.stderr)

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

print("mcp ready", file=sys.stderr)

if __name__ == "__main__":
    mcp.run()
