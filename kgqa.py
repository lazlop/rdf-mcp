from mcp.server.fastmcp import FastMCP
from rdflib import Graph, URIRef, Literal, Namespace, BRICK, RDFS, RDF, BNode
from rdflib.term import Variable
from typing import List, Optional, Dict, Any
import os 
import sys

mcp = FastMCP("GraphDemo", dependencies=["rdflib", "oxrdflib"])

# ontology can be brick or 223 
ontology = Graph().parse("https://brickschema.org/schema/1.4/Brick.ttl")

graph = None

def _ensure_graph_loaded():
    """Lazy load the graph from GRAPH_FILE environment variable"""
    global graph
    if graph is None:
        graph_file = os.getenv("GRAPH_FILE")
        if not graph_file:
            raise ValueError("GRAPH_FILE environment variable not set")
        if not os.path.exists(graph_file):
            raise FileNotFoundError(f"Graph file not found: {graph_file}")
        
        graph = Graph(store='Oxigraph')
        graph.parse(graph_file)
        print(f"✅ Loaded graph from {graph_file} ({len(graph)} triples)")
    return graph

@mcp.tool()
def describe_entity(
    entity: str | URIRef,
) -> str:
    """
    Describe an entity by extracting its local subgraph.
    """
    g = _ensure_graph_loaded() 

    # originally were args
    num_hops = 1
    get_classes = True
    
    # Convert string to URIRef
    if isinstance(entity, str):
        central_uri = URIRef(entity)
    
    subgraph = Graph(store='Oxigraph')
    
    visited_nodes = set()
    current_layer = {central_uri}
    
    for class_uri in g.objects(central_uri, RDF.type): 
        subgraph.add((central_uri, RDF.type, class_uri))
    
    for hop in range(num_hops):
        next_layer = set()
        
        for node in current_layer:
            if node in visited_nodes:
                continue
            visited_nodes.add(node)
            
            for p, o in g.predicate_objects(node): 
                subgraph.add((node, p, o))
                if isinstance(o, URIRef):
                    next_layer.add(o)
                    for class_uri in g.objects(o, RDF.type): 
                        subgraph.add((o, RDF.type, class_uri))
            
            for s, p in g.subject_predicates(node): 
                subgraph.add((s, p, node))
                if isinstance(s, URIRef):
                    next_layer.add(s)
                    for class_uri in g.objects(s, RDF.type): 
                        subgraph.add((s, RDF.type, class_uri))
        
        current_layer = next_layer
    
    if get_classes:
        for s, p, o in subgraph:
            for class_uri in g.objects(s, RDF.type): 
                subgraph.add((s, RDF.type, class_uri))
            for class_uri in g.objects(o, RDF.type): 
                subgraph.add((o, RDF.type, class_uri))
    
    return subgraph.serialize(format="turtle")
@mcp.tool()
def get_building_summary() -> Dict[str, Any]:
    """
    Get a summary of the building model including:
    - All Brick classes present and their counts
    - All relationship types (predicates) and their counts
    
    Returns a structured overview of what's in the building model.
    """
    g = _ensure_graph_loaded()
    
    # Count entities by Brick class using SPARQL
    class_query = """
    PREFIX brick: <https://brickschema.org/schema/Brick#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    
    SELECT ?class (COUNT(?entity) as ?count)
    WHERE {
        ?entity rdf:type ?class .
    }
    GROUP BY ?class
    ORDER BY DESC(?count)
    """
    
    class_counts = {}
    for row in g.query(class_query):
        class_name = str(row['class']).replace(str(BRICK), "brick:")
        class_counts[class_name] = int(row['count'])
    
    # Count relationships by predicate type using SPARQL
    relationship_query = """
    PREFIX brick: <https://brickschema.org/schema/Brick#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?predicate (COUNT(*) as ?count)
    WHERE {
        ?subject ?predicate ?object .
        FILTER(?predicate != rdf:type)
    }
    GROUP BY ?predicate
    ORDER BY DESC(?count)
    """
    
    relationship_counts = {}
    for row in g.query(relationship_query):
        pred_str = str(row.predicate)
        # todo: prefix namespaces
        pred_name = pred_str        
        relationship_counts[pred_name] = int(row['count'])
    
    summary = (
        f"Building model contains {len(g)} total triples, "
        f"{len(class_counts)} distinct Brick classes, "
        f"and {len(relationship_counts)} distinct relationship types."
    )
    
    return {
        "summary": summary,
        "total_triples": len(g),
        "class_counts": class_counts,
        "relationship_counts": relationship_counts,
        "total_classes": len(class_counts),
        "total_relationship_types": len(relationship_counts)
    }

@mcp.tool()
def find_entities_by_type(brick_class: str, include_subclasses: bool = True) -> Dict[str, Any]:
    """
    Find all entities of a given class type.
    
    Args:
        brick_class: The Brick class name (e.g., 'VAV', 'Temperature_Sensor', 'Air_Handling_Unit')
                    Can be provided with or without 'brick:' prefix
        include_subclasses: If True, also returns entities of subclasses (default: True)
    
    Returns:
        List of all matching entities with their URIs and labels (if available)
    """
    g = _ensure_graph_loaded()
    
    # todo: handle namespaces better
    if brick_class.startswith("brick:"):
        brick_class = brick_class.replace("brick:", "")
    elif brick_class.startswith(str(BRICK)):
        brick_class = brick_class.replace(str(BRICK), "")
    
    # Create the class URI
    class_uri = BRICK[brick_class]
    
    # Check if the class exists in the Brick ontology
    if (class_uri, RDF.type, None) not in ontology and \
       (class_uri, RDFS.subClassOf, None) not in ontology:
        return {
            "summary": f"Warning: '{brick_class}' may not be a valid Brick class",
            "class_searched": f"{brick_class}",
            "entities": [],
            "count": 0,
            "include_subclasses": include_subclasses
        }
    
    entities = []
    
    if include_subclasses:
        # Get all subclasses of the target class from the ontology
        subclasses = set([class_uri])
        
        # Query for transitive subclasses
        for subclass in ontology.transitive_subjects(RDFS.subClassOf, class_uri):
            subclasses.add(subclass)
        
        # Find all entities that are instances of any of these classes
        for target_class in subclasses:
            for entity in g.subjects(RDF.type, target_class):
                entity_info = {
                    "uri": str(entity),
                    "class": str(target_class) #.replace(str(BRICK), "brick:")
                }
                
                # Try to get a label
                label = None
                for lbl in g.objects(entity, RDFS.label):
                    label = str(lbl)
                    break
                
                if label:
                    entity_info["label"] = label
                
                entities.append(entity_info)
    else:
        # Only find direct instances
        for entity in g.subjects(RDF.type, class_uri):
            entity_info = {
                "uri": str(entity),
                "class": f"{brick_class}"
            }
            
            # Try to get a label
            label = None
            for lbl in g.objects(entity, RDFS.label):
                label = str(lbl)
                break
            
            if label:
                entity_info["label"] = label
            
            entities.append(entity_info)
    
    # Remove duplicates (can happen with multiple class assertions)
    seen_uris = set()
    unique_entities = []
    for entity in entities:
        if entity["uri"] not in seen_uris:
            seen_uris.add(entity["uri"])
            unique_entities.append(entity)
    
    summary = (
        f"Found {len(unique_entities)} entities of type {brick_class}"
        f"{' (including subclasses)' if include_subclasses else ''}"
    )
    
    return {
        "summary": summary,
        "class_searched": f"{brick_class}",
        "entities": unique_entities,
        "count": len(unique_entities),
        "include_subclasses": include_subclasses
    }

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
def sparql_query(query: str, result_length: int = 10) -> Dict[str, Any]:
    """
    Executes a SPARQL query, dispatching to rdflib 
    """
    print(f"\n🔎 Running SPARQL query... (first 80 chars: {query[:80].replace(chr(10), ' ')}...)")
    g = _ensure_graph_loaded() 
        
    try:
        qres = g.query(query) 
        formatted_results = _format_rdflib_results(qres)
        bindings = formatted_results["results"][:result_length]
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