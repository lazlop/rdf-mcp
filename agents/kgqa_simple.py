from mcp.server.fastmcp import FastMCP
from rdflib import Graph, URIRef, Literal, Namespace, BRICK, RDFS, RDF, BNode, SH
from rdflib.term import Variable
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from collections import deque
import os 
import sys
import signal

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.namespaces import bind_prefixes, get_prefixes, S223, BRICK

mcp = FastMCP("GraphDemo", dependencies=["rdflib", "oxrdflib"])

# Define namespaces to exclude from shortest path search
QUDT = Namespace("http://qudt.org/schema/qudt/")
EXCLUDED_NAMESPACES = [
    str(RDF),
    str(RDFS),
    str(SH),
    str(QUDT)
]

# Specific predicates to exclude
EXCLUDED_PREDICATES = [
    BRICK.aliasOf,
    BRICK.hasAssociatedTag
]

def _is_excluded_predicate(pred: URIRef) -> bool:
    """Check if a predicate belongs to an excluded namespace or is a specific excluded predicate."""
    # Check if it's a specifically excluded predicate
    if pred in EXCLUDED_PREDICATES:
        return True
    # Check if it belongs to an excluded namespace
    pred_str = str(pred)
    return any(pred_str.startswith(ns) for ns in EXCLUDED_NAMESPACES)

# ontology can be brick or 223 
print("Loading ontology graphs...")
ontology_brick = Graph(store = "Oxigraph").parse("https://brickschema.org/schema/1.4/Brick.ttl")
ontology_s223 = Graph(store = "Oxigraph").parse("https://open223.info/223p.ttl")
ontology = ontology_brick + ontology_s223

graph = None

def _ensure_graph_loaded():
    """Lazy load the graph from GRAPH_FILE environment variable"""
    print("loading graph...")
    global graph
    global parsed_graph
    if graph is None:
        graph_file = os.getenv("GRAPH_FILE")
        parsed_graph_file = os.getenv("PARSED_GRAPH_FILE")
        if not graph_file:
            # Attempt to fall back to a default test graph if the environment variable is not set
            default_path = os.path.join(os.path.dirname(__file__), "test-building.ttl")
            if os.path.exists(default_path):
                graph_file = default_path
            else:
                raise ValueError("GRAPH_FILE environment variable not set and default test graph not found")
        if not os.path.exists(graph_file):
            raise FileNotFoundError(f"Graph file not found: {graph_file}")
        
        graph = Graph(store='Oxigraph')
        graph.parse(graph_file)
        bind_prefixes(graph)
        parsed_graph = Graph(store='Oxigraph')
        parsed_graph.parse(parsed_graph_file)
        bind_prefixes(parsed_graph)
        print(f"✅ Loaded graph from {graph_file} ({len(graph)} triples)")
    return graph, parsed_graph

def _format_term(term) -> Dict[str, Any]:
    """Format an RDF term for JSON serialization"""
    if isinstance(term, URIRef):
        return {'type': 'uri', 'value': str(term)}
    elif isinstance(term, Literal):
        result = {'type': 'literal', 'value': str(term)}
        if term.datatype:
            result['datatype'] = str(term.datatype)
        if term.language:
            result['language'] = term.language
        return result
    elif isinstance(term, BNode):
        return {'type': 'bnode', 'value': str(term)}
    else:
        return {'type': 'unknown', 'value': str(term)}


@mcp.tool()
def describe_entity(
    entity: str | URIRef
) -> str:
    """
    Get detailed information about a specific entity including all its properties, 
    relationships, and type information. Use this when you need to understand what 
    an entity is, what properties it has, or how it connects to other entities.
    
    When to use:
    - After finding an entity URI and needing to see its complete details
    - To correct identifiers in a SPARQL query after it fails
    
    Args:
        entity: The full URI of the entity
    
    Returns:
        Turtle-formatted RDF graph showing the entity and its immediate connections
    """

    g, parsed_graph = _ensure_graph_loaded() 

    # originally were args
    num_hops = 1 # may want to make an arg again, but it just retrieves so much info
    get_classes = False
    
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
    Get an overview of frequent types of entities, literals and relationships exist in the building model.
    USE THIS FIRST when starting to work with an unfamiliar building model or dataset.
    
    When to use:
    - At the start of a conversation to understand what data is available
    - Before writing SPARQL queries to know valid class names
    - To check if certain types of entities exist in the model
    
    Returns:
        Frequent entity types (classes), relationships, and literals
    """
    g, parsed_graph = _ensure_graph_loaded()

    # switching to taking top 50 of each 
    top_n = 50
    percentile = 0  # Exclude bottom 50%
    # Validate percentile
    if not 0.0 <= percentile <= 1.0:
        percentile = 0.0
    
    # Count entities by class using SPARQL
    class_query = """
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
        class_name = str(row['class'])
        class_counts[class_name] = int(row['count'])
    
    # Count relationships by predicate type using SPARQL
    relationship_query = """
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
    
    # Count literals by predicate and datatype using SPARQL
    literal_query = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    
    SELECT ?object ?datatype (COUNT(*) as ?count)
    WHERE {
        ?subject ?predicate ?object .
        FILTER(isLiteral(?object))
        BIND(DATATYPE(?object) AS ?datatype)
    }
    GROUP BY ?object ?datatype
    ORDER BY DESC(?count)
    """
    
    literal_counts = {}
    for row in g.query(literal_query):
        pred_str = str(row.object)
        datatype_str = str(row.datatype) if row.datatype else "plain"
        key = f"{pred_str} ({datatype_str})"
        literal_counts[key] = int(row['count'])
    
    # Apply percentile filtering if requested
    if percentile > 0.0:
        # Filter classes by percentile
        if class_counts:
            class_values = sorted(class_counts.values())
            percentile_index = int(len(class_values) * percentile)
            class_threshold = class_values[percentile_index] if percentile_index < len(class_values) else 0
            class_counts = {k: v for k, v in class_counts.items() if v > class_threshold}
        
        # Filter relationships by percentile
        if relationship_counts:
            rel_values = sorted(relationship_counts.values())
            percentile_index = int(len(rel_values) * percentile)
            rel_threshold = rel_values[percentile_index] if percentile_index < len(rel_values) else 0
            relationship_counts = {k: v for k, v in relationship_counts.items() if v > rel_threshold}
        
        # Filter literals by percentile
        if literal_counts:
            lit_values = sorted(literal_counts.values())
            percentile_index = int(len(lit_values) * percentile)
            lit_threshold = lit_values[percentile_index] if percentile_index < len(lit_values) else 0
            literal_counts = {k: v for k, v in literal_counts.items() if v > lit_threshold}
    
    return {
        "classes": [k for k in class_counts.keys()][:top_n],
        "relationships": [k for k in relationship_counts.keys()][:top_n],
        "literals": [k for k in literal_counts.keys()][:top_n]
    }

def add_limit_to_sparql(query: str, limit: int = 1000) -> str:
    """Add LIMIT clause to a SPARQL query string."""
    query = query.strip()
    
    # Remove trailing semicolon if present
    if query.endswith(';'):
        query = query[:-1].strip()
    
    # Check if LIMIT already exists (case-insensitive)
    if 'LIMIT' in query.upper():
        # Replace existing LIMIT
        import re
        query = re.sub(r'\bLIMIT\s+\d+\b', f'LIMIT {limit}', query, flags=re.IGNORECASE)
    else:
        # Add new LIMIT
        query = f"{query}\nLIMIT {limit}"
    
    return query

@mcp.tool()
def find_entities_by_type(klass: str | URIRef) -> Dict[str, Any]:
    """
    Find all entities of a specific type (class). Use this to discover what specific 
    entities exist in the building model before querying details about them.
    
    When to use:
    - After get_building_summary() to find actual instances of a class
    - To get entity URIs needed for describe_entity()
    - Before writing complex SPARQL queries to verify entities exist
    
    Args:
        klass: Full URI of the class 
    
    Returns:
        List of matching entities with their URIs and labels
    """
    g, parsed_graph = _ensure_graph_loaded()
    class_uri = URIRef(klass)
    include_subclasses = True # may not want this all the time
    # Check if the class exists in the ontology
    if (class_uri, RDF.type, None) not in ontology and \
       (class_uri, RDFS.subClassOf, None) not in ontology:
        return {
            "summary": f"Warning: '{klass}' may not be a valid class in the ontology.",
            "class_searched": f"{klass}",
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
            for entity in parsed_graph.subjects(RDF.type, target_class):
                entity_info = {
                    "uri": str(entity),
                    "class": str(target_class)
                }
                
                # Try to get a label
                label = None
                for lbl in parsed_graph.objects(entity, RDFS.label):
                    label = str(lbl)
                    break
                
                if label:
                    entity_info["label"] = label
                
                entities.append(entity_info)
    else:
        # Only find direct instances
        for entity in parsed_graph.subjects(RDF.type, class_uri):
            entity_info = {
                "uri": str(entity),
                "class": f"{klass}"
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
        f"Found {len(unique_entities)} entities of type {klass}"
        f"{' (including subclasses)' if include_subclasses else ''}"
    )
    
    return {
        # "summary": summary,
        "class_searched": f"{klass}",
        "entities": unique_entities,
        # "count": len(unique_entities),
        # "include_subclasses": include_subclasses
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
def _timeout_handler(signum, frame):
    raise TimeoutError("SPARQL query timed out after 60 seconds")

@mcp.tool()
def sparql_query(query: str, result_length: int = 3) -> Dict[str, Any]:
    """
    Execute custom SPARQL queries for complex questions that other tools cannot answer.
    Only use this for multi-step reasoning, filtering, or complex graph patterns.
    
    When to use:
    - After using simpler tools to better understand complex topologies
    - After using simpler tools to identify specific elements of information needed for a query
    
    When NOT to use:
    - Simple "list all X" questions → use find_entities_by_type() instead
    - Getting details about one entity → use describe_entity() instead
    - Understanding what's in the model → use get_building_summary() instead
    
    Args:
        query: SPARQL SELECT query (prefixes will be added automatically)
        result_length: Maximum results to return (default: 3)
    
    Returns:
        Query results with bindings for each variable
    """
    # Set alarm for timeout
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(60)  # 60 seconds timeout
    try:
        print(f"\n🔎 Running SPARQL query... (first 80 chars: {query[:80].replace(chr(10), ' ')}...)")
        if result_length >= 1:
            query = add_limit_to_sparql(query, limit=result_length)
        else:
            query = add_limit_to_sparql(query, limit=1000)
        g, parsed_graph = _ensure_graph_loaded()
        prefixes = get_prefixes(parsed_graph)
        full_query = prefixes + "\n" + query
        qres = parsed_graph.query(full_query)
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
    except TimeoutError as te:
        print(f"   -> SPARQL Query timed out: {te}")
        return {
            "summary_string": "Query timed out after 60 seconds. Please provide a more specific SPARQL query.",
            "results": [],
            "row_count": 0,
            "col_count": 0,
            "syntax_ok": False,
            "error_message": str(te)
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
    finally:
        signal.alarm(0)