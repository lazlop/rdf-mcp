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
ontology_brick = Graph(store = "Oxigraph").parse("https://brickschema.org/schema/1.4/Brick.ttl")
ontology_s223 = Graph(store = "Oxigraph").parse("https://open223.info/223p.ttl")
ontology = ontology_brick + ontology_s223

graph = None

def _ensure_graph_loaded():
    """Lazy load the graph from GRAPH_FILE environment variable"""
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

# @mcp.tool()
# def subjects(
#     predicate: Optional[str] = None,
#     object: Optional[str] = None,
#     limit: int = 100
# ) -> Dict[str, Any]:
#     """
#     Get subjects matching the given predicate and object pattern.
    
#     Args:
#         predicate: Optional predicate URI to filter by
#         object: Optional object value to filter by (URI or literal)
#         limit: Maximum number of results to return (default: 100)
    
#     Returns:
#         List of matching subjects with their types
#     """
#     g = _ensure_graph_loaded()
    
#     # Convert string inputs to RDF terms
#     pred = URIRef(predicate) if predicate else None
#     obj = URIRef(object) if object and object.startswith('http') else (Literal(object) if object else None)
    
#     subjects_list = []
#     count = 0
    
#     for subj in g.subjects(pred, obj):
#         if count >= limit:
#             break
#         subjects_list.append(_format_term(subj))
#         count += 1
    
#     pattern = f"predicate={predicate or 'ANY'}, object={object or 'ANY'}"
#     summary = f"Found {len(subjects_list)} subjects matching pattern: {pattern}"
    
#     return {
#         "summary": summary,
#         "pattern": {"predicate": predicate, "object": object},
#         "subjects": subjects_list,
#         "count": len(subjects_list),
#         "limited": count >= limit
#     }

# @mcp.tool()
# def predicates(
#     subject: Optional[str] = None,
#     object: Optional[str] = None,
#     limit: int = 100
# ) -> Dict[str, Any]:
#     """
#     Get predicates matching the given subject and object pattern.
    
#     Args:
#         subject: Optional subject URI to filter by
#         object: Optional object value to filter by (URI or literal)
#         limit: Maximum number of results to return (default: 100)
    
#     Returns:
#         List of matching predicates
#     """
#     g = _ensure_graph_loaded()
    
#     # Convert string inputs to RDF terms
#     subj = URIRef(subject) if subject else None
#     obj = URIRef(object) if object and object.startswith('http') else (Literal(object) if object else None)
    
#     predicates_list = []
#     count = 0
    
#     for pred in g.predicates(subj, obj):
#         if count >= limit:
#             break
#         predicates_list.append(_format_term(pred))
#         count += 1
    
#     pattern = f"subject={subject or 'ANY'}, object={object or 'ANY'}"
#     summary = f"Found {len(predicates_list)} predicates matching pattern: {pattern}"
    
#     return {
#         "summary": summary,
#         "pattern": {"subject": subject, "object": object},
#         "predicates": predicates_list,
#         "count": len(predicates_list),
#         "limited": count >= limit
#     }

# @mcp.tool()
# def objects(
#     subject: Optional[str] = None,
#     predicate: Optional[str] = None,
#     limit: int = 100
# ) -> Dict[str, Any]:
#     """
#     Get objects matching the given subject and predicate pattern.
    
#     Args:
#         subject: Optional subject URI to filter by
#         predicate: Optional predicate URI to filter by
#         limit: Maximum number of results to return (default: 100)
    
#     Returns:
#         List of matching objects with their types
#     """
#     g = _ensure_graph_loaded()
    
#     # Convert string inputs to RDF terms
#     subj = URIRef(subject) if subject else None
#     pred = URIRef(predicate) if predicate else None
    
#     objects_list = []
#     count = 0
    
#     for obj in g.objects(subj, pred):
#         if count >= limit:
#             break
#         objects_list.append(_format_term(obj))
#         count += 1
    
#     pattern = f"subject={subject or 'ANY'}, predicate={predicate or 'ANY'}"
#     summary = f"Found {len(objects_list)} objects matching pattern: {pattern}"
    
#     return {
#         "summary": summary,
#         "pattern": {"subject": subject, "predicate": predicate},
#         "objects": objects_list,
#         "count": len(objects_list),
#         "limited": count >= limit
#     }

# @mcp.tool()
# def subject_predicates(
#     object: Optional[str] = None,
#     limit: int = 100
# ) -> Dict[str, Any]:
#     """
#     Get subject-predicate pairs for triples matching the given object.
    
#     Args:
#         object: Optional object value to filter by (URI or literal)
#         limit: Maximum number of results to return (default: 100)
    
#     Returns:
#         List of (subject, predicate) pairs
#     """
#     g = _ensure_graph_loaded()
    
#     # Convert string input to RDF term
#     obj = URIRef(object) if object and object.startswith('http') else (Literal(object) if object else None)
    
#     pairs_list = []
#     count = 0
    
#     for subj, pred in g.subject_predicates(obj):
#         if count >= limit:
#             break
#         pairs_list.append({
#             "subject": _format_term(subj),
#             "predicate": _format_term(pred)
#         })
#         count += 1
    
#     pattern = f"object={object or 'ANY'}"
#     summary = f"Found {len(pairs_list)} subject-predicate pairs matching pattern: {pattern}"
    
#     return {
#         "summary": summary,
#         "pattern": {"object": object},
#         "pairs": pairs_list,
#         "count": len(pairs_list),
#         "limited": count >= limit
#     }

# @mcp.tool()
# def predicate_objects(
#     subject: Optional[str] = None,
#     limit: int = 100
# ) -> Dict[str, Any]:
#     """
#     Get predicate-object pairs for triples matching the given subject.
    
#     Args:
#         subject: Optional subject URI to filter by
#         limit: Maximum number of results to return (default: 100)
    
#     Returns:
#         List of (predicate, object) pairs
#     """
#     g = _ensure_graph_loaded()
    
#     # Convert string input to RDF term
#     subj = URIRef(subject) if subject else None
    
#     pairs_list = []
#     count = 0
    
#     for pred, obj in g.predicate_objects(subj):
#         if count >= limit:
#             break
#         pairs_list.append({
#             "predicate": _format_term(pred),
#             "object": _format_term(obj)
#         })
#         count += 1
    
#     pattern = f"subject={subject or 'ANY'}"
#     summary = f"Found {len(pairs_list)} predicate-object pairs matching pattern: {pattern}"
    
#     return {
#         "summary": summary,
#         "pattern": {"subject": subject},
#         "pairs": pairs_list,
#         "count": len(pairs_list),
#         "limited": count >= limit
#     }

# @mcp.tool()
# def subject_objects(
#     predicate: Optional[str] = None,
#     limit: int = 100
# ) -> Dict[str, Any]:
#     """
#     Get subject-object pairs for triples matching the given predicate.
    
#     Args:
#         predicate: Optional predicate URI to filter by
#         limit: Maximum number of results to return (default: 100)
    
#     Returns:
#         List of (subject, object) pairs
#     """
#     g = _ensure_graph_loaded()
    
#     # Convert string input to RDF term
#     pred = URIRef(predicate) if predicate else None
    
#     pairs_list = []
#     count = 0
    
#     for subj, obj in g.subject_objects(pred):
#         if count >= limit:
#             break
#         pairs_list.append({
#             "subject": _format_term(subj),
#             "object": _format_term(obj)
#         })
#         count += 1
    
#     pattern = f"predicate={predicate or 'ANY'}"
#     summary = f"Found {len(pairs_list)} subject-object pairs matching pattern: {pattern}"
    
#     return {
#         "summary": summary,
#         "pattern": {"predicate": predicate},
#         "pairs": pairs_list,
#         "count": len(pairs_list),
#         "limited": count >= limit
#     }


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
        num_hops: The number of hops to traverse from the entity (default: 1)
    
    Returns:
        Turtle-formatted RDF graph showing the entity and its immediate connections
    """

    g, parsed_graph = _ensure_graph_loaded() 

    # originally were args
    num_hops = 1 # may want to make an arg again, but it just retrieves so much info
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

# @mcp.tool()
def fuzzy_search_concept(
    concept: str,
    limit: int = 10):
    """
    Fuzzy search for concepts in the graph that match the given string.
    
    Args:
        concept: The string to search for.
        limit: The maximum number of results to return.
    
    Returns:
        A list of matching concepts.
    """
    pass

@mcp.tool()
def get_building_summary() -> Dict[str, Any]:
    """
    Get an overview of what types of entities and relationships exist in the building model.
    USE THIS FIRST when starting to work with an unfamiliar building model or dataset.
    
    When to use:
    - At the start of a conversation to understand what data is available
    - Before writing SPARQL queries to know valid class names
    - To check if certain types of entities exist in the model
    
    Returns:
        Counts of all entity types (classes)
    """
    g, parsed_graph = _ensure_graph_loaded()
    
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
    
    # summary = (
    #     f"Building model contains {len(g)} total triples, "
    #     f"{len(class_counts)} distinct classes, "
    #     f"and {len(relationship_counts)} distinct relationship types."
    # )
    
    return {
        # "summary": summary,
        # "total_triples": len(g),
        "class_counts": class_counts,
        "relationship_counts": relationship_counts,
        # "total_classes": len(class_counts),
        # "total_relationship_types": len(relationship_counts)
    }

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
            for entity in g.subjects(RDF.type, target_class):
                entity_info = {
                    "uri": str(entity),
                    "class": str(target_class)
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
def sparql_query(query: str, result_length: int = 10) -> Dict[str, Any]:
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
        result_length: Maximum results to return (default: 10)
    
    Returns:
        Query results with bindings for each variable
    """
    # Set alarm for timeout
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(60)  # 60 seconds timeout
    try:
        print(f"\n🔎 Running SPARQL query... (first 80 chars: {query[:80].replace(chr(10), ' ')}...)")
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

@mcp.tool()
def get_relationship_between_classes(
    start_class_uri: str,
    end_class_uri: str
) -> Dict[str, Any]:
    """
    Find the shortest path between instances of two classes in the graph.
    The path starts from an instance of start_class (via inverse rdf:type) and 
    ends at an instance of end_class (via rdf:type).

    When to use:
    - Finding the relationship between two types of entities
    - Questions about paths or indirect relationships between entities
    
    Args:
        start_class_uri: The URI of the starting class
        end_class_uri: The URI of the ending class
    
    Returns:
        Dictionary containing the shortest path information including:
        - path: List of nodes in the path (class -> instance -> ... -> instance -> class)
        - predicates: List of predicates connecting the nodes
        - length: Length of the path
        - found: Whether a path was found
    """
    g, parsed_graph = _ensure_graph_loaded()

    max_depth: int = 7
    
    # Convert strings to URIRefs
    start_class = URIRef(start_class_uri)
    end_class = URIRef(end_class_uri)
    
    # Get all instances of the start class
    start_instances = list(g.subjects(RDF.type, start_class))
    if not start_instances:
        return {
            "summary": f"No instances found for start class: {start_class_uri}",
            "found": False,
            "path": [],
            "predicates": [],
            "length": 0,
            "error": "no_start_instances"
        }
    
    # Get all instances of the end class
    end_instances = set(g.subjects(RDF.type, end_class))
    if not end_instances:
        return {
            "summary": f"No instances found for end class: {end_class_uri}",
            "found": False,
            "path": [],
            "predicates": [],
            "length": 0,
            "error": "no_end_instances"
        }
    
    # Check if any instance belongs to both classes
    common_instances = set(start_instances) & end_instances
    if common_instances:
        instance = list(common_instances)[0]
        return {
            "summary": f"Found common instance of both classes",
            "found": True,
            "path": [str(start_class), str(instance), str(end_class)],
            "predicates": ["^rdf:type", "rdf:type"],
            "length": 2
        }
    
    # Find shortest path from any start instance to any end instance
    result = _find_instance_to_instance_path(g, start_instances, end_instances, max_depth)
    
    if result["found"]:
        # Prepend start class and append end class to the path
        full_path = [str(start_class)] + result["path"] + [str(end_class)]
        full_predicates = ["^rdf:type"] + result["predicates"] + ["rdf:type"]
        
        path_str = " -> ".join(full_path)
        pred_str = " -> ".join(full_predicates)
        summary = (
            f"Found path of length {len(full_path) - 1} from {start_class_uri} to {end_class_uri}.\n"
            f"Path: {path_str}\n"
            f"Predicates: {pred_str}"
        )
        
        return {
            "summary": summary,
            "found": True,
            "path": full_path,
            "predicates": full_predicates,
            "length": len(full_path) - 1
        }
    else:
        summary = f"No path found between instances of {start_class_uri} and {end_class_uri} within {max_depth} hops"
        return {
            "summary": summary,
            "found": False,
            "path": [],
            "predicates": [],
            "length": 0
        }

def _find_instance_to_instance_path(
    g: Graph,
    start_instances: List[URIRef],
    end_instances: set,
    max_depth: int
) -> Dict[str, Any]:
    """
    Find shortest path from any instance in start_instances to any instance in end_instances.
    Uses BFS to explore from all start instances simultaneously.
    """
    # Queue stores: (current_node, path_nodes, path_predicates)
    queue = deque()
    visited = set()
    
    # Initialize queue with all start instances
    for instance in start_instances:
        queue.append((instance, [instance], []))
        visited.add(instance)
    
    while queue:
        current, path_nodes, path_predicates = queue.popleft()
        
        # Check if we've exceeded max depth
        if len(path_nodes) > max_depth:
            continue
        
        # Check if current node is an end instance
        if current in end_instances:
            return {
                "found": True,
                "path": [str(node) for node in path_nodes],
                "predicates": [str(pred) for pred in path_predicates],
                "length": len(path_nodes) - 1
            }
        
        # Explore outgoing edges (current as subject)
        for pred, obj in g.predicate_objects(current):
            # Skip predicates from excluded namespaces (RDF, RDFS, SHACL, QUDT)
            if _is_excluded_predicate(pred):
                continue
            if isinstance(obj, URIRef):
                if obj in end_instances:
                    # Found a target instance
                    return {
                        "found": True,
                        "path": [str(node) for node in path_nodes + [obj]],
                        "predicates": [str(pred) for pred in path_predicates + [pred]],
                        "length": len(path_nodes)
                    }
                
                if obj not in visited:
                    visited.add(obj)
                    queue.append((obj, path_nodes + [obj], path_predicates + [pred]))
        
        # Explore incoming edges (current as object)
        for subj, pred in g.subject_predicates(current):
            # Skip predicates from excluded namespaces (RDF, RDFS, SHACL, QUDT)
            if _is_excluded_predicate(pred):
                continue
            if isinstance(subj, URIRef):
                if subj in end_instances:
                    # Found a target instance
                    return {
                        "found": True,
                        "path": [str(node) for node in path_nodes + [subj]],
                        "predicates": [str(pred) for pred in path_predicates + [f"^{pred}"]],
                        "length": len(path_nodes)
                    }
                
                if subj not in visited:
                    visited.add(subj)
                    queue.append((subj, path_nodes + [subj], path_predicates + [f"^{pred}"]))
    
    return {
        "found": False,
        "path": [],
        "predicates": [],
        "length": 0
    }

def _bidirectional_bfs(
    g: Graph,
    start: URIRef,
    end: URIRef,
    max_depth: int
) -> Dict[str, Any]:
    """
    Perform bidirectional BFS to find shortest path more efficiently.
    """
    # Forward search from start
    forward_queue = deque([(start, [start], [])])
    forward_visited = {start: ([], [])}  # node -> (path_nodes, path_predicates)
    
    # Backward search from end
    backward_queue = deque([(end, [end], [])])
    backward_visited = {end: ([], [])}  # node -> (path_nodes, path_predicates)
    
    depth = 0
    
    while forward_queue or backward_queue:
        depth += 1
        if depth > max_depth:
            break
        
        # Expand forward frontier
        if forward_queue:
            for _ in range(len(forward_queue)):
                current, path_nodes, path_predicates = forward_queue.popleft()
                
                # Explore outgoing edges
                for pred, obj in g.predicate_objects(current):
                    # Skip predicates from excluded namespaces (RDF, RDFS, SHACL, QUDT)
                    if _is_excluded_predicate(pred):
                        continue
                    if isinstance(obj, URIRef):
                        # Check if we've met the backward search
                        if obj in backward_visited:
                            back_path, back_preds = backward_visited[obj]
                            full_path = path_nodes + [obj] + back_path[::-1]
                            full_preds = path_predicates + [pred] + back_preds[::-1]
                            return {
                                "found": True,
                                "path": [str(node) for node in full_path],
                                "predicates": [str(p) for p in full_preds],
                                "length": len(full_path) - 1,
                                "search_type": "bidirectional"
                            }
                        
                        if obj not in forward_visited:
                            forward_visited[obj] = (path_nodes + [obj], path_predicates + [pred])
                            forward_queue.append((obj, path_nodes + [obj], path_predicates + [pred]))
                
                # Explore incoming edges
                for subj, pred in g.subject_predicates(current):
                    # Skip predicates from excluded namespaces (RDF, RDFS, SHACL, QUDT)
                    if _is_excluded_predicate(pred):
                        continue
                    if isinstance(subj, URIRef):
                        # Check if we've met the backward search
                        if subj in backward_visited:
                            back_path, back_preds = backward_visited[subj]
                            full_path = path_nodes + [subj] + back_path[::-1]
                            full_preds = path_predicates + [pred] + back_preds[::-1]
                            return {
                                "found": True,
                                "path": [str(node) for node in full_path],
                                "predicates": [str(p) for p in full_preds],
                                "length": len(full_path) - 1,
                                "search_type": "bidirectional"
                            }
                        
                        if subj not in forward_visited:
                            forward_visited[subj] = (path_nodes + [subj], path_predicates + [pred])
                            forward_queue.append((subj, path_nodes + [subj], path_predicates + [pred]))
        
        # Expand backward frontier
        if backward_queue:
            for _ in range(len(backward_queue)):
                current, path_nodes, path_predicates = backward_queue.popleft()
                
                # Explore incoming edges (going backward)
                for subj, pred in g.subject_predicates(current):
                    # Skip predicates from excluded namespaces (RDF, RDFS, SHACL, QUDT)
                    if _is_excluded_predicate(pred):
                        continue
                    if isinstance(subj, URIRef):
                        # Check if we've met the forward search
                        if subj in forward_visited:
                            fwd_path, fwd_preds = forward_visited[subj]
                            full_path = fwd_path + [current] + path_nodes[1:][::-1]
                            full_preds = fwd_preds + [pred] + path_predicates[::-1]
                            return {
                                "found": True,
                                "path": [str(node) for node in full_path],
                                "predicates": [str(p) for p in full_preds],
                                "length": len(full_path) - 1,
                                "search_type": "bidirectional"
                            }
                        
                        if subj not in backward_visited:
                            backward_visited[subj] = (path_nodes + [subj], path_predicates + [pred])
                            backward_queue.append((subj, path_nodes + [subj], path_predicates + [pred]))
                
                # Explore outgoing edges (going backward)
                for pred, obj in g.predicate_objects(current):
                    # Skip predicates from excluded namespaces (RDF, RDFS, SHACL, QUDT)
                    if _is_excluded_predicate(pred):
                        continue
                    if isinstance(obj, URIRef):
                        # Check if we've met the forward search
                        if obj in forward_visited:
                            fwd_path, fwd_preds = forward_visited[obj]
                            full_path = fwd_path + [current] + path_nodes[1:][::-1]
                            full_preds = fwd_preds + [pred] + path_predicates[::-1]
                            return {
                                "found": True,
                                "path": [str(node) for node in full_path],
                                "predicates": [str(p) for p in full_preds],
                                "length": len(full_path) - 1,
                                "search_type": "bidirectional"
                            }
                        
                        if obj not in backward_visited:
                            backward_visited[obj] = (path_nodes + [obj], path_predicates + [pred])
                            backward_queue.append((obj, path_nodes + [obj], path_predicates + [pred]))
    
    return {
        "found": False,
        "path": [],
        "predicates": [],
        "length": 0
    }
