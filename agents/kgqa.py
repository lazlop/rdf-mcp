from mcp.server.fastmcp import FastMCP
from rdflib import Graph, URIRef, Literal, Namespace, BRICK, RDFS, RDF, BNode
from rdflib.term import Variable
from typing import List, Optional, Dict, Any
from pathlib import Path
import os 
import sys
import signal

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.namespaces import bind_prefixes, get_prefixes, S223, BRICK

mcp = FastMCP("GraphDemo", dependencies=["rdflib", "oxrdflib"])

# ontology can be brick or 223 
ontology_brick = Graph(store = "Oxigraph").parse("https://brickschema.org/schema/1.4/Brick.ttl")
ontology_s223 = Graph(store = "Oxigraph").parse("https://open223.info/223p.ttl")
ontology = ontology_brick + ontology_s223

graph = None

def _ensure_graph_loaded():
    """Lazy load the graph from GRAPH_FILE environment variable"""
    global graph
    if graph is None:
        graph_file = os.getenv("GRAPH_FILE")
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
        print(f"✅ Loaded graph from {graph_file} ({len(graph)} triples)")
    return graph

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
    - All Brick or S223 classes present and their counts
    - All relationship types (predicates) and their counts
    
    Returns a structured overview of what's in the building model.
    """
    g = _ensure_graph_loaded()
    
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
    
    summary = (
        f"Building model contains {len(g)} total triples, "
        f"{len(class_counts)} distinct classes, "
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
def find_entities_by_type(klass: str | URIRef, include_subclasses: bool = True) -> Dict[str, Any]:
    """
    Find all entities of a given class type.
    
    Args:
        klass: The Brick or S223 class name
        include_subclasses: If True, also returns entities of subclasses (default: True)
    
    Returns:
        List of all matching entities with their URIs and labels (if available)
    """
    g = _ensure_graph_loaded()
    class_uri = URIRef(klass)
    
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
        "summary": summary,
        "class_searched": f"{klass}",
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
def _timeout_handler(signum, frame):
    raise TimeoutError("SPARQL query timed out after 60 seconds")

@mcp.tool()
def sparql_query(query: str, result_length: int = 10) -> Dict[str, Any]:
    """
    Executes a SPARQL query with a timeout (default 60 seconds). If the query exceeds the timeout,
    returns an error message asking for a more specific query.
    """
    # Set alarm for timeout
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(60)  # 60 seconds timeout
    try:
        print(f"\n🔎 Running SPARQL query... (first 80 chars: {query[:80].replace(chr(10), ' ')}...)")
        g = _ensure_graph_loaded()
        prefixes = get_prefixes(g)
        full_query = prefixes + "\n" + query
        qres = g.query(full_query)
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
