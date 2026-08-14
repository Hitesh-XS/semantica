"""Google ADK FunctionTools backed by Semantica's ContextGraph."""

from typing import Any, Dict, List, Optional

from semantica.context import ContextGraph
from semantica.semantic_extract import NERExtractor


def _serialize(value: Any) -> Any:
    """Convert Semantica objects into JSON-friendly values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize(v) for v in value]
    if hasattr(value, "to_dict"):
        return _serialize(value.to_dict())
    if hasattr(value, "__dict__"):
        return _serialize(vars(value))
    return str(value)


def extract_entities(text: str) -> dict:
    """Extract named entities from text using Semantica's NER pipeline."""
    extractor = NERExtractor(method="ml")
    entities = extractor.extract_entities(text)
    return {"entities": _serialize(entities)}


def _build_tools(graph: ContextGraph) -> List[Any]:
    try:
        from google.adk.tools import FunctionTool
    except ImportError as exc:
        raise ImportError(
            "Google ADK is required for semantica_kg_tools(). "
            "Install it with: pip install semantica[google-adk]"
        ) from exc

    def add_entity(name: str, entity_type: str = "entity", properties: Optional[dict] = None) -> dict:
        """Add an entity node to the shared Semantica knowledge graph."""
        graph.add_node(name, entity_type, content=name, **(properties or {}))
        return {"id": name, "type": entity_type, "properties": properties or {}}

    def add_relationship(source: str, target: str, relationship: str = "related_to") -> dict:
        """Add a relationship between two entities in the shared Semantica knowledge graph."""
        graph.add_edge(source, target, edge_type=relationship)
        return {"source": source, "target": target, "type": relationship}

    def query_graph(query: str, limit: int = 20) -> dict:
        """Query the shared Semantica knowledge graph by keyword."""
        return {"results": _serialize(graph.query(query, limit=limit))}

    def extract_and_add(text: str) -> dict:
        """Extract named entities from text and add them to the shared knowledge graph."""
        result = extract_entities(text)
        added = []
        for entity in result["entities"]:
            name = entity.get("text") or entity.get("name")
            if not name:
                continue
            entity_type = entity.get("label") or entity.get("type") or "entity"
            graph.add_node(name, entity_type, content=name, **entity.get("metadata", {}))
            added.append(name)
        return {"entities": result["entities"], "added": added}

    return [
        FunctionTool(func=extract_entities),
        FunctionTool(func=extract_and_add),
        FunctionTool(func=add_entity),
        FunctionTool(func=add_relationship),
        FunctionTool(func=query_graph),
    ]


def semantica_kg_tools(graph: Optional[ContextGraph] = None) -> List[Any]:
    """Return ADK FunctionTools bound to a shared ContextGraph instance."""
    return _build_tools(graph or ContextGraph())
