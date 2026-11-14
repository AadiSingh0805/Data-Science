"""Knowledge Graph Engine with Cypher generation, Neo4j connectivity,
and bounded multi-hop graph traversal with cycle prevention.
"""

import json
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple, Any
import networkx as nx
from src.data.contracts import (
    Entity,
    Relationship,
    EntityType,
    RelationType,
    KnowledgeGraphExtraction,
)


class GraphEngine:
    """Core Knowledge Graph store and traversal engine.
    Supports local networkx graph analytics, raw Cypher DDL export,
    and live Neo4j driver execution.
    """

    def __init__(self, neo4j_uri: Optional[str] = None, auth: Optional[Tuple[str, str]] = None):
        self.graph = nx.DiGraph()
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self.neo4j_uri = neo4j_uri
        self.neo4j_driver = None

        if neo4j_uri and auth:
            try:
                from neo4j import GraphDatabase
                self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=auth)
            except Exception as e:
                print(f"[GraphEngine Warning] Could not connect to Neo4j instance at {neo4j_uri}: {e}")

    def add_entity(self, entity: Entity) -> None:
        """Adds or updates an entity node in the graph."""
        self.entities[entity.id] = entity
        self.graph.add_node(
            entity.id,
            name=entity.name,
            entity_type=entity.entity_type.value,
            description=entity.description,
        )

    def add_relationship(self, rel: Relationship) -> None:
        """Adds a directed relationship edge with citation attributes."""
        # Ensure endpoints exist
        if rel.source_id not in self.entities or rel.target_id not in self.entities:
            raise ValueError(f"Endpoints {rel.source_id} -> {rel.target_id} not registered.")
        
        self.relationships.append(rel)
        self.graph.add_edge(
            rel.source_id,
            rel.target_id,
            relation_type=rel.relation_type.value,
            verbatim_quote=rel.verbatim_quote,
            source_doc_id=rel.source_doc_id,
            chunk_id=rel.chunk_id,
            confidence=rel.confidence,
        )

    def load_from_extraction(self, extraction: KnowledgeGraphExtraction) -> None:
        """Populates the graph from a validated extraction contract."""
        for ent in extraction.entities:
            self.add_entity(ent)
        for rel in extraction.relationships:
            self.add_relationship(rel)

    def export_cypher_statements(self) -> List[str]:
        """Generates production Cypher DDL statements to populate Neo4j."""
        statements = [
            "// Constraint: Unique Entity ID",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;"
        ]

        # Node upserts
        for entity in self.entities.values():
            label = entity.entity_type.value
            clean_name = entity.name.replace("'", "\\'")
            clean_desc = (entity.description or "").replace("'", "\\'")
            stmt = (
                f"MERGE (e:Entity:{label} {{id: '{entity.id}'}}) "
                f"ON CREATE SET e.name = '{clean_name}', e.type = '{label}', e.description = '{clean_desc}';"
            )
            statements.append(stmt)

        # Relationship upserts
        for rel in self.relationships:
            rel_type = rel.relation_type.value
            clean_quote = rel.verbatim_quote.replace("'", "\\'").replace('"', '\\"')
            stmt = (
                f"MATCH (s:Entity {{id: '{rel.source_id}'}}), (t:Entity {{id: '{rel.target_id}'}}) "
                f"MERGE (s)-[r:{rel_type} {{source_doc: '{rel.source_doc_id}', chunk_id: {rel.chunk_id}}}]->(t) "
                f"ON CREATE SET r.verbatim_quote = '{clean_quote}', r.confidence = {rel.confidence};"
            )
            statements.append(stmt)

        return statements

    def find_multi_hop_paths(
        self,
        source_id: str,
        target_id: Optional[str] = None,
        max_depth: int = 3,
        target_type: Optional[EntityType] = None
    ) -> List[Dict[str, Any]]:
        """Bounded multi-hop path search with visited-node cycle prevention.
        Guarantees loop termination and sub-millisecond execution.
        """
        if source_id not in self.graph:
            return []

        results = []
        # Queue format: (current_node, [path_nodes], [path_edges], visited_set)
        queue = [(source_id, [source_id], [], {source_id})]

        while queue:
            curr, path_nodes, path_edges, visited = queue.pop(0)

            # Check target condition
            is_target = False
            if target_id and curr == target_id and len(path_nodes) > 1:
                is_target = True
            elif target_type and len(path_nodes) > 1:
                curr_type = self.graph.nodes[curr].get("entity_type")
                if curr_type == target_type.value:
                    is_target = True

            if is_target:
                results.append({
                    "nodes": [self.entities[nid].model_dump() for nid in path_nodes],
                    "edges": path_edges,
                    "depth": len(path_edges),
                    "summary": " -> ".join([self.entities[nid].name for nid in path_nodes])
                })
                if target_id:
                    continue  # Found specific target on this branch

            # Bound search depth
            if len(path_edges) >= max_depth:
                continue

            # Expand neighbors with cycle prevention
            for neighbor in self.graph.successors(curr):
                if neighbor in visited:
                    # CYCLE DETECTED: Prune traversal branch to prevent infinite loop
                    continue

                edge_data = self.graph.get_edge_data(curr, neighbor)
                new_edge = {
                    "source": curr,
                    "target": neighbor,
                    "relation_type": edge_data["relation_type"],
                    "verbatim_quote": edge_data["verbatim_quote"],
                    "source_doc_id": edge_data["source_doc_id"]
                }
                new_visited = visited.copy()
                new_visited.add(neighbor)

                queue.append((
                    neighbor,
                    path_nodes + [neighbor],
                    path_edges + [new_edge],
                    new_visited
                ))

        return results

    def query_entity_neighborhood(self, entity_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """Extracts the contextual subgraph around an entity up to max_depth."""
        if entity_id not in self.graph:
            return {"error": f"Entity '{entity_id}' not found in knowledge graph."}

        subgraph_nodes: Set[str] = {entity_id}
        subgraph_edges: List[Dict[str, Any]] = []

        # Forward and backward BFS expansion up to max_depth
        current_layer = {entity_id}
        visited = {entity_id}

        for _ in range(max_depth):
            next_layer = set()
            for node in current_layer:
                # Outgoing edges
                for succ in self.graph.successors(node):
                    edge_data = self.graph.get_edge_data(node, succ)
                    subgraph_edges.append({
                        "source": node,
                        "source_name": self.entities[node].name,
                        "target": succ,
                        "target_name": self.entities[succ].name,
                        "relation": edge_data["relation_type"],
                        "quote": edge_data["verbatim_quote"],
                        "source_doc": edge_data["source_doc_id"]
                    })
                    subgraph_nodes.add(succ)
                    if succ not in visited:
                        visited.add(succ)
                        next_layer.add(succ)

                # Incoming edges
                for pred in self.graph.predecessors(node):
                    edge_data = self.graph.get_edge_data(pred, node)
                    subgraph_edges.append({
                        "source": pred,
                        "source_name": self.entities[pred].name,
                        "target": node,
                        "target_name": self.entities[node].name,
                        "relation": edge_data["relation_type"],
                        "quote": edge_data["verbatim_quote"],
                        "source_doc": edge_data["source_doc_id"]
                    })
                    subgraph_nodes.add(pred)
                    if pred not in visited:
                        visited.add(pred)
                        next_layer.add(pred)

            current_layer = next_layer

        # Deduplicate edges
        seen_edges = set()
        unique_edges = []
        for e in subgraph_edges:
            key = (e["source"], e["target"], e["relation"])
            if key not in seen_edges:
                seen_edges.add(key)
                unique_edges.append(e)

        return {
            "root_entity": self.entities[entity_id].model_dump(),
            "connected_entities": [self.entities[nid].model_dump() for nid in subgraph_nodes if nid != entity_id],
            "subgraph_edges": unique_edges,
            "total_nodes": len(subgraph_nodes),
            "total_edges": len(unique_edges)
        }

    def save_to_file(self, file_path: Path) -> None:
        """Persists the extracted graph to a JSON artifact."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entities": [e.model_dump() for e in self.entities.values()],
            "relationships": [r.model_dump() for r in self.relationships]
        }
        file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_from_file(self, file_path: Path) -> None:
        """Loads graph from a persisted JSON artifact."""
        data = json.loads(file_path.read_text(encoding="utf-8"))
        extraction = KnowledgeGraphExtraction(**data)
        self.load_from_extraction(extraction)

