"""Hybrid Retrieval Engine implementing Reciprocal Rank Fusion (RRF).
Merges dense vector search (Qdrant) with multi-hop knowledge graph traversals.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from src.features.triplets import EntityResolver
from src.models.vector import VectorEngine
from src.models.graph import GraphEngine


class HybridRetriever:
    """Fuses dense vector semantic search with deterministic knowledge graph paths
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        vector_engine: VectorEngine,
        graph_engine: GraphEngine,
        rrf_k: int = 60
    ):
        self.vector_engine = vector_engine
        self.graph_engine = graph_engine
        self.rrf_k = rrf_k
        self.resolver = EntityResolver()

    def identify_query_entities(self, query: str) -> List[str]:
        """Resolves entity mentions in the user query to canonical entity IDs."""
        q_lower = query.lower()
        matched_entity_ids: Set[str] = set()

        for alias_key, (canon_name, entity_type) in self.resolver.ALIAS_MAP.items():
            if len(alias_key) >= 3 and alias_key in q_lower:
                eid, _, _ = self.resolver.resolve(alias_key, entity_type)
                if eid in self.graph_engine.entities:
                    matched_entity_ids.add(eid)

        return list(matched_entity_ids)

    def retrieve_graph_evidence(
        self,
        query: str,
        max_hops: int = 4
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Performs multi-hop graph exploration and returns ranked chunks and path traces."""
        entity_ids = self.identify_query_entities(query)
        if not entity_ids:
            return [], []

        paths_found = []
        supporting_chunks_dict: Dict[str, Dict[str, Any]] = {}

        # If 2 or more entities detected, search for connecting paths
        if len(entity_ids) >= 2:
            for i in range(len(entity_ids)):
                for j in range(len(entity_ids)):
                    if i != j:
                        p = self.graph_engine.find_multi_hop_paths(
                            source_id=entity_ids[i],
                            target_id=entity_ids[j],
                            max_depth=max_hops
                        )
                        paths_found.extend(p)

        # Also retrieve 1-to-2 hop neighborhoods for each entity
        for eid in entity_ids:
            neighborhood = self.graph_engine.query_entity_neighborhood(eid, max_depth=2)
            for edge in neighborhood.get("subgraph_edges", []):
                chunk_key = f"{edge['source_doc']}:{edge.get('chunk_id', 0)}"
                if chunk_key not in supporting_chunks_dict:
                    supporting_chunks_dict[chunk_key] = {
                        "doc_id": edge["source_doc"],
                        "chunk_id": edge.get("chunk_id", 0),
                        "content": edge["quote"],
                        "relevance_weight": 1.0,
                        "verbatim_quote": edge["quote"],
                        "source": "graph_edge"
                    }

        # Boost chunks that appear directly on multi-hop paths
        for path in paths_found:
            for edge in path.get("edges", []):
                chunk_key = f"{edge['source_doc_id']}:0"
                if chunk_key in supporting_chunks_dict:
                    supporting_chunks_dict[chunk_key]["relevance_weight"] += 2.0
                else:
                    supporting_chunks_dict[chunk_key] = {
                        "doc_id": edge["source_doc_id"],
                        "chunk_id": 0,
                        "content": edge["verbatim_quote"],
                        "relevance_weight": 2.0,
                        "verbatim_quote": edge["verbatim_quote"],
                        "source": "multi_hop_path"
                    }

        # Rank graph chunks by accumulated relevance weight
        sorted_graph_chunks = sorted(
            supporting_chunks_dict.values(),
            key=lambda x: x["relevance_weight"],
            reverse=True
        )

        ranked_graph_results = []
        for rank, item in enumerate(sorted_graph_chunks, start=1):
            ranked_graph_results.append({
                "rank": rank,
                "doc_id": item["doc_id"],
                "chunk_id": item["chunk_id"],
                "chunk_key": f"{item['doc_id']}:{item['chunk_id']}",
                "content": item["content"],
                "verbatim_quote": item.get("verbatim_quote", ""),
                "score": float(item["relevance_weight"])
            })

        return ranked_graph_results, paths_found

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        vector_weight: float = 1.0,
        graph_weight: float = 1.0
    ) -> Dict[str, Any]:
        """Fuses Vector and Graph retrieval via Reciprocal Rank Fusion (RRF)."""
        # 1. Dense Vector Retrieval
        vector_results = self.vector_engine.search(query, top_k=top_k * 2)

        # 2. Knowledge Graph Multi-Hop Retrieval
        graph_results, paths_found = self.retrieve_graph_evidence(query)

        # 3. Reciprocal Rank Fusion Computation
        # RRF(d) = sum_m ( weight_m / (k + rank_m(d)) )
        fused_scores: Dict[str, float] = {}
        chunk_data: Dict[str, Dict[str, Any]] = {}

        for hit in vector_results:
            key = hit["chunk_key"]
            rrf_val = vector_weight / (self.rrf_k + hit["rank"])
            fused_scores[key] = fused_scores.get(key, 0.0) + rrf_val
            chunk_data[key] = {
                "chunk_key": key,
                "doc_id": hit["doc_id"],
                "chunk_id": hit["chunk_id"],
                "content": hit["content"],
                "vector_rank": hit["rank"],
                "vector_score": hit["score"],
                "graph_rank": None,
            }

        for hit in graph_results:
            key = hit["chunk_key"]
            rrf_val = graph_weight / (self.rrf_k + hit["rank"])
            fused_scores[key] = fused_scores.get(key, 0.0) + rrf_val
            if key not in chunk_data:
                chunk_data[key] = {
                    "chunk_key": key,
                    "doc_id": hit["doc_id"],
                    "chunk_id": hit["chunk_id"],
                    "content": hit["content"],
                    "vector_rank": None,
                    "vector_score": None,
                    "graph_rank": hit["rank"],
                }
            else:
                chunk_data[key]["graph_rank"] = hit["rank"]

        # Sort by final RRF score descending
        sorted_keys = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)
        top_fused = []
        for rank, key in enumerate(sorted_keys[:top_k], start=1):
            info = chunk_data[key]
            info["final_rank"] = rank
            info["rrf_score"] = fused_scores[key]
            top_fused.append(info)

        return {
            "query": query,
            "fused_results": top_fused,
            "paths_found": paths_found,
            "vector_candidates_count": len(vector_results),
            "graph_candidates_count": len(graph_results),
            "entities_detected": self.identify_query_entities(query)
        }
