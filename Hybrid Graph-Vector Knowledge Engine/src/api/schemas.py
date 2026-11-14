"""Pydantic API request and response schemas for GraphRAG serving layer."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="User query or risk assessment question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of fused chunks to return")
    vector_weight: float = Field(default=1.0, ge=0.0, le=5.0)
    graph_weight: float = Field(default=1.0, ge=0.0, le=5.0)


class FusedChunkItem(BaseModel):
    final_rank: int
    rrf_score: float
    doc_id: str
    chunk_id: int
    content: str
    vector_rank: Optional[int] = None
    graph_rank: Optional[int] = None


class GraphPathItem(BaseModel):
    summary: str
    depth: int
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


class QueryResponse(BaseModel):
    query: str
    answer: str
    entities_detected: List[str]
    fused_results: List[FusedChunkItem]
    paths_found: List[GraphPathItem]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    entities_count: int
    relationships_count: int
    chunks_count: int

