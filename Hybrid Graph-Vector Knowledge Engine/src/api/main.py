"""FastAPI async REST serving layer for Hybrid GraphRAG Engine.
Provides sub-50ms query endpoints, graph traversals, and citation payloads.
"""

import time
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.features.triplets import TripletExtractionPipeline
from src.models.graph import GraphEngine
from src.models.vector import VectorEngine
from src.models.hybrid import HybridRetriever
from src.models.evaluation import RAGTriadEvaluator
from src.api.schemas import (
    QueryRequest,
    QueryResponse,
    HealthResponse,
    FusedChunkItem,
    GraphPathItem,
)

# Global singleton engine state
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize pipeline and engines
    raw_dir = Path("data/raw")
    pipe = TripletExtractionPipeline()
    chunks, kg = pipe.process_directory(raw_dir)

    graph_engine = GraphEngine()
    graph_engine.load_from_extraction(kg)
    graph_engine.save_to_file(Path("data/processed/knowledge_graph.json"))

    vector_engine = VectorEngine(storage_path="data/processed/qdrant_db")
    vector_engine.index_chunks(chunks)

    hybrid_retriever = HybridRetriever(vector_engine, graph_engine)
    evaluator = RAGTriadEvaluator(vector_engine, hybrid_retriever)

    state["chunks"] = chunks
    state["kg"] = kg
    state["graph_engine"] = graph_engine
    state["vector_engine"] = vector_engine
    state["retriever"] = hybrid_retriever
    state["evaluator"] = evaluator

    print("[API Lifespan] Hybrid GraphRAG engines initialized successfully.")
    yield
    state.clear()


app = FastAPI(
    title="Hybrid Graph-Vector Knowledge Engine API",
    version="1.0.0",
    description="Production GraphRAG API merging Qdrant vector search and Neo4j/Cypher graph traversals via RRF.",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    if "retriever" not in state:
        raise HTTPException(status_code=503, detail="Engines not ready")
    return HealthResponse(
        status="healthy",
        entities_count=len(state["graph_engine"].entities),
        relationships_count=len(state["graph_engine"].relationships),
        chunks_count=len(state["chunks"])
    )


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_knowledge_engine(req: QueryRequest):
    if "retriever" not in state:
        raise HTTPException(status_code=503, detail="Retriever engine not initialized")

    t0 = time.perf_counter()
    retriever: HybridRetriever = state["retriever"]

    res = retriever.hybrid_search(
        query=req.query,
        top_k=req.top_k,
        vector_weight=req.vector_weight,
        graph_weight=req.graph_weight
    )

    t_elapsed_ms = (time.perf_counter() - t0) * 1000

    # Build response chunks
    fused_items = [
        FusedChunkItem(
            final_rank=c["final_rank"],
            rrf_score=c["rrf_score"],
            doc_id=c["doc_id"],
            chunk_id=c["chunk_id"],
            content=c["content"],
            vector_rank=c.get("vector_rank"),
            graph_rank=c.get("graph_rank")
        )
        for c in res["fused_results"]
    ]

    path_items = [
        GraphPathItem(
            summary=p["summary"],
            depth=p["depth"],
            nodes=p["nodes"],
            edges=p["edges"]
        )
        for p in res["paths_found"]
    ]

    # Synthesize citation-grounded response
    answer_parts = []
    if path_items:
        for p in path_items:
            answer_parts.append(f"Supply chain path: {p.summary}.")
    for c in fused_items[:2]:
        first_sentence = c.content.split(".")[0].strip()
        if len(first_sentence) > 15:
            answer_parts.append(f"Factual citation ({c.doc_id}): {first_sentence}.")

    answer_str = " ".join(answer_parts) if answer_parts else "No multi-hop path found for query."

    return QueryResponse(
        query=req.query,
        answer=answer_str,
        entities_detected=res["entities_detected"],
        fused_results=fused_items,
        paths_found=path_items,
        latency_ms=round(t_elapsed_ms, 2)
    )


@app.get("/api/v1/graph/neighborhood/{entity_id}")
async def get_entity_neighborhood(entity_id: str, max_depth: int = 2):
    if "graph_engine" not in state:
        raise HTTPException(status_code=503, detail="Graph engine not initialized")
    graph: GraphEngine = state["graph_engine"]
    return graph.query_entity_neighborhood(entity_id, max_depth=max_depth)


@app.get("/api/v1/benchmark")
async def run_rag_triad_benchmark():
    if "evaluator" not in state:
        raise HTTPException(status_code=503, detail="Evaluator not initialized")
    evaluator: RAGTriadEvaluator = state["evaluator"]
    return evaluator.run_comparative_benchmark(top_k=3)

