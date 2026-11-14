"""Unit tests for VectorEngine and HybridRetriever with Reciprocal Rank Fusion (RRF)."""

import pytest
from pathlib import Path
from src.features.triplets import TripletExtractionPipeline
from src.models.graph import GraphEngine
from src.models.vector import VectorEngine
from src.models.hybrid import HybridRetriever


@pytest.fixture(scope="module")
def setup_hybrid_system():
    raw_dir = Path("data/raw")
    pipe = TripletExtractionPipeline()
    chunks, kg = pipe.process_directory(raw_dir)

    graph_engine = GraphEngine()
    graph_engine.load_from_extraction(kg)

    vector_engine = VectorEngine(storage_path=None)  # in-memory
    vector_engine.index_chunks(chunks)

    retriever = HybridRetriever(vector_engine, graph_engine, rrf_k=60)
    return retriever


def test_rrf_scoring_formula():
    """Validates the mathematical computation of RRF: sum(1 / (k + rank))."""
    k = 60
    # Document ranked 1 in vector and 10 in graph
    score_a = (1.0 / (k + 1)) + (1.0 / (k + 10))
    # Document ranked 3 in vector and 3 in graph
    score_b = (1.0 / (k + 3)) + (1.0 / (k + 3))

    assert score_b > score_a
    assert pytest.approx(score_a, rel=1e-3) == 0.030679
    assert pytest.approx(score_b, rel=1e-3) == 0.031746


def test_hybrid_search_resolves_multi_hop_query(setup_hybrid_system):
    retriever = setup_hybrid_system
    query = "How could bottlenecks at ASML affect server shipments to Microsoft Azure?"
    result = retriever.hybrid_search(query, top_k=3)

    assert len(result["fused_results"]) > 0
    assert "company:asml_holding_nv" in result["entities_detected"] or "company:microsoft_azure" in result["entities_detected"]
    assert len(result["paths_found"]) >= 1

    # Verify path connectivity
    path_summary = result["paths_found"][0]["summary"]
    assert "ASML" in path_summary
    assert "TSMC" in path_summary
    assert "NVIDIA" in path_summary


def test_vector_search_returns_ranked_results(setup_hybrid_system):
    retriever = setup_hybrid_system
    vector_results = retriever.vector_engine.search("extreme ultraviolet photolithography equipment", top_k=3)
    assert len(vector_results) == 3
    assert vector_results[0]["rank"] == 1
    assert vector_results[0]["score"] > 0.0

