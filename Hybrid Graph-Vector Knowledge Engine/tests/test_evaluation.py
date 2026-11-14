"""Unit tests for RAGTriadEvaluator and comparative benchmarking."""

import pytest
from pathlib import Path
from src.features.triplets import TripletExtractionPipeline
from src.models.graph import GraphEngine
from src.models.vector import VectorEngine
from src.models.hybrid import HybridRetriever
from src.models.evaluation import RAGTriadEvaluator, GOLDEN_BENCHMARK_DATASET


@pytest.fixture(scope="module")
def setup_evaluator():
    raw_dir = Path("data/raw")
    pipe = TripletExtractionPipeline()
    chunks, kg = pipe.process_directory(raw_dir)

    graph_engine = GraphEngine()
    graph_engine.load_from_extraction(kg)

    vector_engine = VectorEngine(storage_path=None)
    vector_engine.index_chunks(chunks)

    retriever = HybridRetriever(vector_engine, graph_engine)
    return RAGTriadEvaluator(vector_engine, retriever)


def test_faithfulness_metric_calculation(setup_evaluator):
    evaluator = setup_evaluator
    retrieved_texts = [
        "ASML Holding N.V. supplies EUV photolithography scanners to TSMC in Taiwan.",
        "TSMC is the sole foundry manufacturing chips for NVIDIA Corporation."
    ]
    # 2 grounded claims, 1 hallucinated claim
    claims = [
        "ASML supplies EUV photolithography scanners to TSMC.",
        "TSMC manufactures chips for NVIDIA.",
        "NVIDIA builds its own fabs in Germany."  # Hallucination!
    ]

    score = evaluator.evaluate_faithfulness(claims, retrieved_texts)
    assert 0.60 <= score <= 0.70  # ~2/3 = 0.667


def test_answer_relevance_metric(setup_evaluator):
    evaluator = setup_evaluator
    query = "What does ASML supply to TSMC?"
    relevant_answer = "ASML is the sole supplier of Extreme Ultraviolet (EUV) photolithography scanners to TSMC."
    irrelevant_answer = "The quick brown fox jumps over the lazy dog in the middle of summer."

    score_rel = evaluator.evaluate_answer_relevance(query, relevant_answer)
    score_irrel = evaluator.evaluate_answer_relevance(query, irrelevant_answer)

    assert score_rel > score_irrel
    assert score_rel > 0.50


def test_comparative_benchmark_executes_successfully(setup_evaluator):
    evaluator = setup_evaluator
    summary = evaluator.run_comparative_benchmark(top_k=3)

    assert "naive_vector" in summary
    assert "hybrid_graph" in summary
    assert summary["hybrid_graph"]["mean_latency_ms"] < 100.0  # Sub-100ms SLA

