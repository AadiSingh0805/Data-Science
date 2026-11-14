"""Integration tests for FastAPI endpoints using TestClient."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["entities_count"] > 0
    assert data["relationships_count"] > 0
    assert data["chunks_count"] > 0


def test_query_endpoint(client):
    payload = {
        "query": "How do EUV photolithography lead times at ASML affect TSMC and NVIDIA?",
        "top_k": 3
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["fused_results"]) > 0
    assert data["latency_ms"] < 250.0  # Fast response SLA
    assert "answer" in data
    assert len(data["entities_detected"]) > 0


def test_graph_neighborhood_endpoint(client):
    response = client.get("/api/v1/graph/neighborhood/company:tsmc?max_depth=1")
    assert response.status_code == 200
    data = response.json()
    assert "subgraph_edges" in data
    assert data["total_nodes"] > 1
    assert data["root_entity"]["id"] == "company:tsmc"

