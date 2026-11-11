import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_api_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["optimal_threshold"] == 0.1337


def test_predict_approves_safe_borrower():
    # Strong applicant: High income, low debt, excellent bureau scores
    payload = {
        "amt_income_total": 120000.0,
        "amt_credit": 150000.0,
        "amt_annuity": 7500.0,
        "days_birth": -15000,
        "days_employed": -3000,
        "ext_source_1": 0.85,
        "ext_source_2": 0.90,
        "ext_source_3": 0.88,
        "region_rating_client": 1,
    }
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "APPROVED"
    assert data["risk_tier"] == "LOW_RISK"
    assert data["default_probability"] < 0.1337
    assert len(data["adverse_action_reasons"]) == 0


def test_predict_rejects_high_risk_and_provides_adverse_action():
    # Risky applicant: Low income, crushing debt annuity, rock-bottom bureau scores
    payload = {
        "amt_income_total": 30000.0,
        "amt_credit": 200000.0,
        "amt_annuity": 18000.0,  # 60% DTI
        "days_birth": -8000,
        "days_employed": -200,
        "ext_source_1": 0.10,
        "ext_source_2": 0.12,
        "ext_source_3": 0.15,
        "region_rating_client": 3,
    }
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "REJECTED"
    assert data["default_probability"] >= 0.1337
    # Mandated compliance: top adverse action reason codes must be present
    assert len(data["adverse_action_reasons"]) > 0
    assert "reason_code" in data["adverse_action_reasons"][0]
    assert data["adverse_action_reasons"][0]["shap_value"] > 0


def test_predict_pydantic_schema_validation_error():
    # Invalid payload: negative income and invalid ext_source_2 > 1.0
    payload = {
        "amt_income_total": -5000.0,  # Fails gt=0
        "amt_credit": 100000.0,
        "amt_annuity": 5000.0,
        "days_birth": -12000,
        "days_employed": -1000,
        "ext_source_2": 1.50,  # Fails le=1.0
    }
    response = client.post("/v1/predict", json=payload)
    # Pydantic must reject with 422 before touching the model
    assert response.status_code == 422
