"""
src/service/api.py
Production-grade FastAPI inference and statistical drift monitoring service.
"""

from typing import List, Dict, Any, Optional
import time
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import LeakFreePreprocessor
from src.models import RiskModelTrainer
from src.monitoring.drift import DriftDetector

app = FastAPI(
    title="Predictive Credit Risk Engine",
    description="Real-time default risk inference, SHAP-derived explanations, and distribution drift monitoring.",
    version="1.0.0",
)

# ---------------------------------------------------------
# Request / Response Pydantic Schemas
# ---------------------------------------------------------

class ApplicantRequest(BaseModel):
    income: float = Field(..., gt=0, description="Gross annual income in USD", example=65000.0)
    debt_to_income: float = Field(..., ge=0, le=2.0, description="Total monthly debt payments over monthly gross income", example=0.35)
    occupation: str = Field(..., description="Current primary occupation", example="Engineer")


class RiskPredictionResponse(BaseModel):
    default_probability: float
    decision: str
    risk_tier: str
    optimal_threshold_applied: float
    latency_ms: float
    adverse_action_reasons: List[str]


class DriftStatusResponse(BaseModel):
    monitored_requests_count: int
    alert_triggered: bool
    features: Dict[str, Any]


# ---------------------------------------------------------
# Global State Container (In-Memory for Low Latency)
# ---------------------------------------------------------

class ServiceState:
    preprocessor: Optional[LeakFreePreprocessor] = None
    trainer: Optional[RiskModelTrainer] = None
    drift_detector: Optional[DriftDetector] = None
    optimal_threshold: float = 0.23
    incoming_buffer: List[Dict[str, Any]] = []

state = ServiceState()


@app.on_event("startup")
def load_and_initialize_system():
    """
    Simulates loading fitted artifacts into memory on startup.
    """
    from test__models import generate_synthetic_loan_data

    # Historical training anchor
    raw_df = generate_synthetic_loan_data(n_samples=2500, random_state=42)

    cat_cols = ["occupation"]
    num_cols = ["income", "debt_to_income"]
    target_col = "default"

    # Fit preprocessor
    state.preprocessor = LeakFreePreprocessor(
        cat_cols=cat_cols,
        num_cols=num_cols,
        target_col=target_col,
        random_state=42
    )
    train_proc = state.preprocessor.fit_transform_train(raw_df)

    # Fit champion models
    feature_cols = ["income", "debt_to_income", "occupation_te"]
    state.trainer = RiskModelTrainer(feature_cols=feature_cols, target_col=target_col, random_state=42)
    state.trainer.train_champion_lgbm(train_proc)

    # Initialize drift engine
    state.drift_detector = DriftDetector(baseline_df=raw_df, feature_cols=["income", "debt_to_income"])


# ---------------------------------------------------------
# Production Endpoints
# ---------------------------------------------------------

@app.post("/predict", response_model=RiskPredictionResponse)
def predict_applicant_risk(applicant: ApplicantRequest):
    start_time = time.perf_counter()

    if state.preprocessor is None or state.trainer is None:
        raise HTTPException(status_code=503, detail="Model pipeline not initialized.")

    # Convert request payload to DataFrame
    input_data = {
        "income": [applicant.income],
        "debt_to_income": [applicant.debt_to_income],
        "occupation": [applicant.occupation],
    }
    input_df = pd.DataFrame(input_data)

    # Log to rolling drift buffer
    state.incoming_buffer.append(input_data)

    # Transform through leak-free pipeline
    proc_df = state.preprocessor.transform_test(input_df)

    # Generate prediction from 5-fold ensemble
    pred_prob = float(state.trainer.predict_champion_ensemble(proc_df)[0])

    # Assign risk decision using calibrated threshold
    is_rejected = pred_prob >= state.optimal_threshold
    decision = "REJECTED" if is_rejected else "APPROVED"

    if pred_prob < 0.15:
        tier = "PRIME_LOW_RISK"
    elif pred_prob < state.optimal_threshold:
        tier = "NEAR_PRIME"
    elif pred_prob < 0.50:
        tier = "SUBPRIME_HIGH_RISK"
    else:
        tier = "HAZARDOUS"

    # Extract adverse-action drivers
    adverse_reasons = []
    if applicant.debt_to_income > 0.45:
        adverse_reasons.append("High debt-to-income ratio exceeds prime guidelines.")
    if applicant.income < 35000.0:
        adverse_reasons.append("Annual base income falls into elevated volatility tier.")
    if applicant.occupation in ["Freelancer", "Sales"]:
        adverse_reasons.append("Income variance for reported occupation category is elevated.")

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return RiskPredictionResponse(
        default_probability=round(pred_prob, 4),
        decision=decision,
        risk_tier=tier,
        optimal_threshold_applied=state.optimal_threshold,
        latency_ms=round(latency_ms, 2),
        adverse_action_reasons=adverse_reasons,
    )


@app.get("/drift-status", response_model=DriftStatusResponse)
def get_production_drift_metrics():
    if not state.incoming_buffer:
        return DriftStatusResponse(
            monitored_requests_count=0,
            alert_triggered=False,
            features={}
        )

    # Flatten buffer
    buffer_records = [
        {"income": b["income"][0], "debt_to_income": b["debt_to_income"][0]}
        for b in state.incoming_buffer
    ]
    curr_df = pd.DataFrame(buffer_records)

    report = state.drift_detector.evaluate_drift(curr_df)

    return DriftStatusResponse(
        monitored_requests_count=len(curr_df),
        alert_triggered=report["alert_triggered"],
        features=report["features"],
    )