import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from fastapi import FastAPI, HTTPException, status
from typing import Dict, Any

from src.api.schemas import (
    LoanApplicationRequest,
    LoanApplicationResponse,
    AdverseActionReason,
    DriftCheckRequest,
    DriftCheckResponse,
)
from src.models.explainability import RiskExplainer
from src.monitoring.drift import DriftDetector

app = FastAPI(
    title="Enterprise Credit Risk & Production Drift Engine",
    description="Mission-critical underwriting API with sub-50ms latency, TreeSHAP Adverse Action Notices, and live PSI drift monitoring.",
    version="1.0.0",
)

# Core Feature Schema used by the underlying model
MODEL_FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_INCOME_RATIO",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_EMPLOYED_ANOM",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "REGION_RATING_CLIENT",
]

# Profit-maximizing threshold derived in Session 1.3
OPTIMAL_THRESHOLD = 0.1337


class ModelService:
    """Singleton service holding the LightGBM model, TreeSHAP explainer, and Drift detector."""
    def __init__(self):
        self.model = None
        self.explainer = None
        self.drift_detector = None
        self.initialize_engine()

    def initialize_engine(self):
        # Build baseline synthetic reference dataset to initialize model and drift baseline
        np.random.seed(42)
        n = 1000
        
        income = np.random.lognormal(mean=11.5, sigma=0.5, size=n)
        credit = income * np.random.uniform(2.0, 5.0, size=n)
        annuity = credit * np.random.uniform(0.04, 0.08, size=n)
        ann_inc = annuity / income
        cred_inc = credit / income
        age_days = -np.random.uniform(7500, 24000, size=n)
        emp_days = -np.random.uniform(100, 8000, size=n)
        anom_emp = np.zeros(n)
        # Apply retiree anomaly to ~15%
        retirees = np.random.rand(n) < 0.15
        emp_days[retirees] = np.nan
        anom_emp[retirees] = 1.0

        ext1 = np.where(np.random.rand(n) < 0.50, np.nan, np.random.beta(3, 3, size=n))
        ext2 = np.random.beta(4, 4, size=n)
        ext3 = np.where(np.random.rand(n) < 0.20, np.nan, np.random.beta(3, 3, size=n))
        region = np.random.choice([1, 2, 3], size=n, p=[0.2, 0.6, 0.2])

        # Underlying latent default probability with realistic risk sensitivity
        s2 = np.nan_to_num(ext2, nan=0.5)
        s3 = np.nan_to_num(ext3, nan=0.5)
        z = -2.3 - 3.5 * (s2 - 0.5) - 3.0 * (s3 - 0.5) + 3.5 * (ann_inc - 0.22)
        prob = 1.0 / (1.0 + np.exp(-z))
        y = (np.random.rand(n) < prob).astype(int)

        ref_df = pd.DataFrame({
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": annuity,
            "ANNUITY_INCOME_RATIO": ann_inc,
            "CREDIT_INCOME_RATIO": cred_inc,
            "DAYS_BIRTH": age_days,
            "DAYS_EMPLOYED": emp_days,
            "DAYS_EMPLOYED_ANOM": anom_emp,
            "EXT_SOURCE_1": ext1,
            "EXT_SOURCE_2": ext2,
            "EXT_SOURCE_3": ext3,
            "REGION_RATING_CLIENT": region,
        })

        train_data = lgb.Dataset(ref_df, label=y, feature_name=MODEL_FEATURES)
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 15,
            "learning_rate": 0.15,
            "min_data_in_leaf": 10,
            "verbose": -1,
        }
        self.model = lgb.train(params, train_data, num_boost_round=60)
        self.explainer = RiskExplainer(model=self.model, feature_names=MODEL_FEATURES)
        self.drift_detector = DriftDetector(reference_data=ref_df, feature_names=MODEL_FEATURES)

    def extract_features(self, req: LoanApplicationRequest) -> np.ndarray:
        ann_inc = req.amt_annuity / req.amt_income_total
        cred_inc = req.amt_credit / req.amt_income_total
        is_anom = 1.0 if req.days_employed == 365243 else 0.0
        emp_clean = np.nan if req.days_employed == 365243 else float(req.days_employed)

        row = [
            req.amt_income_total,
            req.amt_credit,
            req.amt_annuity,
            ann_inc,
            cred_inc,
            float(req.days_birth),
            emp_clean,
            is_anom,
            req.ext_source_1 if req.ext_source_1 is not None else np.nan,
            req.ext_source_2 if req.ext_source_2 is not None else np.nan,
            req.ext_source_3 if req.ext_source_3 is not None else np.nan,
            float(req.region_rating_client),
        ]
        return np.array([row])


service = ModelService()


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {
        "status": "healthy",
        "service": "Enterprise Credit Risk Engine",
        "model_loaded": service.model is not None,
        "optimal_threshold": OPTIMAL_THRESHOLD,
    }


@app.post("/v1/predict", response_model=LoanApplicationResponse)
def score_applicant(request: LoanApplicationRequest):
    """
    Evaluates an applicant, enforces profit-optimal threshold, and generates
    TreeSHAP Adverse Action Notices for loan refusals.
    """
    feature_vector = service.extract_features(request)
    
    # Compute default probability
    prob_default = float(service.model.predict(feature_vector)[0])
    
    # Decision rule using profit-maximizing threshold
    if prob_default >= OPTIMAL_THRESHOLD:
        decision = "REJECTED"
        risk_tier = "HIGH_RISK" if prob_default >= 0.25 else "MEDIUM_RISK"
        # Generate legal Adverse Action Notice using TreeSHAP
        explanation = service.explainer.explain_applicant(feature_vector, top_k=4)
        adverse_reasons = [
            AdverseActionReason(
                feature=r["feature"],
                shap_value=r["shap_value"],
                reason_code=r["reason_code"],
            )
            for r in explanation["adverse_action_reasons"]
        ]
    else:
        decision = "APPROVED"
        risk_tier = "LOW_RISK"
        adverse_reasons = []

    return LoanApplicationResponse(
        decision=decision,
        default_probability=round(prob_default, 4),
        threshold_applied=OPTIMAL_THRESHOLD,
        risk_tier=risk_tier,
        adverse_action_reasons=adverse_reasons,
    )


@app.post("/v1/drift", response_model=DriftCheckResponse)
def check_drift(batch: DriftCheckRequest):
    """
    Ingests streaming applications, evaluates PSI & KS test statistics against baseline.
    """
    if len(batch.applications) < 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Minimum 10 applications required for valid distribution drift calculation.",
        )

    rows = [service.extract_features(app)[0] for app in batch.applications]
    curr_df = pd.DataFrame(rows, columns=MODEL_FEATURES)

    drift_report = service.drift_detector.evaluate_drift(curr_df)

    return DriftCheckResponse(
        overall_status=drift_report["overall_status"],
        max_psi=drift_report["max_psi"],
        feature_metrics=drift_report["features"],
        alert_triggered=drift_report["overall_status"] in ["YELLOW", "RED"],
    )
