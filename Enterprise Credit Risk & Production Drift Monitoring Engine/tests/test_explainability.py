import pytest
import numpy as np
import lightgbm as lgb
from src.models.explainability import RiskExplainer, REASON_CODE_MAPPING

def test_treeshap_additivity_and_adverse_action():
    # 1. Train a minimal LightGBM model on synthetic tabular data
    np.random.seed(42)
    n = 200
    features = [
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "AMT_ANNUITY",
        "ANNUITY_INCOME_RATIO",
        "DAYS_EMPLOYED",
    ]
    
    X = np.random.randn(n, len(features))
    # Make EXT_SOURCE_2 and ANNUITY_INCOME_RATIO strong drivers
    y = (X[:, 1] * -1.5 + X[:, 3] * 1.8 + np.random.randn(n) * 0.5 > 0).astype(int)
    
    train_data = lgb.Dataset(X, label=y, feature_name=features)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 8,
        "learning_rate": 0.1,
        "verbose": -1,
    }
    model = lgb.train(params, train_data, num_boost_round=15)
    
    # 2. Initialize RiskExplainer
    explainer = RiskExplainer(model=model, feature_names=features)
    
    # 3. Test on an applicant with high default risk (high debt ratio, low external score)
    applicant = np.array([[-0.5, -1.8, 1.2, 2.5, -0.3]])
    explanation = explainer.explain_applicant(applicant, top_k=3)
    
    # Verify Additivity Property: reconstructed prediction must match model raw margin
    raw_pred = model.predict(applicant, raw_score=True)[0]
    expected_pred = explanation["base_value"] + explanation["total_shap_attribution"]
    assert np.isclose(expected_pred, raw_pred, atol=1e-3), (
        f"Additivity violated! Expected {raw_pred}, got {expected_pred}"
    )
    
    # Verify Adverse Action Reasons
    reasons = explanation["adverse_action_reasons"]
    assert len(reasons) <= 3
    assert len(reasons) > 0
    
    # Top reason must have positive SHAP (risk increasing)
    assert reasons[0]["shap_value"] > 0
    # Must be sorted descending
    if len(reasons) > 1:
        assert reasons[0]["shap_value"] >= reasons[1]["shap_value"]
        
    # Reason text must match our regulatory dictionary
    top_feat = reasons[0]["feature"]
    assert reasons[0]["reason_code"] == REASON_CODE_MAPPING[top_feat]

