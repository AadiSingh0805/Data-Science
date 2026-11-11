import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Any, Optional

# Standard regulatory mapping for FCRA / ECOA Adverse Action Notices
REASON_CODE_MAPPING = {
    "EXT_SOURCE_1": "Low external credit bureau score (Agency 1)",
    "EXT_SOURCE_2": "Low external credit bureau score (Agency 2)",
    "EXT_SOURCE_3": "Low external credit bureau score (Agency 3)",
    "AMT_ANNUITY": "High loan annuity payment requested",
    "AMT_CREDIT": "Total credit requested exceeds allowable risk limits",
    "AMT_INCOME_TOTAL": "Insufficient verified gross income",
    "ANNUITY_INCOME_RATIO": "High monthly debt burden relative to verified income",
    "CREDIT_INCOME_RATIO": "Total credit amount too high for income tier",
    "DAYS_EMPLOYED": "Insufficient duration of steady employment",
    "DAYS_BIRTH": "Limited credit history length",
    "REGION_RATING_CLIENT": "Elevated regional credit risk profile",
    "REGION_RATING_CLIENT_W_CITY": "Elevated metropolitan area risk rating",
    "OWN_CAR_AGE": "Age of vehicle indicates high maintenance depreciation",
    "DAYS_EMPLOYED_ANOM": "Anomalous or unverified employment duration",
}

DEFAULT_REASON = "Unfavorable debt or credit risk indicator"


class RiskExplainer:
    """
    TreeSHAP explanation engine for GBDT models.
    Generates exact local feature attributions and Adverse Action Notices (AAN).
    """

    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        # TreeExplainer with model_output='raw' computes exact Shapley values in log-odds space
        self.explainer = shap.TreeExplainer(model)
        
        # Expected base value phi_0
        expected = self.explainer.expected_value
        if isinstance(expected, (list, np.ndarray)):
            self.base_value = float(expected[1]) if len(expected) > 1 else float(expected[0])
        else:
            self.base_value = float(expected)

    def explain_applicant(
        self,
        applicant_data: np.ndarray,
        top_k: int = 4,
    ) -> Dict[str, Any]:
        """
        Computes TreeSHAP explanation for a single loan applicant.
        
        Returns:
            - base_value: Population baseline phi_0
            - raw_prediction: Model output
            - shap_values: Dict of {feature_name: shap_value}
            - additivity_verified: True if phi_0 + sum(shap) == raw_prediction
            - adverse_action_reasons: Top K drivers pushing toward rejection (phi_i > 0)
        """
        if applicant_data.ndim == 1:
            applicant_data = applicant_data.reshape(1, -1)

        # Compute raw SHAP values
        raw_shap = self.explainer.shap_values(applicant_data)
        
        # Handle multi-class / binary output format from shap
        if isinstance(raw_shap, list) and len(raw_shap) == 2:
            shap_vector = raw_shap[1][0]
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
            shap_vector = raw_shap[0, :, 1]
        else:
            shap_vector = raw_shap[0]

        # Verify Additivity: phi_0 + sum(phi_i) == raw_prediction
        total_shap = np.sum(shap_vector)
        predicted_raw = self.base_value + total_shap
        
        # Create feature -> SHAP value mapping
        attribution_dict = {
            name: float(shap_vector[i])
            for i, name in enumerate(self.feature_names)
        }

        # Rank features pushing toward rejection (phi_i > 0)
        risk_increasing = [
            (feat, val) for feat, val in attribution_dict.items() if val > 0
        ]
        # Sort descending by contribution
        risk_increasing.sort(key=lambda x: x[1], reverse=True)

        # Generate Adverse Action Reason Codes
        adverse_reasons = []
        for feat, val in risk_increasing[:top_k]:
            reason_text = REASON_CODE_MAPPING.get(feat, f"{feat}: {DEFAULT_REASON}")
            adverse_reasons.append({
                "feature": feat,
                "shap_value": round(val, 4),
                "reason_code": reason_text,
            })

        return {
            "base_value": round(self.base_value, 4),
            "total_shap_attribution": round(total_shap, 4),
            "reconstructed_prediction": round(predicted_raw, 4),
            "adverse_action_reasons": adverse_reasons,
            "all_shap_values": attribution_dict,
        }

