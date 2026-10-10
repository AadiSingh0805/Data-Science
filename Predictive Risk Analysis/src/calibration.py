"""
src/calibration.py
Post-hoc probability calibration and asymmetric financial cost optimization.
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss


class RiskCalibrator:
    """
    Handles probability calibration and financial threshold optimization.
    """

    def __init__(self, method: str = "isotonic") -> None:
        """
        method: 'isotonic' (non-parametric, flexible) or 'sigmoid' (Platt scaling, parametric)
        """
        self.method = method
        self.optimal_threshold: float = 0.5
        self.calibrator = None

    def evaluate_calibration(
        self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Computes Brier score and calibration curve bins.
        Lower Brier score indicates better calibrated probabilities.
        """
        brier = float(brier_score_loss(y_true, y_prob))
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        return {
            "brier_score": brier,
            "prob_true": prob_true,
            "prob_pred": prob_pred,
        }

    def optimize_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        cost_fp: float = 500.0,
        cost_fn: float = 5000.0,
        steps: int = 100,
    ) -> Tuple[float, float, pd.DataFrame]:
        """
        Finds the decision threshold that minimizes total asymmetric financial loss:
        Total Cost = (Cost_FP * FP) + (Cost_FN * FN)

        Returns: (optimal_threshold, min_cost, sweep_dataframe)
        """
        thresholds = np.linspace(0.01, 0.99, steps)
        records = []

        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            total_cost = (cost_fp * fp) + (cost_fn * fn)

            records.append({
                "threshold": thresh,
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "total_cost": float(total_cost),
            })

        sweep_df = pd.DataFrame(records)
        best_row = sweep_df.loc[sweep_df["total_cost"].idxmin()]
        self.optimal_threshold = float(best_row["threshold"])

        return self.optimal_threshold, float(best_row["total_cost"]), sweep_df