"""
src/monitoring/drift.py
Statistical drift detection engine:
- Population Stability Index (PSI)
- Two-sample Kolmogorov-Smirnov (KS) tests for continuous feature shift
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


class DriftDetector:
    """
    Monitors distribution shift between reference baseline and live production data.
    """

    def __init__(self, baseline_df: pd.DataFrame, feature_cols: List[str], n_bins: int = 10) -> None:
        self.baseline_df = baseline_df.copy()
        self.feature_cols = feature_cols
        self.n_bins = n_bins

        # Precompute quantile bin edges from baseline distribution
        self.bin_edges_: Dict[str, np.ndarray] = {}
        for col in self.feature_cols:
            vals = self.baseline_df[col].dropna().values
            quantiles = np.linspace(0, 1, self.n_bins + 1)
            edges = np.percentile(vals, quantiles * 100)
            edges[0] = -np.inf
            edges[-1] = np.inf
            self.bin_edges_[col] = np.unique(edges)

    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, edges: np.ndarray) -> float:
        """
        Calculates Population Stability Index across defined bin edges.
        Adds epsilon (1e-4) smoothing to prevent log(0) and division by zero.
        """
        exp_counts, _ = np.histogram(expected, bins=edges)
        act_counts, _ = np.histogram(actual, bins=edges)

        exp_pct = exp_counts / len(expected)
        act_pct = act_counts / len(actual)

        # Smooth zero bins
        eps = 1e-4
        exp_pct = np.where(exp_pct == 0, eps, exp_pct)
        act_pct = np.where(act_pct == 0, eps, act_pct)

        # Renormalize
        exp_pct = exp_pct / np.sum(exp_pct)
        act_pct = act_pct / np.sum(act_pct)

        psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
        return float(psi_val)

    def evaluate_drift(self, current_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs PSI and 2-sample KS test across all monitored feature columns.
        """
        report: Dict[str, Any] = {"features": {}, "alert_triggered": False}

        for col in self.feature_cols:
            base_vals = self.baseline_df[col].dropna().values
            curr_vals = current_df[col].dropna().values
            edges = self.bin_edges_[col]

            psi = self.calculate_psi(base_vals, curr_vals, edges)
            ks_stat, ks_pval = ks_2samp(base_vals, curr_vals)

            drift_level = "NO_DRIFT"
            if psi >= 0.20:
                drift_level = "CRITICAL_DRIFT"
                report["alert_triggered"] = True
            elif psi >= 0.10:
                drift_level = "MODERATE_DRIFT"

            report["features"][col] = {
                "psi": round(psi, 4),
                "ks_statistic": round(float(ks_stat), 4),
                "ks_pvalue": float(ks_pval),
                "status": drift_level,
            }

        return report