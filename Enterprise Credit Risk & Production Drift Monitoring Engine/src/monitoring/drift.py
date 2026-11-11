import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from typing import Dict, List, Tuple, Any

def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> Tuple[float, pd.DataFrame]:
    """
    Computes the Population Stability Index (PSI) from scratch.
    
    Formula:
      PSI = sum( (P_k - Q_k) * ln(P_k / Q_k) )
      
    Args:
      reference: Baseline / Training feature distribution (P)
      current: Production streaming feature distribution (Q)
      num_bins: Number of quantile bins established on reference (default: 10 deciles)
      epsilon: Small pseudo-constant to prevent division by zero or ln(0)
    """
    ref = np.asarray(reference)
    curr = np.asarray(current)
    
    # Remove NaN values for distribution comparison
    ref = ref[~np.isnan(ref)]
    curr = curr[~np.isnan(curr)]
    
    if len(ref) == 0 or len(curr) == 0:
        return 0.0, pd.DataFrame()

    # 1. Establish quantile bin cutoffs on the baseline reference data
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref, quantiles)
    
    # Ensure strictly increasing bin edges (handles low-cardinality ties)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    # Add minor noise to duplicates if any
    for i in range(1, len(bin_edges) - 1):
        if bin_edges[i] <= bin_edges[i - 1]:
            bin_edges[i] = bin_edges[i - 1] + 1e-5

    # 2. Count samples in each bin
    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    curr_counts, _ = np.histogram(curr, bins=bin_edges)

    # 3. Compute proportions (P_k and Q_k)
    p_k = ref_counts / len(ref)
    q_k = curr_counts / len(curr)

    # 4. Apply epsilon smoothing for numerical stability
    p_k = np.clip(p_k, epsilon, 1.0)
    q_k = np.clip(q_k, epsilon, 1.0)

    # Re-normalize to sum to 1.0
    p_k = p_k / np.sum(p_k)
    q_k = q_k / np.sum(q_k)

    # 5. Compute PSI per bin and sum
    bin_psi = (p_k - q_k) * np.log(p_k / q_k)
    total_psi = float(np.sum(bin_psi))

    breakdown_df = pd.DataFrame({
        "bin": range(1, len(bin_psi) + 1),
        "ref_proportion": p_k,
        "curr_proportion": q_k,
        "bin_psi": bin_psi,
    })

    return total_psi, breakdown_df


def calculate_ks(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
    """
    Computes Two-Sample Kolmogorov-Smirnov statistic and p-value.
    D_ks = max |F_ref(x) - F_curr(x)|
    """
    ref = np.asarray(reference)
    curr = np.asarray(current)
    ref = ref[~np.isnan(ref)]
    curr = curr[~np.isnan(curr)]
    
    res = ks_2samp(ref, curr)
    return float(res.statistic), float(res.pvalue)


class DriftDetector:
    """
    Production drift monitoring engine.
    Compares incoming streaming feature buffers against baseline reference data.
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        feature_names: List[str],
        psi_threshold_yellow: float = 0.10,
        psi_threshold_red: float = 0.25,
    ):
        self.reference_data = reference_data.copy()
        self.feature_names = feature_names
        self.psi_yellow = psi_threshold_yellow
        self.psi_red = psi_threshold_red

    def evaluate_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluates PSI and KS statistics for all monitored features.
        Returns detailed summary and alert levels.
        """
        feature_reports = {}
        max_psi = 0.0
        overall_status = "GREEN"

        for feat in self.feature_names:
            if feat not in self.reference_data.columns or feat not in current_data.columns:
                continue

            ref_series = self.reference_data[feat].values
            curr_series = current_data[feat].values

            psi_val, _ = calculate_psi(ref_series, curr_series)
            ks_stat, ks_pval = calculate_ks(ref_series, curr_series)

            # Determine traffic light status
            if psi_val >= self.psi_red:
                status = "RED"
            elif psi_val >= self.psi_yellow:
                status = "YELLOW"
            else:
                status = "GREEN"

            if psi_val > max_psi:
                max_psi = psi_val

            feature_reports[feat] = {
                "psi": round(psi_val, 4),
                "ks_stat": round(ks_stat, 4),
                "ks_pvalue": round(ks_pval, 6),
                "status": status,
            }

        # Overall status takes worst case
        if max_psi >= self.psi_red:
            overall_status = "RED"
        elif max_psi >= self.psi_yellow:
            overall_status = "YELLOW"
        else:
            overall_status = "GREEN"

        return {
            "overall_status": overall_status,
            "max_psi": round(max_psi, 4),
            "features": feature_reports,
        }

