import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

def compute_profit_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    interest_gain: float = 350.0,
    default_cost: float = 5000.0,
    thresholds: np.ndarray = None,
) -> pd.DataFrame:
    """
    Computes net portfolio profit and confusion matrix quantities across a threshold grid.
    
    Rejection rule:
      y_pred = 1 (Reject)  if y_prob >= threshold
      y_pred = 0 (Approve) if y_prob < threshold
      
    Financial matrix:
      TN: Approved & Repaid -> +interest_gain ($350)
      FP: Rejected & Safe   -> -interest_gain ($350 lost interest)
      FN: Approved & Default -> -default_cost ($5,000 capital write-off)
      TP: Rejected & Default -> $0 (Avoided default)
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.50, 100)
        
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    results = []
    total_samples = len(y_true)
    
    for tau in thresholds:
        # 1 = Reject, 0 = Approve
        y_pred = (y_prob >= tau).astype(int)
        
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        
        # Dollar economics
        revenue = tn * interest_gain
        opportunity_loss = fp * interest_gain
        capital_loss = fn * default_cost
        net_profit = revenue - opportunity_loss - capital_loss
        
        results.append({
            "threshold": tau,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "approval_rate": (tn + fn) / total_samples,
            "rejection_rate": (tp + fp) / total_samples,
            "net_profit": net_profit,
            "capital_loss": capital_loss,
            "opportunity_loss": opportunity_loss,
        })
        
    return pd.DataFrame(results)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    interest_gain: float = 350.0,
    default_cost: float = 5000.0,
) -> Tuple[float, float, pd.DataFrame]:
    """
    Finds the exact probability cutoff that maximizes net portfolio profit.
    Returns: (optimal_threshold, max_profit, curve_dataframe)
    """
    curve_df = compute_profit_curve(
        y_true=y_true,
        y_prob=y_prob,
        interest_gain=interest_gain,
        default_cost=default_cost,
    )
    
    best_row = curve_df.loc[curve_df["net_profit"].idxmax()]
    optimal_tau = float(best_row["threshold"])
    max_profit = float(best_row["net_profit"])
    
    return optimal_tau, max_profit, curve_df

