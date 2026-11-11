import pytest
import numpy as np
from src.models.cost_matrix import compute_profit_curve, find_optimal_threshold

def test_optimal_threshold_is_far_below_half():
    # 10,000 synthetic loans with ~8% default rate
    np.random.seed(42)
    n = 10_000
    
    # Ground truth probabilities centered around 0.08
    true_probs = np.random.beta(a=1, b=11, size=n)
    y_true = (np.random.rand(n) < true_probs).astype(int)
    
    # Model predictions (slightly noisy version of true probs)
    y_prob = np.clip(true_probs + np.random.normal(0, 0.02, size=n), 0.001, 0.999)
    
    opt_tau, max_profit, curve_df = find_optimal_threshold(
        y_true=y_true,
        y_prob=y_prob,
        interest_gain=350.0,
        default_cost=5000.0,
    )
    
    # Assert optimal threshold is significantly below 0.50 (around 0.05 - 0.15)
    assert 0.04 <= opt_tau <= 0.15, f"Expected optimal tau in [0.04, 0.15], got {opt_tau}"
    
    # Compare profit at optimal threshold vs naive 0.50 threshold
    profit_at_half = curve_df.loc[(curve_df["threshold"] - 0.50).abs().idxmin(), "net_profit"]
    
    # The optimal threshold must earn substantially more than naive 0.50
    assert max_profit > profit_at_half, "Optimal threshold failed to beat naive 0.50 threshold"
    assert (max_profit - profit_at_half) > 100_000, "Profit uplift should be massive in dollars"

