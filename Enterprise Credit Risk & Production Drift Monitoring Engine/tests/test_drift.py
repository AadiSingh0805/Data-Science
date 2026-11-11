import pytest
import numpy as np
import pandas as pd
from src.monitoring.drift import calculate_psi, calculate_ks, DriftDetector

def test_psi_and_ks_stable_distribution():
    np.random.seed(42)
    # Two samples from the same normal distribution
    ref = np.random.normal(loc=50000, scale=15000, size=5000)
    curr = np.random.normal(loc=50000, scale=15000, size=2000)

    psi_val, breakdown = calculate_psi(ref, curr, num_bins=10)
    ks_stat, ks_pval = calculate_ks(ref, curr)

    # Invariant: No drift should yield PSI well below 0.10 and small KS distance (< 0.05)
    assert psi_val < 0.05, f"Expected negligible PSI, got {psi_val}"
    assert ks_stat < 0.05, f"Expected small KS statistic, got {ks_stat}"
    assert len(breakdown) == 10


def test_psi_and_ks_detects_severe_macroeconomic_drift():
    np.random.seed(42)
    # Baseline: average bureau score 0.55
    ref = np.random.beta(a=4, b=4, size=5000)
    # Macroeconomic recession: bureau scores collapse to mean 0.35
    curr_drifted = np.random.beta(a=2, b=5, size=2000)

    psi_val, _ = calculate_psi(ref, curr_drifted, num_bins=10)
    ks_stat, ks_pval = calculate_ks(ref, curr_drifted)

    # Invariant: Shift should trigger RED alert (PSI > 0.25)
    assert psi_val > 0.25, f"Expected severe drift PSI > 0.25, got {psi_val}"
    assert ks_pval < 1e-5, f"KS test should reject null hypothesis, got {ks_pval}"


def test_drift_detector_end_to_end():
    np.random.seed(42)
    features = ["AMT_INCOME_TOTAL", "EXT_SOURCE_2"]
    
    ref_df = pd.DataFrame({
        "AMT_INCOME_TOTAL": np.random.lognormal(mean=11.5, sigma=0.5, size=3000),
        "EXT_SOURCE_2": np.random.uniform(0.1, 0.9, size=3000),
    })

    # Shifted production data: incomes drop significantly
    curr_df = pd.DataFrame({
        "AMT_INCOME_TOTAL": np.random.lognormal(mean=10.5, sigma=0.5, size=1000),
        "EXT_SOURCE_2": np.random.uniform(0.1, 0.9, size=1000),
    })

    detector = DriftDetector(reference_data=ref_df, feature_names=features)
    report = detector.evaluate_drift(curr_df)

    # Income should be RED, Ext Source 2 should be GREEN
    assert report["features"]["AMT_INCOME_TOTAL"]["status"] == "RED"
    assert report["features"]["EXT_SOURCE_2"]["status"] == "GREEN"
    assert report["overall_status"] == "RED"
