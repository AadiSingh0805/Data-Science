"""
test_drift.py
Validates drift detector against normal traffic vs. shifted economic conditions.
"""

import numpy as np
import pandas as pd
from src.monitoring.drift import DriftDetector
from test__models import generate_synthetic_loan_data

# 1. Baseline historical data
baseline_df = generate_synthetic_loan_data(n_samples=2500, random_state=42)
feature_cols = ["income", "debt_to_income"]

detector = DriftDetector(baseline_df=baseline_df, feature_cols=feature_cols)

# 2. Case A: Normal live traffic (same distribution)
normal_live_df = generate_synthetic_loan_data(n_samples=1000, random_state=999)
normal_report = detector.evaluate_drift(normal_live_df)

print("=" * 65)
print("SCENARIO 1: NORMAL PRODUCTION TRAFFIC")
for feat, metrics in normal_report["features"].items():
    print(f"  {feat:<16} | PSI: {metrics['psi']:.4f} | Status: {metrics['status']}")
print(f"  Alert Triggered : {normal_report['alert_triggered']}")

# 3. Case B: Economic shock (incomes drop 30%, debt-to-income spikes 40%)
drifted_live_df = normal_live_df.copy()
drifted_live_df["income"] = drifted_live_df["income"] * 0.70
drifted_live_df["debt_to_income"] = drifted_live_df["debt_to_income"] * 1.40

drifted_report = detector.evaluate_drift(drifted_live_df)

print("\n" + "=" * 65)
print("SCENARIO 2: ECONOMIC SHOCK (SEVERE FEATURE DRIFT)")
for feat, metrics in drifted_report["features"].items():
    print(f"  {feat:<16} | PSI: {metrics['psi']:.4f} | KS p-val: {metrics['ks_pvalue']:.2e} | Status: {metrics['status']}")
print(f"  Alert Triggered : {drifted_report['alert_triggered']}")
print("=" * 65)