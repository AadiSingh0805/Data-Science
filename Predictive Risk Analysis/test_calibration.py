"""
test_calibration.py
Validates calibration curve, Brier score, and business cost optimization.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from src.features import LeakFreePreprocessor
from src.models import RiskModelTrainer
from src.calibration import RiskCalibrator
from test__models import generate_synthetic_loan_data

# 1. Synthesize and prepare data
raw_df = generate_synthetic_loan_data(n_samples=3000, random_state=42)
train_df, test_df = train_test_split(raw_df, test_size=0.20, stratify=raw_df["default"], random_state=42)

preprocessor = LeakFreePreprocessor(
    cat_cols=["occupation"],
    num_cols=["income", "debt_to_income"],
    target_col="default",
    random_state=42
)
train_proc = preprocessor.fit_transform_train(train_df)
test_proc = preprocessor.transform_test(test_df)

feature_cols = ["income", "debt_to_income", "occupation_te"]
trainer = RiskModelTrainer(feature_cols=feature_cols, target_col="default", random_state=42)
trainer.train_champion_lgbm(train_proc)

# 2. Get predictions on holdout test set
test_preds = trainer.predict_champion_ensemble(test_proc)
y_test = test_proc["default"].values

# 3. Evaluate Calibration & Find Business-Optimal Cutoff
calibrator = RiskCalibrator()
metrics = calibrator.evaluate_calibration(y_test, test_preds)
print(f"Brier Score Loss: {metrics['brier_score']:.4f}")

# Business loss: FP = $500 (rejected good borrower), FN = $5,000 (bad loan write-off)
best_thresh, min_cost, sweep_df = calibrator.optimize_threshold(
    y_test, test_preds, cost_fp=500.0, cost_fn=5000.0
)

# Compare default 0.5 cutoff vs optimal cutoff cost
cost_at_05 = sweep_df.iloc[(sweep_df["threshold"] - 0.5).abs().argsort()[:1]]["total_cost"].values[0]

print("\n" + "=" * 60)
print("FINANCIAL THRESHOLD OPTIMIZATION RESULTS:")
print(f"  Naive Threshold (0.50) Total Cost  : ${cost_at_05:,.2f}")
print(f"  Optimal Threshold ({best_thresh:.2f}) Total Cost: ${min_cost:,.2f}")
print(f"  Financial Savings                  : ${cost_at_05 - min_cost:,.2f}")
print("=" * 60)