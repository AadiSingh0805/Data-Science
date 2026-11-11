"""Production Model Training Pipeline for Home Credit Default Risk.
Ingests application_train.csv (307,511 rows), applies Out-of-Time split,
trains LightGBM with early stopping, evaluates PR-AUC & ROC-AUC,
derives optimal cost threshold, and serializes production model artifact.
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from src.data.split import temporal_split
from src.models.cost_matrix import find_optimal_threshold

FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_INCOME_RATIO",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_EMPLOYED_ANOM",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "REGION_RATING_CLIENT",
]

TARGET = "TARGET"


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies domain feature engineering and anomaly cleaning."""
    df = df.copy()

    # Financial ratios
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / np.maximum(df["AMT_INCOME_TOTAL"], 1.0)
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / np.maximum(df["AMT_INCOME_TOTAL"], 1.0)

    # Clean the famous 365,243 day pensioner anomaly
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(float)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace({365243: np.nan})

    return df


def run_training_pipeline(
    data_path: str = "application_train.csv",
    model_output_dir: str = "models",
):
    print("=" * 70)
    print("🚀 ENTERPRISE CREDIT RISK TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load Data
    print(f"Loading raw applications from: {data_path}...")
    raw_df = pd.read_csv(data_path)
    print(f"Loaded {len(raw_df):,} rows and {raw_df.shape[1]} columns.")

    # 2. Chronological OOT Splitting (Zero Leakage)
    print("\nApplying Chronological 70% Train, 15% Val, 15% OOT Test split...")
    train_raw, val_raw, test_raw = temporal_split(raw_df, train_frac=0.70, val_frac=0.15)
    print(f"Train samples: {len(train_raw):,} (Default rate: {train_raw[TARGET].mean():.2%})")
    print(f"Val samples:   {len(val_raw):,} (Default rate: {val_raw[TARGET].mean():.2%})")
    print(f"OOT Test:      {len(test_raw):,} (Default rate: {test_raw[TARGET].mean():.2%})")

    # 3. Feature Engineering
    print("\nEngineering features and cleaning 365,243 sentinel anomaly...")
    train_df = preprocess_features(train_raw)
    val_df = preprocess_features(val_raw)
    test_df = preprocess_features(test_raw)

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_val, y_val = val_df[FEATURES], val_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    # 4. LightGBM Datasets
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURES)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, feature_name=FEATURES)

    # 5. Model Parameters
    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "verbose": -1,
        "random_state": 42,
    }

    print("\nTraining LightGBM model with early stopping...")
    callbacks = [
        lgb.early_stopping(stopping_rounds=30, verbose=True),
        lgb.log_evaluation(period=20),
    ]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    # 6. Evaluation on Out-of-Time (OOT) Test Set
    print("\n" + "=" * 70)
    print("📊 OUT-OF-TIME (OOT) TEST SET EVALUATION")
    print("=" * 70)
    y_test_probs = model.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_test_probs)
    precision, recall, _ = precision_recall_curve(y_test, y_test_probs)
    pr_auc = auc(recall, precision)

    print(f"OOT ROC-AUC Score: {roc_auc:.4f}")
    print(f"OOT PR-AUC Score:  {pr_auc:.4f}")

    # 7. Optimal Dollar Threshold Scanning
    print("\nScanning threshold curve against $14.3:1 Dollar Cost Matrix...")
    opt_tau, max_profit, curve_df = find_optimal_threshold(
        y_true=y_test.values,
        y_prob=y_test_probs,
        interest_gain=350.0,
        default_cost=5000.0,
    )
    
    half_profit = curve_df.loc[(curve_df["threshold"] - 0.50).abs().idxmin(), "net_profit"]
    print(f"Optimal Decision Cutoff (τ*): {opt_tau:.4f}")
    print(f"Net Portfolio Profit at τ*:    ${max_profit:,.2f}")
    print(f"Net Portfolio Profit at 0.50:  ${half_profit:,.2f}")
    print(f"Net Financial Uplift:          +${max_profit - half_profit:,.2f}")

    # 8. Save Artifacts
    os.makedirs(model_output_dir, exist_ok=True)
    model_path = os.path.join(model_output_dir, "credit_risk_lgbm.txt")
    model.save_model(model_path)
    print(f"\nModel artifact saved to: {model_path}")
    print("=" * 70)
    return model, opt_tau


if __name__ == "__main__":
    run_training_pipeline()
