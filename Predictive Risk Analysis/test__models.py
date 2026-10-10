"""
test_models.py
Executes Day 2 Track 2 deliverables:
1. Baseline vs. Regularized LightGBM benchmark.
2. Metrics: ROC-AUC and PR-AUC.
3. SHAP TreeExplainer feature attributions.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from src.features import LeakFreePreprocessor
from src.models import RiskModelTrainer


def generate_synthetic_loan_data(n_samples: int = 3000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)

    incomes = np.random.exponential(scale=45000, size=n_samples) + 25000
    debt_to_incomes = np.random.beta(a=2, b=5, size=n_samples)
    occupations = np.random.choice(
        ["Engineer", "Sales", "Clerk", "Executive", "Freelancer"],
        size=n_samples,
        p=[0.25, 0.25, 0.20, 0.15, 0.15],
    )

    occ_risk = {
        "Executive": -1.2,
        "Engineer": -0.8,
        "Clerk": 0.3,
        "Sales": 0.7,
        "Freelancer": 1.1,
    }

    # Non-linear interaction: high debt combined with low income amplifies default risk
    income_factor = np.clip(100000 / incomes, 0.5, 3.5)
    interaction_term = debt_to_incomes * income_factor * 1.5

    logits = (
        interaction_term
        + np.array([occ_risk[o] for o in occupations])
        - 2.8
    )

    probs = 1 / (1 + np.exp(-logits))
    defaults = (np.random.uniform(0, 1, size=n_samples) < probs).astype(int)

    df = pd.DataFrame({
        "income": incomes,
        "debt_to_income": debt_to_incomes,
        "occupation": occupations,
        "default": defaults,
    })

    nan_mask = np.random.uniform(0, 1, size=n_samples) < 0.04
    df.loc[nan_mask, "income"] = np.nan

    return df


def run_benchmark():
    print("=" * 70)
    print("TRACK 2 COMPLETE BENCHMARK: METRICS (ROC-AUC / PR-AUC) & SHAP EXPLAINABILITY")
    print("=" * 70)

    # 1. Synthesize dataset
    raw_df = generate_synthetic_loan_data(n_samples=3000, random_state=42)
    base_rate = raw_df["default"].mean()
    print(f"Applicants: {len(raw_df):,} | Default Base Rate (PR-AUC baseline floor): {base_rate:.2%}")

    # 2. Stratified Holdout Split
    train_df, test_df = train_test_split(
        raw_df, test_size=0.20, stratify=raw_df["default"], random_state=42
    )

    # 3. Preprocess Features
    feature_cols = ["income", "debt_to_income", "occupation_te"]
    preprocessor = LeakFreePreprocessor(
        cat_cols=["occupation"],
        num_cols=["income", "debt_to_income"],
        target_col="default",
        n_splits=5,
        smoothing=10.0,
        random_state=42,
    )

    train_proc = preprocessor.fit_transform_train(train_df)
    test_proc = preprocessor.transform_test(test_df)

    trainer = RiskModelTrainer(
        feature_cols=feature_cols,
        target_col="default",
        n_splits=5,
        random_state=42,
    )

    # 4. Train Baseline (Logistic Regression)
    sub_train, sub_val = train_test_split(
        train_proc, test_size=0.20, stratify=train_proc["default"], random_state=42
    )
    _, base_metrics = trainer.train_baseline(sub_train, sub_val)
    print(f"\n[Baseline Logistic Regression (Validation Split)]")
    print(f"  ROC-AUC: {base_metrics['val_roc_auc']:.4f} | PR-AUC: {base_metrics['val_pr_auc']:.4f}")

    # 5. Train Tuned Champion (LightGBM)
    _, cv_metrics = trainer.train_champion_lgbm(train_proc)
    print(f"\n[Champion LightGBM (5-Fold Stratified OOF)]")
    print(f"  ROC-AUC: {cv_metrics['oof_roc_auc']:.4f} | PR-AUC: {cv_metrics['oof_pr_auc']:.4f}")

    # 6. Evaluate Both on Unseen Test Split
    X_test_scaled = trainer.scaler.transform(test_proc[feature_cols].values)
    base_test_preds = trainer.baseline_model.predict_proba(X_test_scaled)[:, 1]
    base_test_roc = roc_auc_score(test_proc["default"], base_test_preds)
    base_test_pr = average_precision_score(test_proc["default"], base_test_preds)

    champ_test_preds = trainer.predict_champion_ensemble(test_proc)
    champ_test_roc = roc_auc_score(test_proc["default"], champ_test_preds)
    champ_test_pr = average_precision_score(test_proc["default"], champ_test_preds)

    print("\n" + "=" * 70)
    print("UNSEEN HOLDOUT TEST SET PERFORMANCE:")
    print(f"{'Metric':<15} | {'Baseline (LogReg)':<20} | {'Champion (LightGBM)':<20} | {'Uplift':<10}")
    print("-" * 70)
    print(f"{'ROC-AUC':<15} | {base_test_roc:<20.4f} | {champ_test_roc:<20.4f} | {champ_test_roc - base_test_roc:+.4f}")
    print(f"{'PR-AUC':<15} | {base_test_pr:<20.4f} | {champ_test_pr:<20.4f} | {champ_test_pr - base_test_pr:+.4f}")
    print("=" * 70)

    # 7. Compute SHAP Attributions
    print("\n[Computing SHAP Values for Model Explainability...]")
    shap_vals, _ = trainer.explain_with_shap(test_proc)

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Mean_|SHAP|": mean_abs_shap,
    }).sort_values(by="Mean_|SHAP|", ascending=False).reset_index(drop=True)

    print("\nGlobal Feature Importance (Mean Absolute SHAP Value):")
    for idx, row in importance_df.iterrows():
        print(f"  {idx + 1}. {row['Feature']:<16} : {row['Mean_|SHAP|']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()