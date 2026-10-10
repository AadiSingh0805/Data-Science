"""
src/models.py
Benchmark model training and explainability module:
1. Baseline L2-regularized Logistic Regression (scaled).
2. Tuned Champion LightGBM classifier with Stratified K-Fold CV, early stopping, and ensemble inference.
3. SHAP TreeExplainer integration for global and local attribution.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import shap


class RiskModelTrainer:
    """
    Orchestrates baseline and gradient-boosted risk model training, evaluation, and SHAP explainability.
    """

    def __init__(
        self,
        feature_cols: List[str],
        target_col: str,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> None:
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.n_splits = n_splits
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.baseline_model = LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=self.random_state,
        )
        self.champion_models: List[lgb.LGBMClassifier] = []

    def train_baseline(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame
    ) -> Tuple[LogisticRegression, Dict[str, float]]:
        """
        Trains L2 Logistic Regression on scaled features.
        Returns: (fitted_model, metrics_dict)
        """
        X_train = train_df[self.feature_cols].values
        y_train = train_df[self.target_col].values
        X_val = val_df[self.feature_cols].values
        y_val = val_df[self.target_col].values

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        self.baseline_model.fit(X_train_scaled, y_train)

        val_preds = self.baseline_model.predict_proba(X_val_scaled)[:, 1]

        metrics = {
            "val_roc_auc": float(roc_auc_score(y_val, val_preds)),
            "val_pr_auc": float(average_precision_score(y_val, val_preds)),
        }
        return self.baseline_model, metrics

    def train_champion_lgbm(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Trains regularized LightGBM using Stratified K-Fold CV.
        Returns: (oof_predictions, cv_metrics_dict)
        """
        X = df[self.feature_cols].values
        y = df[self.target_col].values

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        oof_preds = np.zeros(len(df))
        self.champion_models = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]

            # Tuned hyperparameters for small-to-mid tabular datasets
            model = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.03,
                num_leaves=12,              # Constrain depth to prevent leaf overfitting
                min_child_samples=30,        # Minimum samples required in terminal leaves
                subsample=0.8,               # Row subsampling
                subsample_freq=1,
                colsample_bytree=0.8,        # Feature subsampling
                class_weight="balanced",
                random_state=self.random_state + fold,
                verbosity=-1,
            )

            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
            )

            val_preds = model.predict_proba(X_va)[:, 1]
            oof_preds[val_idx] = val_preds
            self.champion_models.append(model)

        cv_metrics = {
            "oof_roc_auc": float(roc_auc_score(y, oof_preds)),
            "oof_pr_auc": float(average_precision_score(y, oof_preds)),
        }
        return oof_preds, cv_metrics

    def predict_champion_ensemble(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Ensemble prediction by averaging output probabilities across all K fold models.
        """
        X_test = test_df[self.feature_cols].values
        fold_preds = [
            model.predict_proba(X_test)[:, 1] for model in self.champion_models
        ]
        return np.mean(fold_preds, axis=0)

    def explain_with_shap(
        self, test_df: pd.DataFrame
    ) -> Tuple[np.ndarray, shap.TreeExplainer]:
        """
        Computes SHAP values using the first fold champion model.
        Returns: (shap_values, explainer)
        """
        X_test = test_df[self.feature_cols]
        primary_model = self.champion_models[0]
        explainer = shap.TreeExplainer(primary_model)
        shap_values = explainer.shap_values(X_test)

        # Handle binary classification output shape differences across SHAP versions
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]

        return shap_values, explainer