"""
src/features.py
Production-grade tabular preprocessing with leak-free Out-of-Fold (OOF)
target encoding and deterministic median imputation.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


class LeakFreePreprocessor:
    """
    Leak-free feature processor for tabular risk datasets.
    
    Attributes:
        cat_cols (List[str]): Categorical column names to target-encode.
        num_cols (List[str]): Numerical column names to median-impute.
        target_col (str): Target column name (binary: 0 or 1).
        n_splits (int): Number of folds for out-of-fold target encoding.
        smoothing (float): Additive m-estimate smoothing weight.
    """

    def __init__(
        self,
        cat_cols: List[str],
        num_cols: List[str],
        target_col: str,
        n_splits: int = 5,
        smoothing: float = 10.0,
        random_state: int = 42,
    ) -> None:
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.target_col = target_col
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.random_state = random_state

        self.global_target_mean_: Optional[float] = None
        self.train_medians_: Dict[str, float] = {}
        self.full_target_encodings_: Dict[str, pd.Series] = {}

    def _calc_smoothed_mean(
        self, count: pd.Series, mean: pd.Series, global_mean: float
    ) -> pd.Series:
        """
        Calculates m-estimate smoothed category mean:
        S_i = (count * mean + m * global_mean) / (count + m)
        """
        return (count * mean + self.smoothing * global_mean) / (count + self.smoothing)

    def fit_transform_train(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fits numerical medians and categorical encodings on training data.
        Returns transformed training DataFrame with OOF target-encoded features.
        """
        df = df.copy()
        y = df[self.target_col].values
        self.global_target_mean_ = float(np.mean(y))

        # 1. Learn and apply numerical medians strictly on train
        for col in self.num_cols:
            median_val = float(df[col].median())
            self.train_medians_[col] = median_val
            df[col] = df[col].fillna(median_val)

        # 2. Out-of-Fold target encoding for categoricals
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        for col in self.cat_cols:
            oof_col = np.full(len(df), np.nan)

            for train_idx, val_idx in skf.split(df, y):
                tr_subset = df.iloc[train_idx]
                val_subset = df.iloc[val_idx]

                # Aggregate only over the complementary training folds
                stats = tr_subset.groupby(col, observed=True)[self.target_col].agg(
                    ["count", "mean"]
                )
                smoothed = self._calc_smoothed_mean(
                    stats["count"], stats["mean"], self.global_target_mean_
                )

                # Map onto validation fold; fallback to global mean for unseen categories
                oof_col[val_idx] = (
                    val_subset[col]
                    .map(smoothed)
                    .fillna(self.global_target_mean_)
                    .astype(float)
                )

            df[f"{col}_te"] = oof_col

            # 3. Store full training-set encodings for test/production inference
            full_stats = df.groupby(col, observed=True)[self.target_col].agg(
                ["count", "mean"]
            )
            self.full_target_encodings_[col] = self._calc_smoothed_mean(
                full_stats["count"], full_stats["mean"], self.global_target_mean_
            )

        return df

    def transform_test(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms unseen test or real-time inference data using stored training artifacts.
        Does not look at or require the target column.
        """
        df = df.copy()

        # Apply stored training medians
        for col in self.num_cols:
            df[col] = df[col].fillna(self.train_medians_.get(col, 0.0))

        # Apply stored target encodings
        for col in self.cat_cols:
            encoding_map = self.full_target_encodings_.get(col)
            if encoding_map is not None and self.global_target_mean_ is not None:
                df[f"{col}_te"] = (
                    df[col]
                    .map(encoding_map)
                    .fillna(self.global_target_mean_)
                    .astype(float)
                )
            else:
                fallback = self.global_target_mean_ if self.global_target_mean_ is not None else 0.0
                df[f"{col}_te"] = fallback

        return df