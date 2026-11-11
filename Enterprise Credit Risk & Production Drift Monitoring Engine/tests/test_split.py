import pytest
import pandas as pd
import numpy as np
from src.data.split import temporal_split

def test_temporal_split_invariants():
    # Construct a synthetic dataframe with sequential IDs
    n = 1000
    df = pd.DataFrame({
        "SK_ID_CURR": np.arange(100000, 100000 + n),
        "TARGET": np.random.choice([0, 1], size=n, p=[0.92, 0.08]),
        "AMT_CREDIT": np.random.uniform(10000, 500000, size=n),
    })

    train_df, val_df, test_df = temporal_split(df, train_frac=0.70, val_frac=0.15)

    # Invariant 1: Total rows conserved
    assert len(train_df) == 700
    assert len(val_df) == 150
    assert len(test_df) == 150
    assert len(train_df) + len(val_df) + len(test_df) == len(df)

    # Invariant 2: Zero temporal overlap / strict chronological separation
    assert train_df["SK_ID_CURR"].max() < val_df["SK_ID_CURR"].min()
    assert val_df["SK_ID_CURR"].max() < test_df["SK_ID_CURR"].min()

    # Invariant 3: Both classes present in each fold
    assert train_df["TARGET"].nunique() == 2
    assert val_df["TARGET"].nunique() == 2
    assert test_df["TARGET"].nunique() == 2

