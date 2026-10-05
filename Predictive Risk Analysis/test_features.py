"""
test_features.py
Direct verification script for LeakFreePreprocessor.
"""

import numpy as np
import pandas as pd
from src.features import LeakFreePreprocessor


def run_verification():
    print("=" * 60)
    print("RUNNING LEAK-FREE PREPROCESSOR TEST")
    print("=" * 60)

    # 1. Construct synthetic training data (8 applicants)
    train_df = pd.DataFrame({
        "applicant_id": [1, 2, 3, 4, 5, 6, 7, 8],
        "income": [50000.0, 75000.0, np.nan, 120000.0, 30000.0, 85000.0, np.nan, 45000.0],
        "debt_to_income": [0.25, 0.40, 0.75, 0.15, 0.55, 0.30, 0.85, 0.20],
        "occupation": [
            "Engineer", "Sales", "Sales", "Manager",
            "Clerk", "Engineer", "Clerk", "Manager"
        ],
        "default": [0, 0, 1, 0, 1, 0, 1, 0],  # Overall mean = 3 / 8 = 0.375
    })

    # 2. Construct synthetic test data (3 applicants, including an unseen occupation)
    test_df = pd.DataFrame({
        "applicant_id": [101, 102, 103],
        "income": [np.nan, 95000.0, 60000.0],  # Row 1 has missing income
        "debt_to_income": [0.35, 0.18, 0.50],
        "occupation": ["Engineer", "Sales", "Astronaut"],  # 'Astronaut' was never in train
    })

    cat_cols = ["occupation"]
    num_cols = ["income", "debt_to_income"]
    target_col = "default"

    # Initialize with 2 splits and m=5.0 smoothing
    preprocessor = LeakFreePreprocessor(
        cat_cols=cat_cols,
        num_cols=num_cols,
        target_col=target_col,
        n_splits=2,
        smoothing=5.0,
        random_state=42,
    )

    print("\n[Step 1] Fitting and transforming training data...")
    train_processed = preprocessor.fit_transform_train(train_df)

    print("\nProcessed Training Split:")
    print(train_processed[["applicant_id", "income", "occupation", "occupation_te", "default"]])

    # Assertions on train
    assert not train_processed["income"].isna().any(), "Train income still contains NaNs!"
    assert not train_processed["occupation_te"].isna().any(), "Train occupation_te contains NaNs!"
    expected_median = pd.Series([50000.0, 75000.0, 120000.0, 30000.0, 85000.0, 45000.0]).median()
    assert preprocessor.train_medians_["income"] == expected_median, "Median calculation mismatch!"
    print(f"\nTraining median learned for 'income': ${preprocessor.train_medians_['income']:,.2f}")
    print(f"Global training default rate: {preprocessor.global_target_mean_:.4f}")

    print("\n[Step 2] Transforming unseen test data...")
    test_processed = preprocessor.transform_test(test_df)

    print("\nProcessed Test Split:")
    print(test_processed[["applicant_id", "income", "occupation", "occupation_te"]])

    # Assertions on test
    assert not test_processed["income"].isna().any(), "Test income still contains NaNs!"
    assert not test_processed["occupation_te"].isna().any(), "Test occupation_te contains NaNs!"
    
    # Check that unseen 'Astronaut' defaulted to global target mean
    astronaut_score = test_processed.loc[test_processed["occupation"] == "Astronaut", "occupation_te"].values[0]
    assert np.isclose(astronaut_score, preprocessor.global_target_mean_), "Unseen category did not get global mean!"
    print(f"\nUnseen category 'Astronaut' correctly mapped to global mean: {astronaut_score:.4f}")

    print("\n[SUCCESS] Pipeline passed all leakage and transformation checks.")
    print("=" * 60)


if __name__ == "__main__":
    run_verification()