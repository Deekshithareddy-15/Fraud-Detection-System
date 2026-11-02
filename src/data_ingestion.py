"""
Data Ingestion and Validation Module for Credit Card Fraud Detection.

Maintains two strictly isolated modes:
1. 'real': Strictly expects the genuine Kaggle Credit Card Fraud Detection dataset
   (284,807 rows, 492 fraud cases). Validates schema, types, and class balance.
   NEVER silently generates or falls back to synthetic data.
2. 'synthetic': Benchmark generator using make_classification for development,
   testing, and CI environments without downloading the 150MB file.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("data_ingestion")

# Canonical schema definition
PCA_FEATURES: List[str] = [f"V{i}" for i in range(1, 29)]
NUMERICAL_FEATURES: List[str] = ["Time", "Amount"]
ALL_FEATURE_COLUMNS: List[str] = PCA_FEATURES + NUMERICAL_FEATURES
FEATURE_COLUMNS: List[str] = ALL_FEATURE_COLUMNS
COLS_TO_SCALE: List[str] = ["Amount", "Time"]
TARGET_COLUMN: str = "Class"
REQUIRED_COLUMNS: List[str] = ["Time"] + PCA_FEATURES + ["Amount", TARGET_COLUMN]

# Kaggle dataset ground truth benchmarks
KAGGLE_EXPECTED_ROWS: int = 284807
KAGGLE_EXPECTED_FRAUD: int = 492
KAGGLE_URL: str = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"


class DatasetError(Exception):
    """Base exception for dataset-related errors."""
    pass


class DatasetNotFoundError(DatasetError, FileNotFoundError):
    """Raised when the Kaggle credit card dataset cannot be found on disk."""
    pass


class DatasetValidationError(DatasetError, ValueError):
    """Raised when dataset structure, schema, or row counts fail validation checks."""
    pass


def validate_dataset(
    df: pd.DataFrame,
    is_training: bool = True,
    strict_numeric: bool = True
) -> pd.DataFrame:
    """
    Validates that a DataFrame conforms to the expected Credit Card Fraud schema.

    Parameters:
        df: Input DataFrame to validate.
        is_training: If True, checks for presence and validity of TARGET_COLUMN ('Class').
        strict_numeric: If True, converts and verifies all feature columns are numeric.

    Returns:
        pd.DataFrame: Sanitized DataFrame without NaN, infinite, or corrupted rows.

    Raises:
        DatasetValidationError: If required columns are missing, data types are invalid,
                                or the target column contains invalid class labels.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise DatasetValidationError("Expected a pandas DataFrame, received None or invalid object.")

    if df.empty:
        raise DatasetValidationError("Provided dataset is empty (0 rows).")

    # Check required columns
    expected_cols = REQUIRED_COLUMNS if is_training else ALL_FEATURE_COLUMNS
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        raise DatasetValidationError(
            f"Dataset is missing required columns: {missing_cols}. "
            f"Expected columns: {expected_cols}"
        )

    df_cleaned = df.copy()

    # Validate numeric types and coerce if possible
    if strict_numeric:
        check_cols = ALL_FEATURE_COLUMNS + ([TARGET_COLUMN] if is_training else [])
        for col in check_cols:
            if not pd.api.types.is_numeric_dtype(df_cleaned[col]):
                try:
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce")
                except Exception as e:
                    raise DatasetValidationError(f"Column '{col}' cannot be converted to numeric: {e}")

    # Check target column semantics
    if is_training:
        unique_classes = set(df_cleaned[TARGET_COLUMN].dropna().unique())
        if not unique_classes.issubset({0, 1, 0.0, 1.0}):
            raise DatasetValidationError(
                f"Target column '{TARGET_COLUMN}' contains invalid labels: {unique_classes}. "
                "Expected binary classification values {0, 1}."
            )
        df_cleaned[TARGET_COLUMN] = df_cleaned[TARGET_COLUMN].astype(int)

    # Sanitize invalid records (NaN, Inf, -Inf)
    initial_count = len(df_cleaned)
    df_cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
    null_counts = df_cleaned[expected_cols].isnull().sum().sum()

    if null_counts > 0:
        logger.warning("Found %d null/invalid values. Dropping affected rows.", null_counts)
        df_cleaned.dropna(subset=expected_cols, inplace=True)
        logger.info("Dropped %d corrupted rows.", initial_count - len(df_cleaned))

    if df_cleaned.empty:
        raise DatasetValidationError("All rows were dropped during sanitization due to invalid values.")

    return df_cleaned


def load_real_dataset(
    filepath: Union[str, Path] = "data/raw/creditcard.csv",
    validate: bool = True,
    strict_kaggle_check: bool = True
) -> pd.DataFrame:
    """
    Loads and validates the real Kaggle Credit Card Fraud Detection dataset.
    
    Strictly verifies:
    - File exists
    - Schema matches Time, V1-V28, Amount, Class
    - Row count is 284,807 and fraud count is 492 (when strict_kaggle_check=True)
    - NEVER silently substitutes with synthetic data
    """
    path_obj = Path(filepath).resolve()

    if not path_obj.exists() or not path_obj.is_file():
        err_msg = (
            f"\n[REAL DATASET MISSING] Kaggle Credit Card Fraud dataset not found at '{path_obj}'.\n"
            "To use the real dataset:\n"
            f"  1. Download 'creditcard.csv' from Kaggle: {KAGGLE_URL}\n"
            f"  2. Place the uncompressed CSV at: '{path_obj}'\n"
            "For automated testing or development without the Kaggle file, explicitly run in "
            "synthetic mode using: python src/train.py --mode synthetic"
        )
        logger.error(err_msg)
        raise DatasetNotFoundError(err_msg)

    logger.info("Loading dataset from: %s", path_obj)
    try:
        df = pd.read_csv(path_obj)
    except Exception as e:
        raise DatasetError(f"Failed to read CSV at '{path_obj}': {str(e)}") from e

    logger.info("Read CSV with %d rows and %d columns.", len(df), len(df.columns))

    if validate:
        df = validate_dataset(df, is_training=True)

    row_count = len(df)
    fraud_count = int(df[TARGET_COLUMN].sum())

    if strict_kaggle_check:
        if row_count != KAGGLE_EXPECTED_ROWS or fraud_count != KAGGLE_EXPECTED_FRAUD:
            err_msg = (
                f"\n[DATASET VERIFICATION FAILED] The file at '{path_obj}' does not match the official "
                f"Kaggle Credit Card Fraud Detection dataset.\n"
                f"  Found:    {row_count:,} rows with {fraud_count} fraud cases (synthetic benchmark).\n"
                f"  Expected: {KAGGLE_EXPECTED_ROWS:,} rows with {KAGGLE_EXPECTED_FRAUD} fraud cases.\n"
                "To integrate the genuine Kaggle dataset:\n"
                f"  1. Download 'creditcard.csv' from Kaggle: {KAGGLE_URL}\n"
                f"  2. Replace '{path_obj}' with the downloaded file (size ~150 MB).\n"
                "To continue using the synthetic benchmark for testing, run with: --mode synthetic"
            )
            logger.error(err_msg)
            raise DatasetValidationError(err_msg)

    logger.info(
        "Kaggle verification PASSED: %d rows, %d fraud cases (%.4f%% fraud rate).",
        row_count, fraud_count, (fraud_count / row_count) * 100
    )
    return df


def generate_synthetic_data(
    n_samples: int = 25000,
    output_path: Optional[Union[str, Path]] = "data/raw/creditcard_synthetic.csv",
    random_state: int = 42,
    fraud_ratio: float = 0.0066
) -> pd.DataFrame:
    """
    Generates synthetic benchmark credit card transactions for development and testing.
    This function is strictly for development and CI testing. It is NEVER to be described as real data.
    """
    logger.info(
        "Generating synthetic test dataset (n=%d, fraud_ratio=%.4f%%, random_state=%d)...",
        n_samples, fraud_ratio * 100, random_state
    )

    n_features = 30  # 28 PCA features + Time + Amount
    weights = [1.0 - fraud_ratio, fraud_ratio]

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=4,
        n_clusters_per_class=1,
        weights=weights,
        random_state=random_state
    )

    columns = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
    df = pd.DataFrame(X, columns=columns)

    rng = np.random.RandomState(random_state)
    df["Time"] = np.sort(np.abs(rng.uniform(0, 172800, size=n_samples)))
    amounts = np.exp(rng.normal(loc=3.0, scale=1.5, size=n_samples))
    df["Amount"] = np.round(np.clip(amounts, 0.01, 10000.0), 2)

    final_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    df = df[final_cols]
    df[TARGET_COLUMN] = y.astype(int)

    df = validate_dataset(df, is_training=True)

    if output_path:
        out_path_obj = Path(output_path).resolve()
        out_path_obj.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path_obj, index=False)
        logger.info("Synthetic benchmark dataset saved to: %s", out_path_obj)

    return df


def get_dataset(
    mode: str = "real",
    filepath: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads dataset in accordance with the requested mode.
    Guarantees strict separation between real and synthetic modes.
    """
    if mode == "real":
        target_path = filepath or "data/raw/creditcard.csv"
        df = load_real_dataset(target_path, validate=True, strict_kaggle_check=True)
        metadata = {
            "mode": "real",
            "is_synthetic": False,
            "source": "Kaggle Credit Card Fraud Detection Dataset",
            "url": KAGGLE_URL,
            "filepath": str(Path(target_path).resolve()),
            "rows": len(df),
            "fraud_count": int(df[TARGET_COLUMN].sum())
        }
        return df, metadata

    elif mode == "synthetic":
        target_path = filepath or "data/raw/creditcard_synthetic.csv"
        # In synthetic mode, read if exists or generate
        if os.path.exists(target_path):
            df = pd.read_csv(target_path)
            df = validate_dataset(df, is_training=True)
        else:
            df = generate_synthetic_data(n_samples=25000, output_path=target_path)
            
        metadata = {
            "mode": "synthetic",
            "is_synthetic": True,
            "source": "Synthetic Benchmark Generator (scikit-learn make_classification)",
            "filepath": str(Path(target_path).resolve()),
            "rows": len(df),
            "fraud_count": int(df[TARGET_COLUMN].sum()),
            "disclaimer": "SYNTHETIC BENCHMARK DATASET - NOT REAL FINANCIAL TRANSACTION DATA."
        }
        return df, metadata
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'real' or 'synthetic'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Data Ingestion CLI for Credit Card Fraud Detection")
    parser.add_argument(
        "--mode",
        choices=["real", "synthetic"],
        default="real",
        help="Data loading mode: 'real' requires Kaggle CSV (284,807 rows); 'synthetic' uses mock data."
    )
    parser.add_argument(
        "--data-path",
        default="data/raw/creditcard.csv",
        help="Path to creditcard CSV file."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        df, meta = get_dataset(mode=args.mode, filepath=args.data_path)
        print(f"\n[SUCCESS] Loaded {meta['mode'].upper()} dataset: {meta['rows']} rows, {meta['fraud_count']} fraud cases.")
        if meta["is_synthetic"]:
            print(f"[NOTE] {meta.get('disclaimer')}")
    except (DatasetNotFoundError, DatasetValidationError) as e:
        print(f"\n{e}")
        exit(1)
