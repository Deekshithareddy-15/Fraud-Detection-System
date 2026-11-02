"""
Unit tests for Fraud Detection Machine Learning Pipeline.
Tests data leakage safeguards, 3-way split, threshold tuning, PR curves, and model persistence.
"""

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import pytest

# Suppress known harmless optimizer warning
warnings.filterwarnings("ignore", message=".*Unknown solver options.*")
from sklearn.linear_model import LogisticRegression
from src.train import (
    verify_dataset,
    split_and_preprocess_data,
    apply_smote,
    tune_classification_threshold,
    evaluate_predictions,
    FEATURE_COLUMNS,
    REQUIRED_COLUMNS,
    MODEL_SAVE_PATH,
    METRICS_PATH,
)


@pytest.fixture
def sample_dataset():
    """Generates a small valid test dataset with the required creditcard schema."""
    np.random.seed(42)
    n = 200
    data = {}
    data["Time"] = np.random.uniform(0, 1000, size=n)
    for i in range(1, 29):
        data[f"V{i}"] = np.random.randn(n)
    data["Amount"] = np.random.exponential(50, size=n)
    # Binary class with imbalance: 180 legit, 20 fraud (10%)
    data["Class"] = np.array([0] * 180 + [1] * 20)
    return pd.DataFrame(data)


def test_verify_dataset_valid(sample_dataset, tmp_path):
    """Verifies that schema check passes on a valid dataset."""
    csv_path = tmp_path / "valid_creditcard.csv"
    sample_dataset.to_csv(csv_path, index=False)
    
    df = verify_dataset(str(csv_path))
    assert len(df) == 200
    assert set(REQUIRED_COLUMNS).issubset(set(df.columns))
    assert set(df["Class"].unique()) == {0, 1}


def test_verify_dataset_missing_column(sample_dataset, tmp_path):
    """Verifies that schema check raises ValueError when a required column is missing."""
    csv_path = tmp_path / "invalid_creditcard.csv"
    invalid_df = sample_dataset.drop(columns=["V10"])
    invalid_df.to_csv(csv_path, index=False)
    
    with pytest.raises(ValueError, match="missing required columns"):
        verify_dataset(str(csv_path))


def test_zero_data_leakage_3way_split(sample_dataset):
    """
    Verifies that:
    1. 3-way stratified split creates Train (60%), Val (20%), Test (20%).
    2. Scaler is fit ONLY on train data (val and test are only transformed).
    3. Neither val nor test distribution leaks into scaler statistics.
    """
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_preprocess_data(
        sample_dataset, test_size=0.20, val_size=0.25, random_state=42
    )
    
    assert len(X_train) == 120  # 60%
    assert len(X_val) == 40     # 20%
    assert len(X_test) == 40    # 20%
    
    # Stratification preserved: ~10% fraud across all splits
    assert y_train.sum() == 12
    assert y_val.sum() == 4
    assert y_test.sum() == 4
    
    # Verify scaler center corresponds strictly to training median
    train_amount_median = np.median(sample_dataset.loc[X_train.index, "Amount"])
    assert np.isclose(scaler.center_[0], train_amount_median, rtol=1e-4)


def test_smote_isolation_train_only(sample_dataset):
    """Verifies that SMOTE is applied strictly to training data and never touches val or test."""
    X_train, X_val, X_test, y_train, y_val, y_test, _ = split_and_preprocess_data(
        sample_dataset, test_size=0.20, val_size=0.25, random_state=42
    )
    
    initial_val_len = len(X_val)
    initial_test_len = len(X_test)
    
    X_resampled, y_resampled = apply_smote(X_train, y_train, random_state=42)
    
    # Training data should be balanced
    assert (y_resampled == 1).sum() == (y_resampled == 0).sum()
    
    # Val and Test MUST remain untouched
    assert len(X_val) == initial_val_len
    assert len(X_test) == initial_test_len


def test_threshold_tuning_validation_only(sample_dataset):
    """Verifies that classification threshold tuning operates on validation data and selects valid threshold."""
    X_train, X_val, X_test, y_train, y_val, y_test, _ = split_and_preprocess_data(
        sample_dataset, test_size=0.20, val_size=0.25, random_state=42
    )
    X_resampled, y_resampled = apply_smote(X_train, y_train, random_state=42)
    
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_resampled, y_resampled)
    
    tuning_info = tune_classification_threshold(model, X_val, y_val, "Logistic Regression")
    
    assert "selected_threshold" in tuning_info
    assert 0.05 <= tuning_info["selected_threshold"] <= 0.95
    assert "threshold_grid_analysis" in tuning_info
    assert len(tuning_info["threshold_grid_analysis"]) == 19
    assert tuning_info["validation_best_f1"] >= 0.0


def test_evaluate_predictions_metrics():
    """Verifies that evaluate_predictions produces all required metrics and valid confusion matrix."""
    y_true = pd.Series([0, 0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.4, 0.7, 0.8, 0.9])
    
    metrics = evaluate_predictions(y_true, y_prob, threshold=0.50)
    
    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc", "confusion_matrix", "tradeoff"]:
        assert key in metrics
    
    # Expected at 0.50 threshold: pred = [0, 0, 0, 1, 1, 1]
    # TP: 2, FP: 1, TN: 3, FN: 0
    cm = metrics["confusion_matrix"]
    assert cm["tp"] == 2
    assert cm["fp"] == 1
    assert cm["tn"] == 3
    assert cm["fn"] == 0


def test_saved_artifacts_integrity():
    """Verifies that all required model artifacts, threshold config, and plots exist and load properly."""
    best_model_file = os.path.join(MODEL_SAVE_PATH, "best_fraud_model.pkl")
    scaler_file = os.path.join(MODEL_SAVE_PATH, "scaler.pkl")
    threshold_file = os.path.join(MODEL_SAVE_PATH, "threshold_config.json")
    metadata_file = os.path.join(MODEL_SAVE_PATH, "metadata.json")
    baseline_stats_file = os.path.join(MODEL_SAVE_PATH, "baseline_stats.json")
    metrics_file = os.path.join(METRICS_PATH, "evaluation_metrics.json")
    val_analysis_file = os.path.join(METRICS_PATH, "threshold_tuning_validation.json")
    pr_curve_plot = os.path.join(METRICS_PATH, "precision_recall_curves.png")
    
    for fpath in [best_model_file, scaler_file, threshold_file, metadata_file, baseline_stats_file, metrics_file, val_analysis_file, pr_curve_plot]:
        assert os.path.exists(fpath), f"Artifact missing: {fpath}"
    
    # Verify threshold config
    with open(threshold_file, "r") as f:
        t_cfg = json.load(f)
    assert "classification_threshold" in t_cfg
    assert 0.0 < t_cfg["classification_threshold"] < 1.0
    
    # Verify model and scaler can perform inference
    model = joblib.load(best_model_file)
    scaler = joblib.load(scaler_file)
    dummy_input = pd.DataFrame([np.zeros(len(FEATURE_COLUMNS))], columns=FEATURE_COLUMNS)
    dummy_input[["Amount", "Time"]] = scaler.transform(dummy_input[["Amount", "Time"]])
    prob = model.predict_proba(dummy_input)
    assert 0.0 <= prob[0][1] <= 1.0


def test_real_mode_missing_file_raises_not_found(tmp_path):
    """Verifies that real dataset mode raises DatasetNotFoundError when CSV is absent."""
    from src.data_ingestion import load_real_dataset, DatasetNotFoundError
    missing_file = tmp_path / "does_not_exist.csv"
    with pytest.raises(DatasetNotFoundError, match="Kaggle Credit Card Fraud dataset not found"):
        load_real_dataset(str(missing_file))


def test_real_mode_rejects_synthetic_benchmark(tmp_path):
    """Verifies that real dataset mode strictly rejects synthetic data."""
    from src.data_ingestion import load_real_dataset, DatasetValidationError, generate_synthetic_data
    synth_path = tmp_path / "synthetic_test.csv"
    generate_synthetic_data(n_samples=500, output_path=str(synth_path))
    with pytest.raises(DatasetValidationError, match="does not match the official Kaggle"):
        load_real_dataset(str(synth_path), strict_kaggle_check=True)


def test_synthetic_mode_loads_and_flags_synthetic():
    """Verifies that synthetic mode explicitly flags data as synthetic with disclaimers."""
    from src.data_ingestion import get_dataset
    df, meta = get_dataset(mode="synthetic")
    assert len(df) == 25000
    assert meta["is_synthetic"] is True
    assert meta["mode"] == "synthetic"
    assert "NOT REAL FINANCIAL TRANSACTION DATA" in meta["disclaimer"]


def test_run_pipeline_real_mode_halts_without_kaggle(tmp_path):
    """Verifies that run_pipeline in real mode halts when the Kaggle 284k dataset is not present."""
    from src.train import run_pipeline
    from src.data_ingestion import DatasetValidationError, generate_synthetic_data
    synth_path = tmp_path / "synthetic_benchmark.csv"
    generate_synthetic_data(n_samples=500, output_path=str(synth_path))
    with pytest.raises(DatasetValidationError):
        run_pipeline(mode="real", data_path=str(synth_path))
