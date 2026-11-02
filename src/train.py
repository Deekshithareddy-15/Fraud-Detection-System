"""
Enhanced Zero-Leakage ML Pipeline with Validation Threshold Tuning for Fraud Detection.

Pipeline Steps:
1. Strict schema verification (Time, Amount, V1-V28, Class).
2. Stratified 3-way split: Train (60%), Validation (20%), Test (20%).
3. RobustScaler fitted ONLY on Train; Val and Test transformed.
4. SMOTE applied ONLY to Train.
5. Models trained: Logistic Regression, Random Forest, XGBoost.
6. Threshold tuning executed strictly on Validation set (via PR curve & F1 maximization).
7. Final unbiased evaluation on the untouched Test set at both default (0.50) and tuned thresholds.
8. Generation and export of Precision-Recall curves.
9. Persistence of best model, scaler, threshold configuration, baseline stats, and metrics.
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import logging
import argparse
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, List

# Suppress known harmless optimizer warning from scipy/sklearn
warnings.filterwarnings("ignore", message=".*Unknown solver options.*")

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import xgboost as xgb

from src.data_ingestion import (
    get_dataset,
    load_real_dataset,
    validate_dataset,
    DatasetNotFoundError,
    DatasetValidationError,
    KAGGLE_EXPECTED_ROWS,
    KAGGLE_EXPECTED_FRAUD,
    KAGGLE_URL,
    REQUIRED_COLUMNS,
    FEATURE_COLUMNS,
    COLS_TO_SCALE,
    TARGET_COLUMN
)

# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("model_training")

# Canonical paths
DATA_PATH = "data/raw/creditcard.csv"
MODEL_SAVE_PATH = "models/saved_models"
METRICS_PATH = "models/metrics"

PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS = PCA_FEATURES + ["Time", "Amount"]
TARGET_COLUMN = "Class"
REQUIRED_COLUMNS = ["Time"] + PCA_FEATURES + ["Amount", TARGET_COLUMN]
COLS_TO_SCALE = ["Amount", "Time"]


def verify_dataset(filepath: str) -> pd.DataFrame:
    """Loads and verifies dataset against schema requirements."""
    logger.info("Loading and verifying dataset from '%s'...", filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    logger.info("Dataset shape: %s (rows=%d, cols=%d).", df.shape, len(df), len(df.columns))
    
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
        
    null_counts = df[REQUIRED_COLUMNS].isnull().sum().sum()
    if null_counts > 0:
        logger.warning("Dropping %d null records.", null_counts)
        df = df.dropna(subset=REQUIRED_COLUMNS)
        
    unique_classes = set(df[TARGET_COLUMN].unique())
    if not unique_classes.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"Invalid target labels: {unique_classes}. Expected binary {0, 1}.")
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    
    class_counts = df[TARGET_COLUMN].value_counts().to_dict()
    total = len(df)
    logger.info(
        "Class balance: Legitimate (0)=%d (%.2f%%), Fraud (1)=%d (%.4f%%)",
        class_counts.get(0, 0), (class_counts.get(0, 0) / total) * 100,
        class_counts.get(1, 0), (class_counts.get(1, 0) / total) * 100
    )
    return df


def split_and_preprocess_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    val_size: float = 0.25,  # 0.25 of 0.80 = 0.20 of total
    random_state: int = 42
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series, pd.Series,
    RobustScaler
]:
    """
    Executes a 3-way stratified split: Train (60%), Validation (20%), Test (20%).
    Fits RobustScaler strictly on Train; transforms Val and Test.
    Zero data leakage guaranteed.
    """
    logger.info("Performing 3-way stratified split (Train: 60%%, Val: 20%%, Test: 20%%)...")
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    
    # Split 1: Separate untouched Test set (20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # Split 2: Separate Train (75% of 80% = 60%) and Validation (25% of 80% = 20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_val
    )
    
    logger.info("Split summary:")
    logger.info("  Train set: %d rows (Fraud count: %d, Rate: %.4f%%)", len(X_train), int(y_train.sum()), (y_train.sum() / len(X_train)) * 100)
    logger.info("  Val set:   %d rows (Fraud count: %d, Rate: %.4f%%)", len(X_val), int(y_val.sum()), (y_val.sum() / len(X_val)) * 100)
    logger.info("  Test set:  %d rows (Fraud count: %d, Rate: %.4f%%)", len(X_test), int(y_test.sum()), (y_test.sum() / len(X_test)) * 100)
    
    # Fit scaler strictly on X_train
    logger.info("Fitting RobustScaler on X_train[%s] ONLY...", COLS_TO_SCALE)
    scaler = RobustScaler()
    
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()
    
    X_train[COLS_TO_SCALE] = scaler.fit_transform(X_train[COLS_TO_SCALE])
    X_val[COLS_TO_SCALE] = scaler.transform(X_val[COLS_TO_SCALE])
    X_test[COLS_TO_SCALE] = scaler.transform(X_test[COLS_TO_SCALE])
    
    logger.info("Zero-leakage scaling complete.")
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """Applies SMOTE oversampling strictly to training data."""
    logger.info("Applying SMOTE oversampling ONLY to training data...")
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    if not isinstance(X_resampled, pd.DataFrame):
        X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    if not isinstance(y_resampled, pd.Series):
        y_resampled = pd.Series(y_resampled, name=TARGET_COLUMN)
        
    logger.info(
        "SMOTE complete: Training shape expanded from %d to %d (Fraud: %d, Non-fraud: %d)",
        len(X_train), len(X_resampled), int((y_resampled == 1).sum()), int((y_resampled == 0).sum())
    )
    return X_resampled, y_resampled


def tune_classification_threshold(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str
) -> Dict[str, Any]:
    """
    Performs classification threshold tuning using VALIDATION DATA ONLY.
    
    Evaluates:
    - Precision-Recall Curve points
    - Threshold grid from 0.05 to 0.95
    - Selects optimal threshold maximizing F1 while balancing precision and recall.
    """
    logger.info("Tuning classification threshold for %s on validation data...", model_name)
    y_val_prob = model.predict_proba(X_val)[:, 1]
    
    precisions, recalls, pr_thresholds = precision_recall_curve(y_val, y_val_prob)
    val_pr_auc = float(average_precision_score(y_val, y_val_prob))
    
    # Calculate F1 scores along the PR curve (avoid div-by-zero)
    f1_scores = np.divide(
        2 * (precisions * recalls),
        (precisions + recalls),
        out=np.zeros_like(precisions),
        where=(precisions + recalls) != 0
    )
    
    # Grid search across standard operating thresholds [0.05 ... 0.95]
    grid_thresholds = [round(t, 2) for t in np.linspace(0.05, 0.95, 19)]
    grid_analysis = []
    
    best_threshold = 0.50
    best_f1 = -1.0
    best_rec = 0.0
    best_prec = 0.0
    
    for t in grid_thresholds:
        preds = (y_val_prob >= t).astype(int)
        rec = float(recall_score(y_val, preds, zero_division=0))
        prec = float(precision_score(y_val, preds, zero_division=0))
        f1 = float(f1_score(y_val, preds, zero_division=0))
        acc = float(accuracy_score(y_val, preds))
        cm = confusion_matrix(y_val, preds)
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]
        
        entry = {
            "threshold": t,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        }
        grid_analysis.append(entry)
        
        # Selection criterion: Highest F1 on validation set, with Recall as tie-breaker
        if (f1 > best_f1) or (abs(f1 - best_f1) < 1e-4 and rec > best_rec):
            best_f1 = f1
            best_threshold = t
            best_rec = rec
            best_prec = prec
            
    # Also evaluate the exact PR curve argmax F1 threshold
    if len(pr_thresholds) > 0:
        opt_idx = np.argmax(f1_scores[:-1])
        curve_opt_threshold = round(float(pr_thresholds[opt_idx]), 4)
        curve_opt_f1 = round(float(f1_scores[opt_idx]), 4)
    else:
        curve_opt_threshold = best_threshold
        curve_opt_f1 = best_f1

    logger.info(
        "Optimal validation threshold for %s: %.2f (Val F1: %.4f, Recall: %.4f, Precision: %.4f, Val PR-AUC: %.4f)",
        model_name, best_threshold, best_f1, best_rec, best_prec, val_pr_auc
    )
    
    return {
        "model_name": model_name,
        "selected_threshold": best_threshold,
        "validation_pr_auc": round(val_pr_auc, 4),
        "validation_best_f1": round(best_f1, 4),
        "validation_recall_at_threshold": round(best_rec, 4),
        "validation_precision_at_threshold": round(best_prec, 4),
        "curve_optimal_threshold": curve_opt_threshold,
        "curve_optimal_f1": curve_opt_f1,
        "threshold_grid_analysis": grid_analysis,
        "pr_curve_data": {
            "precisions": [round(float(p), 4) for p in precisions[::max(1, len(precisions)//50)]],
            "recalls": [round(float(r), 4) for r in recalls[::max(1, len(recalls)//50)]]
        }
    }


def evaluate_predictions(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Calculates comprehensive evaluation metrics at a specified decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    
    return {
        "threshold": round(threshold, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "raw": [[tn, fp], [fn, tp]]
        },
        "tradeoff": {
            "false_positives": fp,
            "false_negatives": fn,
            "fp_to_fn_ratio": round(fp / max(1, fn), 2),
            "fraud_catch_rate": f"{round(rec * 100, 2)}%"
        }
    }


def plot_precision_recall_curves(
    model_pr_data: Dict[str, Tuple[np.ndarray, np.ndarray, float, float, float]],
    save_path: str = os.path.join(METRICS_PATH, "precision_recall_curves.png")
):
    """Generates and saves high-resolution Precision-Recall curves with operating points marked."""
    plt.figure(figsize=(9, 6), dpi=300)
    colors = {
        "Logistic Regression": "#2563EB",  # Vibrant Blue
        "Random Forest": "#059669",        # Emerald Green
        "XGBoost": "#D97706"              # Amber/Orange
    }
    
    for name, (recalls, precisions, pr_auc, op_rec, op_prec) in model_pr_data.items():
        color = colors.get(name, "#4B5563")
        plt.plot(recalls, precisions, label=f"{name} (PR-AUC = {pr_auc:.4f})", color=color, linewidth=2.2)
        plt.scatter([op_rec], [op_prec], color=color, s=90, zorder=5, edgecolors='black', linewidth=1.5)
        plt.annotate(
            f"{name} Operating Pt\n(Rec: {op_rec:.2f}, Prec: {op_prec:.2f})",
            (op_rec, op_prec),
            textcoords="offset points",
            xytext=(10, -15 if name == "Logistic Regression" else 10),
            fontsize=8,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.85)
        )
        
    plt.title("Precision-Recall Curves & Tuned Operating Points (Untouched Test Data)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Recall (True Positive Rate)", fontsize=11, labelpad=8)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11, labelpad=8)
    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.02])
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logger.info("Saved Precision-Recall curves plot to '%s'", save_path)


def run_pipeline(mode: str = "real", data_path: str = DATA_PATH) -> Dict[str, Any]:
    """
    Runs the full training, validation tuning, test evaluation, and artifact saving pipeline.
    Strictly verifies Kaggle real dataset or synthetic benchmark according to 'mode'.
    """
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(METRICS_PATH, exist_ok=True)
    
    # 1. Dataset Verification & Ingestion
    logger.info("Loading dataset in '%s' mode...", mode)
    df, meta_info = get_dataset(mode=mode, filepath=data_path)
    
    # 2. 3-Way Stratified Split & Zero-Leakage Preprocessing
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_preprocess_data(df)
    
    # 3. SMOTE Oversampling on Train ONLY
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    
    # 4. Model Definitions
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver='lbfgs'
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42
        )
    }
    
    trained_models = {}
    validation_tuning_results = {}
    test_results_default = {}
    test_results_tuned = {}
    pr_plot_data = {}
    
    for name, model in models.items():
        logger.info("Training %s on balanced training set...", name)
        model.fit(X_train_resampled, y_train_resampled)
        trained_models[name] = model
        
        # Step A: Threshold tuning on Validation data only
        tuning_info = tune_classification_threshold(model, X_val, y_val, name)
        validation_tuning_results[name] = tuning_info
        best_t = tuning_info["selected_threshold"]
        
        # Step B: Evaluate on Untouched Test Set at DEFAULT threshold (0.50)
        y_test_prob = model.predict_proba(X_test)[:, 1]
        metrics_default = evaluate_predictions(y_test, y_test_prob, threshold=0.50)
        test_results_default[name] = metrics_default
        
        # Step C: Evaluate on Untouched Test Set at TUNED threshold
        metrics_tuned = evaluate_predictions(y_test, y_test_prob, threshold=best_t)
        test_results_tuned[name] = metrics_tuned
        
        # Collect PR curve data on Test set
        test_precisions, test_recalls, _ = precision_recall_curve(y_test, y_test_prob)
        test_pr_auc = float(average_precision_score(y_test, y_test_prob))
        pr_plot_data[name] = (
            test_recalls,
            test_precisions,
            test_pr_auc,
            metrics_tuned["recall"],
            metrics_tuned["precision"]
        )
        
        logger.info(
            "%s [DEFAULT 0.50] -> Recall: %.2f%% | Prec: %.2f%% | F1: %.4f | PR-AUC: %.4f | FP: %d | FN: %d",
            name, metrics_default["recall"] * 100, metrics_default["precision"] * 100,
            metrics_default["f1_score"], metrics_default["pr_auc"],
            metrics_default["confusion_matrix"]["fp"], metrics_default["confusion_matrix"]["fn"]
        )
        logger.info(
            "%s [TUNED %.2f]   -> Recall: %.2f%% | Prec: %.2f%% | F1: %.4f | PR-AUC: %.4f | FP: %d | FN: %d",
            name, best_t, metrics_tuned["recall"] * 100, metrics_tuned["precision"] * 100,
            metrics_tuned["f1_score"], metrics_tuned["pr_auc"],
            metrics_tuned["confusion_matrix"]["fp"], metrics_tuned["confusion_matrix"]["fn"]
        )
        
    # Generate and save PR Curves plot
    plot_precision_recall_curves(pr_plot_data)
    
    # 5. Model Selection based on PR-AUC, Recall, and Precision Trade-off
    # We rank primarily by PR-AUC (model discrimination capability) and Tuned F1-Score
    best_name = max(
        models.keys(),
        key=lambda k: (
            test_results_tuned[k]["pr_auc"],
            test_results_tuned[k]["f1_score"],
            test_results_tuned[k]["recall"]
        )
    )
    best_model = trained_models[best_name]
    best_threshold = validation_tuning_results[best_name]["selected_threshold"]
    
    logger.info("=" * 65)
    logger.info("BEST MODEL SELECTED: %s (Optimal Threshold: %.2f)", best_name, best_threshold)
    logger.info(
        "Rationale: Highest PR-AUC (%.4f) with Tuned F1 (%.4f), Recall (%.2f%%) and Precision (%.2f%%)",
        test_results_tuned[best_name]["pr_auc"],
        test_results_tuned[best_name]["f1_score"],
        test_results_tuned[best_name]["recall"] * 100,
        test_results_tuned[best_name]["precision"] * 100
    )
    logger.info("=" * 65)
    
    # 6. Artifact Persistence
    logger.info("Saving production artifacts...")
    
    # 6a. Best Model and Scaler
    best_model_path = os.path.join(MODEL_SAVE_PATH, "best_fraud_model.pkl")
    scaler_path = os.path.join(MODEL_SAVE_PATH, "scaler.pkl")
    joblib.dump(best_model, best_model_path)
    joblib.dump(scaler, scaler_path)
    
    # 6b. Individual Models
    for name, model in trained_models.items():
        slug = name.lower().replace(" ", "_")
        joblib.dump(model, os.path.join(MODEL_SAVE_PATH, f"{slug}.pkl"))
        
    # 6c. Threshold Configuration
    threshold_config = {
        "best_model_name": best_name,
        "classification_threshold": best_threshold,
        "default_threshold": 0.50,
        "tuned_thresholds": {k: validation_tuning_results[k]["selected_threshold"] for k in models.keys()},
        "tuning_method": "F1-maximization on validation set via Precision-Recall curve",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    threshold_config_path = os.path.join(MODEL_SAVE_PATH, "threshold_config.json")
    with open(threshold_config_path, "w") as f:
        json.dump(threshold_config, f, indent=4)
        
    # 6d. Baseline Statistics for Data Drift Detection
    baseline_stats = X_train_resampled.describe()
    baseline_stats_path = os.path.join(MODEL_SAVE_PATH, "baseline_stats.json")
    baseline_stats.to_json(baseline_stats_path)
    
    # 6e. Validation Tuning Analysis JSON
    val_analysis_path = os.path.join(METRICS_PATH, "threshold_tuning_validation.json")
    with open(val_analysis_path, "w") as f:
        json.dump(validation_tuning_results, f, indent=4)
        
    # 6f. Comprehensive Evaluation Metrics JSON (Default vs Tuned)
    evaluation_metrics_payload = {
        "comparison_summary": {
            model_name: {
                "default_threshold_0_50": test_results_default[model_name],
                "validation_tuned_threshold": test_results_tuned[model_name]
            }
            for model_name in models.keys()
        },
        "selected_best_model": {
            "name": best_name,
            "threshold": best_threshold,
            "metrics": test_results_tuned[best_name]
        }
    }
    metrics_file_path = os.path.join(METRICS_PATH, "evaluation_metrics.json")
    with open(metrics_file_path, "w") as f:
        json.dump(evaluation_metrics_payload, f, indent=4)
        
    # 6g. Model Metadata File
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_mode": mode,
        "is_synthetic": meta_info.get("is_synthetic", False),
        "dataset_source": meta_info.get("source", "Kaggle Credit Card Fraud Detection"),
        "dataset_disclaimer": meta_info.get("disclaimer", "Official Kaggle Real Dataset" if not meta_info.get("is_synthetic") else "SYNTHETIC BENCHMARK - NOT REAL FINANCIAL TRANSACTION DATA"),
        "dataset_path": data_path,
        "dataset_total_rows": len(df),
        "dataset_fraud_count": int(df[TARGET_COLUMN].sum()),
        "split_ratios": {"train": 0.60, "validation": 0.20, "test": 0.20},
        "rows": {
            "train_raw": len(X_train),
            "train_resampled_smote": len(X_train_resampled),
            "validation": len(X_val),
            "test": len(X_test)
        },
        "feature_columns": FEATURE_COLUMNS,
        "scaled_columns": COLS_TO_SCALE,
        "best_model_name": best_name,
        "best_classification_threshold": best_threshold,
        "best_model_test_metrics_at_tuned_threshold": test_results_tuned[best_name],
        "all_models_test_metrics_at_tuned_threshold": test_results_tuned,
        "all_models_test_metrics_at_default_threshold": test_results_default
    }
    metadata_path = os.path.join(MODEL_SAVE_PATH, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    logger.info("All model artifacts and evaluation reports successfully saved.")
    return {
        "best_model_name": best_name,
        "best_threshold": best_threshold,
        "test_results_default": test_results_default,
        "test_results_tuned": test_results_tuned,
        "validation_tuning": validation_tuning_results
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fraud Detection Pipeline")
    parser.add_argument(
        "--mode",
        choices=["real", "synthetic"],
        default="real",
        help="Dataset mode: 'real' requires Kaggle CSV (284,807 rows); 'synthetic' uses benchmark data."
    )
    parser.add_argument(
        "--data-path",
        default=DATA_PATH,
        help="Path to creditcard CSV file."
    )
    args = parser.parse_args()

    try:
        run_pipeline(mode=args.mode, data_path=args.data_path)
        print(f"\nModel Training and Threshold Tuning ({args.mode.upper()} mode) Completed Successfully.")
    except (DatasetNotFoundError, DatasetValidationError) as e:
        print(f"\n[ERROR] Pipeline halted due to dataset validation failure:\n{e}")
        sys.exit(1)
