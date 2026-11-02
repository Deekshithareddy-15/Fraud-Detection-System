
# Model Card: Fraud Detection System

## Model Details
- **Developer**: Antigravity AI
- **Model Date**: February 2026
- **Model Version**: 1.0.0
- **Model Type**: Random Forest / XGBoost Classifier
- **License**: MIT

## Intended Use
- **Primary Use Case**: Real-time detection of fraudulent credit card transactions.
- **Intended Users**: Financial institutions, payment processors.
- **Out of Scope**: Not for use in critical medical or safety applications.

## Training Data
- **Dataset**: Credit Card Fraud Detection Dataset (Kaggle) / Synthetic Proxy
- **Preprocessing**:
  - Outlier removal (RobustScaler) on `Amount` and `Time`.
  - Class Imbalance Handling: SMOTE (Synthetic Minority Over-sampling Technique) on training set.
  - Test Set: Maintain original imbalance for realistic evaluation.

## Performance Metrics
- **Evaluation Set**: 20% Hold-out Test Set.
- **Metrics**:
  - Precision-Recall AUC (Primary Metric mostly due to imbalance)
  - F1-Score (Macro/Weighted)
  - ROC-AUC
  - Confusion Matrix (False Positives vs False Negatives Trade-off)

## Limitations & Bias
- **Synthetic Data**: If trained on synthetic proxy data, real-world performance may vary significantly.
- **Drift**: Model assumes transaction patterns remain stable. Concept drift (e.g., new fraud tactics) requires retraining.
- **Bias**: If training data is biased against certain transaction types/regions, the model will inherit this bias.

## How to Use
Load using `joblib`:
```python
import joblib
import pandas as pd

model = joblib.load("models/saved_models/best_fraud_model.pkl")
scaler = joblib.load("models/saved_models/scaler.pkl")

# Preprocess input (scale Amount/Time)
# Predict
```
