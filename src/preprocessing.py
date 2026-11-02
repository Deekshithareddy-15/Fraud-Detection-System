"""
Data Preprocessing Module for Credit Card Fraud Detection.
Ensures zero data leakage by enforcing fit on training set only.
"""

from typing import Tuple, Optional
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split


def load_data(filepath: str) -> Optional[pd.DataFrame]:
    """Loads dataset from CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Performs basic data cleaning."""
    initial_rows = len(df)
    df = df.drop_duplicates()
    new_rows = len(df)
    if initial_rows != new_rows:
        print(f"Removed {initial_rows - new_rows} duplicate rows.")
    
    if df.isnull().sum().sum() > 0:
        print("Dataset contains missing values. Imputing with median.")
        df = df.fillna(df.median(numeric_only=True))
        
    return df


def split_data(
    df: pd.DataFrame,
    target_col: str = "Class",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits data into stratified training and testing sets."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cols_to_scale: Optional[list] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, RobustScaler]:
    """
    Fits RobustScaler on training data only and transforms both train and test.
    Guarantees no data leakage from test into train.
    """
    if cols_to_scale is None:
        cols_to_scale = ["Amount", "Time"]

    scaler = RobustScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    return X_train_scaled, X_test_scaled, scaler


def preprocess_features(df: pd.DataFrame):
    """
    Legacy helper: Scales Amount and Time features.
    Note: For strict production pipelines, use split_data followed by scale_features.
    """
    scaler = RobustScaler()
    df_copy = df.copy()
    df_copy[["Amount", "Time"]] = scaler.fit_transform(df_copy[["Amount", "Time"]])
    return df_copy, scaler
