"""
Pydantic schemas for request validation, boundary checks, and response serialization.
"""

from datetime import datetime
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class Transaction(BaseModel):
    """
    Input transaction feature vector containing Time, Amount, and 28 PCA features (V1-V28).
    Includes boundary validation to prevent infinite, NaN, or out-of-range numerical payloads.
    """
    Time: float = Field(
        ...,
        ge=0.0,
        le=1000000.0,
        description="Seconds elapsed between each transaction and the first transaction in the dataset"
    )
    Amount: float = Field(
        ...,
        ge=0.0,
        le=500000.0,
        description="Transaction Amount in currency units (non-negative, realistic transaction maximum)"
    )
    V1: float = Field(..., ge=-100.0, le=100.0, description="PCA transformed feature V1")
    V2: float = Field(..., ge=-100.0, le=100.0, description="PCA transformed feature V2")
    V3: float = Field(..., ge=-100.0, le=100.0, description="PCA transformed feature V3")
    V4: float = Field(..., ge=-100.0, le=100.0, description="PCA transformed feature V4")
    V5: float = Field(..., ge=-100.0, le=100.0, description="PCA transformed feature V5")
    V6: float = Field(..., ge=-100.0, le=100.0, description="PCA transformed feature V6")
    V7: float = Field(..., ge=-100.0, le=100.0, description="PCA transformed feature V7")
    V8: float = Field(..., ge=-100.0, le=100.0, description="PCA transformed feature V8")
    V9: float = Field(..., ge=-100.0, le=100.0, description="PCA transformed feature V9")
    V10: float = Field(..., ge=-100.0, le=100.0, description="PCA transformed feature V10")
    V11: float = Field(..., ge=-100.0, le=100.0, description="PCA transformed feature V11")
    V12: float = Field(..., ge=-100.0, le=100.0, description="PCA transformed feature V12")
    V13: float = Field(..., ge=-100.0, le=100.0, description="PCA transformed feature V13")
    V14: float = Field(..., ge=-100.0, le=100.0, description="PCA transformed feature V14")
    V15: float = Field(..., ge=-100.0, le=100.0, description="PCA transformed feature V15")
    V16: float = Field(..., ge=-100.0, le=100.0, description="PCA transformed feature V16")
    V17: float = Field(..., ge=-100.0, le=100.0, description="PCA transformed feature V17")
    V18: float = Field(..., ge=-100.0, le=100.0, description="PCA transformed feature V18")
    V19: float = Field(..., ge=-100.0, le=100.0, description="PCA transformed feature V19")
    V20: float = Field(..., ge=-100.0, le=100.0, description="PCA transformed feature V20")
    V21: float = Field(..., ge=-100.0, le=100.0, description="PCA transformed feature V21")
    V22: float = Field(..., ge=-100.0, le=100.0, description="PCA transformed feature V22")
    V23: float = Field(..., ge=-100.0, le=100.0, description="PCA transformed feature V23")
    V24: float = Field(..., ge=-100.0, le=100.0, description="PCA transformed feature V24")
    V25: float = Field(..., ge=-100.0, le=100.0, description="PCA transformed feature V25")
    V26: float = Field(..., ge=-100.0, le=100.0, description="PCA transformed feature V26")
    V27: float = Field(..., ge=-100.0, le=100.0, description="PCA transformed feature V27")
    V28: float = Field(..., ge=-100.0, le=100.0, description="PCA transformed feature V28")

    class Config:
        json_schema_extra = {
            "example": {
                "Time": 0.0,
                "Amount": 100.0,
                "V1": -1.359807,
                "V2": -0.072781,
                "V3": 2.536347,
                "V4": 1.378155,
                "V5": -0.338321,
                "V6": 0.462388,
                "V7": 0.239599,
                "V8": 0.098698,
                "V9": 0.363787,
                "V10": 0.090794,
                "V11": -0.551600,
                "V12": -0.617801,
                "V13": -0.991390,
                "V14": -0.311169,
                "V15": 1.468177,
                "V16": -0.470401,
                "V17": 0.207971,
                "V18": 0.025791,
                "V19": 0.403993,
                "V20": 0.251412,
                "V21": -0.018307,
                "V22": 0.277838,
                "V23": -0.110474,
                "V24": 0.066928,
                "V25": 0.128539,
                "V26": -0.189115,
                "V27": 0.133558,
                "V28": -0.021053
            }
        }


class PredictionResponse(BaseModel):
    """Inference response model returned by /predict endpoint."""
    prediction: int = Field(..., description="0 for Legitimate, 1 for Fraud")
    probability: float = Field(..., description="Model estimated fraud probability [0.0 - 1.0]")
    is_fraud: bool = Field(..., description="True if probability >= threshold_used, else False")
    threshold_used: float = Field(..., description="Decision threshold applied to the prediction")
    transaction_id: Optional[int] = Field(None, description="Database record ID of the persisted transaction")


class TransactionListItem(BaseModel):
    """Summary item returned in transaction history lists."""
    id: int
    timestamp: datetime
    transaction_time: float
    amount: float
    fraud_probability: float
    prediction: int
    is_fraud: bool
    threshold_used: float

    class Config:
        from_attributes = True


class TransactionDetailResponse(BaseModel):
    """Detailed view of an individual transaction including all input features."""
    id: int
    timestamp: datetime
    transaction_time: float
    amount: float
    fraud_probability: float
    prediction: int
    is_fraud: bool
    threshold_used: float
    features: Dict[str, float]

    class Config:
        from_attributes = True


class PaginatedTransactionsResponse(BaseModel):
    """Paginated collection of transaction history records."""
    items: List[TransactionListItem]
    total: int
    page: int
    page_size: int
    total_pages: int


class TransactionStatsResponse(BaseModel):
    """Aggregate KPI statistics across all recorded transactions."""
    total_transactions: int
    detected_frauds: int
    legitimate_transactions: int
    fraud_percentage: float


class HealthResponse(BaseModel):
    """Comprehensive health check status schema."""
    status: str
    timestamp: str
    model_loaded: bool
    model_name: str
    scaler_loaded: bool
    database_connected: bool
    classification_threshold: float
    dataset_mode: str
