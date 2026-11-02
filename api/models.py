"""
SQLAlchemy ORM models for Transaction History and Predictions.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, Float, Boolean, DateTime, Text
from .database import Base


class TransactionRecord(Base):
    """
    Persisted transaction record storing input features, model prediction,
    fraud probability, and classification threshold for auditing and analytics.
    """
    __tablename__ = "transaction_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        index=True,
        nullable=False,
        doc="UTC timestamp when the prediction was evaluated"
    )
    transaction_time = Column(Float, nullable=False, doc="Seconds elapsed feature (Time)")
    amount = Column(Float, nullable=False, doc="Transaction amount")
    fraud_probability = Column(Float, nullable=False, doc="Probability score of fraud [0.0 - 1.0]")
    prediction = Column(Integer, nullable=False, doc="Binary prediction label (0=Legitimate, 1=Fraud)")
    is_fraud = Column(Boolean, nullable=False, index=True, doc="Flag indicating if probability >= threshold")
    threshold_used = Column(Float, nullable=False, doc="Classification decision threshold applied")
    features_json = Column(Text, nullable=False, doc="JSON-serialized input features (V1-V28, Time, Amount)")

    def __repr__(self) -> str:
        return (
            f"<TransactionRecord(id={self.id}, amount={self.amount}, "
            f"probability={self.fraud_probability:.4f}, is_fraud={self.is_fraud})>"
        )
