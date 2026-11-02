"""
Database CRUD (Create, Read, Update, Delete) operations for Transaction Records.
Includes transaction safety, rollback protection, and boundary validation for date filters.
"""

import json
import logging
from datetime import datetime, time, timezone
from typing import Optional, List, Tuple, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from .models import TransactionRecord

logger = logging.getLogger("fraud_api.crud")


def parse_date_boundary(date_str: str, is_end: bool = False) -> Optional[datetime]:
    """
    Parses an ISO format date or datetime string into a timezone-naive or UTC datetime.
    If only a date (YYYY-MM-DD) is provided, adjusts to start (00:00:00) or end (23:59:59.999999) of day.
    
    Raises:
        ValueError: If date_str cannot be parsed as a valid ISO format date.
    """
    if not date_str:
        return None
    date_str = date_str.strip()
    try:
        # Check if only date is passed (YYYY-MM-DD)
        if len(date_str) == 10:
            parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
            if is_end:
                return datetime.combine(parsed_date.date(), time.max)
            return datetime.combine(parsed_date.date(), time.min)
        # Parse full ISO format (e.g. 2026-11-01T12:00:00)
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        # Strip tzinfo to compare against SQLite naive UTC datetimes
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception as e:
        raise ValueError(
            f"Invalid date format: '{date_str}'. Expected ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)."
        ) from e


def create_transaction_record(
    db: Session,
    transaction_dict: Dict[str, Any],
    prediction_result: Dict[str, Any]
) -> TransactionRecord:
    """
    Persists a transaction payload and its inference result to SQLite with rollback safety.
    """
    record = TransactionRecord(
        transaction_time=float(transaction_dict.get("Time", 0.0)),
        amount=float(transaction_dict.get("Amount", 0.0)),
        fraud_probability=float(prediction_result.get("probability", 0.0)),
        prediction=int(prediction_result.get("prediction", 0)),
        is_fraud=bool(prediction_result.get("is_fraud", False)),
        threshold_used=float(prediction_result.get("threshold_used", 0.50)),
        features_json=json.dumps(transaction_dict)
    )
    try:
        db.add(record)
        db.commit()
        db.refresh(record)
        return record
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to persist transaction record to database: {e}")
        raise RuntimeError(f"Database error while saving transaction record: {str(e)}") from e


def get_transaction_by_id(db: Session, transaction_id: int) -> Optional[TransactionRecord]:
    """
    Retrieves a single transaction by its primary key ID.
    """
    return db.query(TransactionRecord).filter(TransactionRecord.id == transaction_id).first()


def get_transactions(
    db: Session,
    skip: int = 0,
    limit: int = 10,
    is_fraud: Optional[bool] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Tuple[List[TransactionRecord], int]:
    """
    Retrieves a paginated list of transactions with optional filtering by:
    - is_fraud (True for fraud, False for legitimate)
    - start_date (ISO date/datetime)
    - end_date (ISO date/datetime)
    
    Validates that start_date is not after end_date.
    Returns (records, total_matching_records).
    """
    query = db.query(TransactionRecord)

    if is_fraud is not None:
        query = query.filter(TransactionRecord.is_fraud == is_fraud)

    start_dt = None
    end_dt = None

    if start_date:
        start_dt = parse_date_boundary(start_date, is_end=False)
    if end_date:
        end_dt = parse_date_boundary(end_date, is_end=True)

    if start_dt and end_dt and start_dt > end_dt:
        raise ValueError(f"'start_date' ({start_date}) cannot be later than 'end_date' ({end_date}).")

    if start_dt:
        query = query.filter(TransactionRecord.timestamp >= start_dt)
    if end_dt:
        query = query.filter(TransactionRecord.timestamp <= end_dt)

    total_count = query.count()
    records = (
        query.order_by(desc(TransactionRecord.timestamp), desc(TransactionRecord.id))
        .offset(skip)
        .limit(limit)
        .all()
    )

    return records, total_count


def get_transaction_stats(db: Session) -> Dict[str, Any]:
    """
    Calculates summary KPIs over all stored transactions:
    - total_transactions
    - detected_frauds
    - legitimate_transactions
    - fraud_percentage
    """
    total = db.query(func.count(TransactionRecord.id)).scalar() or 0
    frauds = db.query(func.count(TransactionRecord.id)).filter(TransactionRecord.is_fraud == True).scalar() or 0
    legit = total - frauds
    fraud_rate = round((frauds / total * 100), 2) if total > 0 else 0.0

    return {
        "total_transactions": total,
        "detected_frauds": frauds,
        "legitimate_transactions": legit,
        "fraud_percentage": fraud_rate
    }
