"""
Unit tests for SQLite Database operations, TransactionRecord model, and CRUD functions.
"""

import pytest
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from api.database import Base
from api.models import TransactionRecord
from api.crud import (
    create_transaction_record,
    get_transaction_by_id,
    get_transactions,
    get_transaction_stats,
    parse_date_boundary,
)


@pytest.fixture
def test_db():
    """Creates a temporary in-memory SQLite database for isolated unit testing."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_create_and_retrieve_transaction_record(test_db):
    """Verifies that a transaction and its inference result are correctly saved and retrieved."""
    sample_tx = {"Time": 100.0, "Amount": 49.99, "V1": 0.5, "V2": -1.2}
    sample_pred = {
        "prediction": 0,
        "probability": 0.012345,
        "is_fraud": False,
        "threshold_used": 0.95
    }

    record = create_transaction_record(test_db, sample_tx, sample_pred)
    assert record.id is not None
    assert record.amount == 49.99
    assert record.transaction_time == 100.0
    assert record.fraud_probability == 0.012345
    assert record.prediction == 0
    assert record.is_fraud is False
    assert record.threshold_used == 0.95

    # Retrieve by ID
    fetched = get_transaction_by_id(test_db, record.id)
    assert fetched is not None
    assert fetched.id == record.id
    assert fetched.amount == 49.99


def test_get_transaction_by_id_not_found(test_db):
    """Verifies that querying a non-existent ID returns None."""
    fetched = get_transaction_by_id(test_db, transaction_id=999999)
    assert fetched is None


def test_get_transactions_filter_by_fraud_status(test_db):
    """Verifies filtering by is_fraud=True and is_fraud=False."""
    create_transaction_record(
        test_db, {"Time": 10.0, "Amount": 20.0},
        {"prediction": 0, "probability": 0.02, "is_fraud": False, "threshold_used": 0.95}
    )
    create_transaction_record(
        test_db, {"Time": 20.0, "Amount": 40.0},
        {"prediction": 0, "probability": 0.03, "is_fraud": False, "threshold_used": 0.95}
    )
    create_transaction_record(
        test_db, {"Time": 30.0, "Amount": 600.0},
        {"prediction": 1, "probability": 0.99, "is_fraud": True, "threshold_used": 0.95}
    )

    # Filter is_fraud=True
    fraud_records, fraud_total = get_transactions(test_db, is_fraud=True)
    assert fraud_total == 1
    assert len(fraud_records) == 1
    assert fraud_records[0].is_fraud is True

    # Filter is_fraud=False
    legit_records, legit_total = get_transactions(test_db, is_fraud=False)
    assert legit_total == 2
    assert len(legit_records) == 2
    assert all(r.is_fraud is False for r in legit_records)

    # No filter (all records)
    all_records, all_total = get_transactions(test_db, is_fraud=None)
    assert all_total == 3
    assert len(all_records) == 3


def test_get_transactions_pagination(test_db):
    """Verifies pagination skip and limit mechanics."""
    for i in range(15):
        create_transaction_record(
            test_db, {"Time": float(i), "Amount": float(i * 10)},
            {"prediction": 0, "probability": 0.01, "is_fraud": False, "threshold_used": 0.95}
        )

    # Page 1: limit 5, skip 0
    p1_records, total = get_transactions(test_db, skip=0, limit=5)
    assert total == 15
    assert len(p1_records) == 5

    # Page 2: limit 5, skip 5
    p2_records, total = get_transactions(test_db, skip=5, limit=5)
    assert total == 15
    assert len(p2_records) == 5

    # IDs on page 1 and page 2 must be distinct
    p1_ids = {r.id for r in p1_records}
    p2_ids = {r.id for r in p2_records}
    assert p1_ids.isdisjoint(p2_ids)


def test_get_transactions_date_filtering(test_db):
    """Verifies filtering by start_date and end_date."""
    now = datetime.now(timezone.utc)
    yesterday = now - timedelta(days=1)
    two_days_ago = now - timedelta(days=2)

    r1 = TransactionRecord(
        timestamp=two_days_ago.replace(tzinfo=None),
        transaction_time=1.0, amount=10.0, fraud_probability=0.01,
        prediction=0, is_fraud=False, threshold_used=0.95, features_json="{}"
    )
    r2 = TransactionRecord(
        timestamp=yesterday.replace(tzinfo=None),
        transaction_time=2.0, amount=20.0, fraud_probability=0.02,
        prediction=0, is_fraud=False, threshold_used=0.95, features_json="{}"
    )
    r3 = TransactionRecord(
        timestamp=now.replace(tzinfo=None),
        transaction_time=3.0, amount=30.0, fraud_probability=0.03,
        prediction=0, is_fraud=False, threshold_used=0.95, features_json="{}"
    )
    test_db.add_all([r1, r2, r3])
    test_db.commit()

    # Filter with start_date as yesterday's date
    yesterday_str = yesterday.strftime("%Y-%m-%d")
    results, count = get_transactions(test_db, start_date=yesterday_str)
    assert count == 2  # yesterday + now

    # Filter with end_date as two_days_ago
    two_days_ago_str = two_days_ago.strftime("%Y-%m-%d")
    results_old, count_old = get_transactions(test_db, end_date=two_days_ago_str)
    assert count_old == 1


def test_get_transactions_date_range_cross_validation_error(test_db):
    """Verifies that start_date later than end_date raises a ValueError."""
    with pytest.raises(ValueError, match="cannot be later than"):
        get_transactions(test_db, start_date="2026-11-10", end_date="2026-11-01")


def test_parse_date_boundary_invalid():
    """Verifies that malformed date strings raise a ValueError."""
    with pytest.raises(ValueError, match="Invalid date format"):
        parse_date_boundary("not-a-valid-date")


def test_get_transaction_stats(test_db):
    """Verifies KPI calculation (total, frauds, legit, fraud_rate)."""
    stats_empty = get_transaction_stats(test_db)
    assert stats_empty["total_transactions"] == 0
    assert stats_empty["fraud_percentage"] == 0.0

    for _ in range(3):
        create_transaction_record(
            test_db, {"Time": 0.0, "Amount": 10.0},
            {"prediction": 0, "probability": 0.01, "is_fraud": False, "threshold_used": 0.95}
        )
    create_transaction_record(
        test_db, {"Time": 0.0, "Amount": 500.0},
        {"prediction": 1, "probability": 0.98, "is_fraud": True, "threshold_used": 0.95}
    )

    stats = get_transaction_stats(test_db)
    assert stats["total_transactions"] == 4
    assert stats["detected_frauds"] == 1
    assert stats["legitimate_transactions"] == 3
    assert stats["fraud_percentage"] == 25.0
