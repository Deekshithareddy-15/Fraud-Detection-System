"""
Main FastAPI Application module for Credit Card Fraud Detection.
Includes structured logging, request duration tracking, robust input validation,
SQLite persistence, audit history querying, and comprehensive health monitoring.
"""

import os
import json
import math
import time
import logging
from datetime import datetime, timezone
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import text
import joblib
import pandas as pd
import numpy as np

from .database import get_db, init_db
from . import crud
from .schemas import (
    Transaction,
    PredictionResponse,
    TransactionDetailResponse,
    PaginatedTransactionsResponse,
    TransactionStatsResponse,
    HealthResponse,
)
from src.drift_detection import detect_drift

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("fraud_api")

app = FastAPI(
    title="Fraud Detection API",
    version="2.1",
    description="Production-grade real-time credit card fraud inference, audit logging, and analytics API."
)


# HTTP Request Timing and Structured Logging Middleware
@app.middleware("http")
async def log_request_performance(request: Request, call_next):
    """
    Measures request duration in milliseconds, logs incoming endpoint calls,
    and attaches an X-Response-Time-Ms header to the HTTP response.
    """
    start_time = time.perf_counter()
    try:
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        response.headers["X-Response-Time-Ms"] = str(duration_ms)
        
        # Log all API calls (excluding noisy static asset requests)
        if not request.url.path.startswith("/static"):
            logger.info(
                f"{request.method} {request.url.path} -> Status {response.status_code} ({duration_ms}ms)"
            )
        return response
    except Exception as err:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.error(
            f"Unhandled exception during {request.method} {request.url.path} ({duration_ms}ms): {err}",
            exc_info=True
        )
        raise err


# Custom Request Validation Error Handler for Clear, Beginner-Friendly Feedback
@app.exception_handler(RequestValidationError)
async def custom_validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Transforms Pydantic validation errors into structured, user-friendly JSON messages.
    """
    formatted_errors = []
    for err in exc.errors():
        field_path = " -> ".join(str(loc) for loc in err.get("loc", []))
        message = err.get("msg", "Invalid value")
        formatted_errors.append({"field": field_path, "message": message})
        
    logger.warning(f"Validation failure on {request.method} {request.url.path}: {formatted_errors}")
    return JSONResponse(
        status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
        content={
            "detail": "Input validation error. Please verify the supplied fields and value ranges.",
            "errors": formatted_errors
        }
    )


# Mount static files for dashboard frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", tags=["Frontend"])
async def read_index():
    """Serves the front-end analytics dashboard."""
    return FileResponse(os.path.join(static_dir, "index.html"))


# Model, Scaler, and Threshold Configuration
MODEL_PATH = "models/saved_models/best_fraud_model.pkl"
SCALER_PATH = "models/saved_models/scaler.pkl"
THRESHOLD_PATH = "models/saved_models/threshold_config.json"

model = None
scaler = None
classification_threshold = 0.50
model_name = "XGBoost"


@app.on_event("startup")
def load_artifacts_and_init_db():
    """Initializes SQLite database tables and loads model artifacts into memory."""
    global model, scaler, classification_threshold, model_name

    # Initialize SQLite database tables
    try:
        init_db()
        logger.info("SQLite database and transaction_records table initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing SQLite database: {e}")

    # Load machine learning artifacts
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            raise FileNotFoundError("Model or Scaler artifact file not found. Please train model first.")

        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        if os.path.exists(THRESHOLD_PATH):
            with open(THRESHOLD_PATH, "r") as f:
                t_cfg = json.load(f)
                classification_threshold = float(t_cfg.get("classification_threshold", 0.50))
                model_name = t_cfg.get("best_model_name", "XGBoost")
            logger.info(f"Loaded tuned classification threshold: {classification_threshold} for {model_name}")

        logger.info(f"Best ML model ({type(model).__name__}) and Scaler loaded into memory.")
    except Exception as e:
        logger.error(f"Error loading model artifacts: {e}")


def check_database_liveness(db: Session) -> bool:
    """Executes a lightweight query to verify active SQLite connection."""
    try:
        db.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database liveness check failed: {e}")
        return False


@app.get("/health", response_model=HealthResponse, tags=["System"])
@app.get("/api/health", response_model=HealthResponse, tags=["System"])
def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint reporting API health, model loading state, and database connectivity.
    Available at both /health and /api/health.
    """
    is_db_connected = check_database_liveness(db)
    is_model_ready = (model is not None) and (scaler is not None)
    overall_status = "healthy" if (is_db_connected and is_model_ready) else "degraded"

    return {
        "status": overall_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_loaded": is_model_ready,
        "model_name": model_name,
        "scaler_loaded": scaler is not None,
        "database_connected": is_db_connected,
        "classification_threshold": classification_threshold,
        "dataset_mode": "real"
    }


@app.get("/drift", tags=["Monitoring"])
async def check_drift(batch_path: str = Query("data/raw/new_batch.csv", description="Path to evaluation batch CSV")):
    """Runs KS-drift detection comparing incoming batch data against baseline feature stats."""
    drift_result = detect_drift(batch_path)
    if "error" in drift_result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=drift_result["error"]
        )
    return drift_result


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_fraud(transaction: Transaction, db: Session = Depends(get_db)):
    """
    Evaluates fraud risk for an incoming transaction and persists the result to SQLite history.
    Returns fraud probability, binary classification label, and the decision threshold applied.
    """
    if not model or not scaler:
        logger.error("Inference requested but model or scaler is not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Ensure training pipeline has completed successfully."
        )

    # Convert validated Pydantic payload to DataFrame
    data_dict = transaction.dict()
    columns = [f"V{i+1}" for i in range(28)] + ["Time", "Amount"]
    df = pd.DataFrame([data_dict], columns=columns)

    # Apply training RobustScaler to Amount and Time
    try:
        df[['Amount', 'Time']] = scaler.transform(df[['Amount', 'Time']])
    except Exception as e:
        logger.error(f"Feature scaling preprocessing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error during feature scaling preprocessing: {str(e)}"
        )

    # Predict fraud probability using the model
    try:
        probability = float(model.predict_proba(df)[0][1])
        is_fraud = bool(probability >= classification_threshold)
        prediction = int(1 if is_fraud else 0)
        prob_rounded = round(probability, 6)

        pred_dict = {
            "prediction": prediction,
            "probability": prob_rounded,
            "is_fraud": is_fraud,
            "threshold_used": classification_threshold
        }

        # Persist transaction to SQLite audit trail
        record = crud.create_transaction_record(
            db=db,
            transaction_dict=data_dict,
            prediction_result=pred_dict
        )

        logger.info(
            f"Prediction completed: ID=#{record.id}, Amount=${transaction.Amount:.2f}, "
            f"Prob={prob_rounded:.4f}, Label={'FRAUD' if is_fraud else 'LEGITIMATE'} "
            f"(Threshold={classification_threshold})"
        )

        return {
            "prediction": prediction,
            "probability": prob_rounded,
            "is_fraud": is_fraud,
            "threshold_used": classification_threshold,
            "transaction_id": record.id
        }
    except Exception as e:
        logger.error(f"Inference error during model execution: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


@app.get("/api/transactions", response_model=PaginatedTransactionsResponse, tags=["History"])
def get_transactions_history(
    page: int = Query(1, ge=1, description="Page number starting from 1"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page (1 to 100 maximum)"),
    is_fraud: Optional[bool] = Query(None, description="Filter by fraud status (true for fraud, false for legitimate)"),
    start_date: Optional[str] = Query(None, description="Filter transactions on/after this ISO date (e.g. 2026-11-01)"),
    end_date: Optional[str] = Query(None, description="Filter transactions on/before this ISO date (e.g. 2026-11-03)"),
    db: Session = Depends(get_db)
):
    """
    Retrieves a paginated list of transaction audit records with optional status and date filters.
    Includes maximum page size enforcement (100) and strict date range validation.
    """
    try:
        skip = (page - 1) * page_size
        records, total = crud.get_transactions(
            db=db,
            skip=skip,
            limit=page_size,
            is_fraud=is_fraud,
            start_date=start_date,
            end_date=end_date
        )
        total_pages = math.ceil(total / page_size) if total > 0 else 1

        return {
            "items": records,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages
        }
    except ValueError as val_err:
        logger.warning(f"Invalid filter parameter provided to /api/transactions: {val_err}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except Exception as err:
        logger.error(f"Database error querying transaction history: {err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve transaction history."
        )


@app.get("/api/transactions/{transaction_id}", response_model=TransactionDetailResponse, tags=["History"])
def get_transaction_detail(transaction_id: int, db: Session = Depends(get_db)):
    """
    Retrieves full audit details and deserialized input features for a single transaction by ID.
    Returns 404 if the requested transaction does not exist.
    """
    record = crud.get_transaction_by_id(db, transaction_id=transaction_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction with ID {transaction_id} not found."
        )

    try:
        features = json.loads(record.features_json)
    except Exception:
        features = {}

    return {
        "id": record.id,
        "timestamp": record.timestamp,
        "transaction_time": record.transaction_time,
        "amount": record.amount,
        "fraud_probability": record.fraud_probability,
        "prediction": record.prediction,
        "is_fraud": record.is_fraud,
        "threshold_used": record.threshold_used,
        "features": features
    }


@app.get("/api/stats", response_model=TransactionStatsResponse, tags=["Analytics"])
def get_summary_statistics(db: Session = Depends(get_db)):
    """
    Returns aggregate transaction counts, detected fraud counts, and fraud percentage.
    """
    try:
        stats = crud.get_transaction_stats(db)
        return stats
    except Exception as e:
        logger.error(f"Error computing transaction statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compute transaction statistics."
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
