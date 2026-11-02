from fastapi.testclient import TestClient
from api.main import app

def test_health_check():
    """Verifies baseline health check response format."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "healthy"
        assert "classification_threshold" in data


def test_health_endpoints_extended():
    """Verifies both /health and /api/health return full system diagnostics and response headers."""
    with TestClient(app) as client:
        for endpoint in ["/health", "/api/health"]:
            response = client.get(endpoint)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True
            assert data["model_name"] == "XGBoost"
            assert data["scaler_loaded"] is True
            assert data["database_connected"] is True
            assert data["classification_threshold"] == 0.95
            assert "X-Response-Time-Ms" in response.headers


def test_predict_fraud_structure():
    payload = {
        "Time": 0.0,
        "Amount": 100.0,
        "V1": 0.0, "V2": 0.0, "V3": 0.0, "V4": 0.0, "V5": 0.0,
        "V6": 0.0, "V7": 0.0, "V8": 0.0, "V9": 0.0, "V10": 0.0,
        "V11": 0.0, "V12": 0.0, "V13": 0.0, "V14": 0.0, "V15": 0.0,
        "V16": 0.0, "V17": 0.0, "V18": 0.0, "V19": 0.0, "V20": 0.0,
        "V21": 0.0, "V22": 0.0, "V23": 0.0, "V24": 0.0, "V25": 0.0,
        "V26": 0.0, "V27": 0.0, "V28": 0.0
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "is_fraud" in data
        assert "probability" in data
        assert "prediction" in data
        assert "threshold_used" in data
        assert "transaction_id" in data
        assert isinstance(data["transaction_id"], int)
        assert "X-Response-Time-Ms" in response.headers


def test_predict_validation_errors():
    """Tests input range checks and custom error response format."""
    valid_payload = {
        "Time": 10.0, "Amount": 50.0,
        **{f"V{i}": 0.0 for i in range(1, 29)}
    }
    with TestClient(app) as client:
        # Negative Amount
        bad_amount = {**valid_payload, "Amount": -5.0}
        resp = client.post("/predict", json=bad_amount)
        assert resp.status_code == 422
        err = resp.json()
        assert "detail" in err
        assert "errors" in err

        # Negative Time
        bad_time = {**valid_payload, "Time": -1.0}
        assert client.post("/predict", json=bad_time).status_code == 422

        # Extreme out-of-range PCA feature (exceeds 100.0)
        bad_v = {**valid_payload, "V1": 999.0}
        assert client.post("/predict", json=bad_v).status_code == 422

        # Missing feature column
        incomplete = {k: v for k, v in valid_payload.items() if k != "V10"}
        assert client.post("/predict", json=incomplete).status_code == 422


def test_predict_legitimate_transaction():
    """Tests a known low-risk transaction produces is_fraud=False with prob < threshold."""
    normal_payload = {
        "Time": 3111.53, "Amount": 4466.00,
        "V1": 2.8391, "V2": 0.9238, "V3": -3.6642, "V4": 1.1628, "V5": -2.3285, "V6": 0.6929, "V7": -8.2251, "V8": -3.5896, "V9": 1.5926, "V10": 0.6461,
        "V11": -1.5169, "V12": -4.4633, "V13": 0.0745, "V14": -1.0586, "V15": -0.0904, "V16": -0.1871, "V17": 1.7995, "V18": 2.1965, "V19": 0.38,
        "V20": -0.5568, "V21": -0.1681, "V22": 1.4523, "V23": 0.7858, "V24": -2.1967, "V25": -1.3171, "V26": 1.8704, "V27": 1.028, "V28": 2.2098
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=normal_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["is_fraud"] is False
        assert data["prediction"] == 0
        assert data["probability"] < data["threshold_used"]
        assert data["transaction_id"] is not None


def test_predict_fraud_transaction():
    """Tests a known high-risk transaction produces is_fraud=True with prob >= threshold."""
    fraud_payload = {
        "Time": 406.0,
        "V1": -2.3122265423263, "V2": 1.95199201064158, "V3": -1.60985073229769, "V4": 3.9979055875468,
        "V5": -0.522187864667764, "V6": -1.42654531920595, "V7": -2.53738730624579, "V8": 1.39165724829804,
        "V9": -2.77008927719433, "V10": -2.77227214465915, "V11": 3.20203320709635, "V12": -2.89990738849473,
        "V13": -0.595221881324605, "V14": -4.28925378244217, "V15": 0.389724120274487, "V16": -1.14074717980657,
        "V17": -2.83005567450437, "V18": -0.0168224681808257, "V19": 0.416955705037907, "V20": 0.126910559061474,
        "V21": 0.517232370861764, "V22": -0.0350493686052974, "V23": -0.465211076182388, "V24": 0.320198198514526,
        "V25": 0.0445191674731724, "V26": 0.177839798284401, "V27": 0.261145002567677, "V28": -0.143275874698919,
        "Amount": 0.0
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=fraud_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["is_fraud"] is True
        assert data["prediction"] == 1
        assert data["probability"] >= data["threshold_used"]
        assert data["transaction_id"] is not None


def test_index_html_contains_fraud_probability_elements():
    """Verifies that the frontend template includes elements for fraud probability and threshold."""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        html = response.text
        assert "Fraud Probability" in html
        assert "Decision Threshold" in html
        assert 'id="fraudProb"' in html
        assert 'id="thresholdUsed"' in html
        assert 'id="riskExplanation"' in html


def test_api_transactions_history_endpoint():
    """Tests GET /api/transactions returns paginated transactions."""
    with TestClient(app) as client:
        response = client.get("/api/transactions?page=1&page_size=5")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert "total_pages" in data
        assert data["page"] == 1
        assert data["page_size"] == 5


def test_api_transactions_pagination_boundaries():
    """Tests invalid pagination limits (page < 1 or page_size > 100) return 422."""
    with TestClient(app) as client:
        # Page 0 (minimum is 1)
        resp_bad_page = client.get("/api/transactions?page=0")
        assert resp_bad_page.status_code == 422

        # Page size > 100 (maximum is 100)
        resp_bad_size = client.get("/api/transactions?page_size=150")
        assert resp_bad_size.status_code == 422


def test_api_transactions_filtering():
    """Tests GET /api/transactions filtering by is_fraud=true and false."""
    with TestClient(app) as client:
        # Filter fraud
        resp_fraud = client.get("/api/transactions?is_fraud=true")
        assert resp_fraud.status_code == 200
        data_fraud = resp_fraud.json()
        for item in data_fraud["items"]:
            assert item["is_fraud"] is True

        # Filter legit
        resp_legit = client.get("/api/transactions?is_fraud=false")
        assert resp_legit.status_code == 200
        data_legit = resp_legit.json()
        for item in data_legit["items"]:
            assert item["is_fraud"] is False


def test_api_transactions_date_filtering_validation():
    """Tests invalid date inputs and start_date > end_date return 400 Bad Request."""
    with TestClient(app) as client:
        # Malformed date string
        resp_bad_format = client.get("/api/transactions?start_date=not-a-date")
        assert resp_bad_format.status_code == 400
        assert "Invalid date format" in resp_bad_format.json()["detail"]

        # start_date > end_date
        resp_bad_range = client.get("/api/transactions?start_date=2026-11-10&end_date=2026-11-01")
        assert resp_bad_range.status_code == 400
        assert "cannot be later than" in resp_bad_range.json()["detail"]


def test_api_transaction_detail_and_not_found():
    """Tests GET /api/transactions/{id} for existing transaction and 404 for invalid ID."""
    with TestClient(app) as client:
        tx_payload = {
            "Time": 99.0, "Amount": 123.45,
            **{f"V{i}": 0.1 for i in range(1, 29)}
        }
        pred_res = client.post("/predict", json=tx_payload)
        assert pred_res.status_code == 200
        tx_id = pred_res.json()["transaction_id"]

        # Fetch detail
        detail_res = client.get(f"/api/transactions/{tx_id}")
        assert detail_res.status_code == 200
        detail = detail_res.json()
        assert detail["id"] == tx_id
        assert detail["amount"] == 123.45
        assert "features" in detail
        assert detail["features"]["Time"] == 99.0

        # Non-existent ID returns 404
        not_found_res = client.get("/api/transactions/99999999")
        assert not_found_res.status_code == 404
        assert "not found" in not_found_res.json()["detail"].lower()


def test_api_stats_endpoint():
    """Tests GET /api/stats returns valid aggregate metrics."""
    with TestClient(app) as client:
        res = client.get("/api/stats")
        assert res.status_code == 200
        stats = res.json()
        assert "total_transactions" in stats
        assert "detected_frauds" in stats
        assert "legitimate_transactions" in stats
        assert "fraud_percentage" in stats
        assert stats["total_transactions"] >= 0
        assert stats["detected_frauds"] + stats["legitimate_transactions"] == stats["total_transactions"]


def test_predict_empty_payload_fails():
    """Tests that sending an empty JSON object returns HTTP 422."""
    with TestClient(app) as client:
        res = client.post("/predict", json={})
        assert res.status_code == 422
        assert "errors" in res.json()


def test_predict_incorrect_types_fail():
    """Tests that non-numeric types (e.g. strings) are rejected with 422."""
    bad_payload = {
        "Time": "not_a_number", "Amount": "one_hundred",
        **{f"V{i}": 0.0 for i in range(1, 29)}
    }
    with TestClient(app) as client:
        res = client.post("/predict", json=bad_payload)
        assert res.status_code == 422


def test_drift_endpoint_missing_file_returns_404():
    """Tests GET /drift returns 404 when batch file does not exist."""
    with TestClient(app) as client:
        res = client.get("/drift?batch_path=data/raw/non_existent_batch.csv")
        assert res.status_code == 404
        assert "not found" in res.json()["detail"].lower()


def test_drift_endpoint_with_valid_batch(tmp_path):
    """Tests GET /drift produces a valid drift report when given a valid CSV batch."""
    import pandas as pd
    import numpy as np

    # Generate a mini batch matching the 30-feature schema
    batch_file = tmp_path / "valid_batch.csv"
    cols = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
    data = {c: np.random.randn(50) for c in cols}
    pd.DataFrame(data).to_csv(batch_file, index=False)

    with TestClient(app) as client:
        res = client.get(f"/drift?batch_path={str(batch_file)}")
        assert res.status_code == 200
        report = res.json()
        assert "drift_detected" in report
        assert "details" in report
        assert isinstance(report["drift_detected"], bool)


def test_concurrent_predictions():
    """Tests that concurrent prediction requests do not cause SQLite database lock errors."""
    import concurrent.futures

    def send_req(i):
        payload = {
            "Time": float(i * 10), "Amount": float(10 + i),
            **{f"V{k}": 0.01 * i for k in range(1, 29)}
        }
        with TestClient(app) as client:
            resp = client.post("/predict", json=payload)
            return resp.status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(send_req, i) for i in range(12)]
        results = [f.result() for f in futures]

    assert set(results) == {200}

