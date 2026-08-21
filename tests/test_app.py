import sys
import os
import tempfile


import pytest
import app as flask_app


@pytest.fixture
def client():
    # Use a fresh temporary database for each test to avoid polluting history.db
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    flask_app.app.config["TESTING"] = True
    flask_app.DB_PATH = db_path
    flask_app.init_db()

    with flask_app.app.test_client() as client:
        yield client

    os.close(db_fd)
    os.unlink(db_path)


def test_index_page_loads(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"URL Sentry" in res.data

def test_predict_requires_url(client):
    res = client.post("/predict", json={})
    assert res.status_code == 400
    assert "error" in res.get_json()

def test_predict_returns_prediction_fields(client):
    res = client.post("/predict", json={"url": "https://www.google.com"})
    assert res.status_code == 200
    data = res.get_json()
    assert "prediction" in data
    assert "confidence" in data
    assert "reasons" in data
    assert data["prediction"] in ["Phishing", "Legitimate"]

def test_predict_flags_ip_based_url(client):
    res = client.post("/predict", json={"url": "http://192.168.1.1/secure-login-verify"})
    data = res.get_json()
    assert data["prediction"] == "Phishing"

def test_predict_confidence_in_valid_range(client):
    res = client.post("/predict", json={"url": "https://www.wikipedia.org"})
    data = res.get_json()
    assert 0 <= data["confidence"] <= 100

def test_history_endpoint_returns_list(client):
    client.post("/predict", json={"url": "https://www.example.com"})
    res = client.get("/history?limit=5")
    assert res.status_code == 200
    assert isinstance(res.get_json(), list)

def test_predict_empty_url_string(client):
    res = client.post("/predict", json={"url": "   "})
    assert res.status_code == 400
