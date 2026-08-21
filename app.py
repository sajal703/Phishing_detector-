"""
app.py
Flask application for the Phishing URL Detector.

Routes:
  GET  /                 -> web UI
  POST /predict           -> JSON API: {"url": "..."} -> prediction + explanation
  GET  /history           -> JSON API: last N scans
"""

import os
import sqlite3
from datetime import datetime, timezone

from flask import Flask, request, jsonify, render_template, g
import joblib
import pandas as pd

from features import extract_feature_vector, explain_prediction, FEATURE_NAMES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "phishing_model.pkl")
DB_PATH = os.path.join(BASE_DIR, "history.db")

app = Flask(__name__)

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise SystemExit(
        f"[ERROR] Model file not found at: {MODEL_PATH}\n"
        "Run 'python train_model.py' first to generate the model."
    )
except Exception as exc:
    raise SystemExit(f"[ERROR] Failed to load model: {exc}")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            scanned_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
    return db


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        feature_vector = extract_feature_vector(url)
    except Exception as e:
        return jsonify({"error": f"Could not parse URL: {str(e)}"}), 400

    feature_df = pd.DataFrame([feature_vector], columns=FEATURE_NAMES)
    proba = model.predict_proba(feature_df)[0]
    prediction = model.predict(feature_df)[0]
    label = "Phishing" if prediction == 1 else "Legitimate"
    confidence = float(proba[1] if prediction == 1 else proba[0])
    reasons = explain_prediction(url)

    # Log to history
    db = get_db()
    db.execute(
        "INSERT INTO scan_history (url, prediction, confidence, scanned_at) VALUES (?, ?, ?, ?)",
        (url, label, confidence, datetime.now(timezone.utc).isoformat()),
    )
    db.commit()

    return jsonify({
        "url": url,
        "prediction": label,
        "confidence": round(confidence * 100, 2),
        "reasons": reasons,
    })


@app.route("/history", methods=["GET"])
def history():
    limit = request.args.get("limit", default=20, type=int)
    db = get_db()
    cur = db.execute(
        "SELECT url, prediction, confidence, scanned_at FROM scan_history ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    result = [
        {"url": r[0], "prediction": r[1], "confidence": round(r[2] * 100, 2), "scanned_at": r[3]}
        for r in rows
    ]
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
