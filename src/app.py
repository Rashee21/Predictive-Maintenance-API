"""Flask API for the predictive-maintenance model.

Endpoints:
    GET  /           - short description and the endpoint list.
    GET  /health     - "ok" plus the deployed model name. Use this first
                        to check the server is running.
    GET  /model-info - model name, the 18 features, and the test metrics.
    POST /predict    - one engine reading -> maintenance decision.
                        Body: {"cycle": ..., "os1": ..., ..., "s21": ...}
                        All 18 features are required.
                        Reply: {"needs_maintenance": 0 or 1,
                                "probability": 0.0-1.0,
                                "decision": "MAINTENANCE REQUIRED" or "NO MAINTENANCE NEEDED"}

Run from the project folder (after training, or with the supplied
pre-trained models/model.pkl):
    python src/app.py
Then open http://127.0.0.1:5000/health in a browser.

The model file already contains its scaler, so the API applies exactly
the same scaling that was used in training. Never scale by hand.
"""

import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "models", "model.pkl")
METRICS_PATH = os.path.join(BASE, "models", "metrics.json")

app = Flask(__name__)
_bundle = joblib.load(MODEL_PATH)
PIPELINE = _bundle["pipeline"]
FEATURES = _bundle["features"]
MODEL_NAME = _bundle["model_name"]


@app.route("/")
def home():
    return jsonify({
        "project": "Predictive Maintenance API",
        "model": MODEL_NAME,
        "endpoints": ["/health", "/model-info", "/predict"],
        "features": FEATURES,
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME})


@app.route("/model-info")
def model_info():
    import json
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    return jsonify({
        "model": MODEL_NAME,
        "features": FEATURES,
        "deployed_model_metrics": metrics["results"][MODEL_NAME],
        "all_models": metrics["results"],
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object with all 18 features."}), 400
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing features: {missing}"}), 400
    try:
        # A DataFrame (not a plain list) keeps the feature names, so
        # scikit-learn does not warn about missing feature names.
        row = pd.DataFrame([{f: float(data[f]) for f in FEATURES}], columns=FEATURES)
    except (TypeError, ValueError):
        return jsonify({"error": "All feature values must be numbers."}), 400
    prediction = int(PIPELINE.predict(row)[0])
    probability = float(PIPELINE.predict_proba(row)[0][1])
    return jsonify({
        "needs_maintenance": prediction,
        "probability": round(probability, 4),
        "decision": ("MAINTENANCE REQUIRED" if prediction == 1
                     else "NO MAINTENANCE NEEDED"),
    })


if __name__ == "__main__":
    # debug=False on purpose: this is a college demo server, not a
    # production deployment. Do not use Flask's dev server in production.
    app.run(
    host="0.0.0.0",
    port=int(os.environ.get("PORT", 5000)),
    debug=False
)
