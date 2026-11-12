"""Command-line demo: predict for one row without starting the API.

Usage (from the project folder):
    python src/predict.py --example          # uses examples/example_request.json
    python src/predict.py --file my_row.json # uses your own JSON file
"""

import argparse
import json
import os

import joblib
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", action="store_true")
    parser.add_argument("--file", default=None)
    args = parser.parse_args()
    path = args.file or os.path.join(BASE, "examples", "example_request.json")
    bundle = joblib.load(os.path.join(BASE, "models", "model.pkl"))
    with open(path) as f:
        data = json.load(f)
    row = pd.DataFrame([{feat: float(data[feat]) for feat in bundle["features"]}],
                       columns=bundle["features"])
    pred = int(bundle["pipeline"].predict(row)[0])
    prob = float(bundle["pipeline"].predict_proba(row)[0][1])
    print(f"Model: {bundle['model_name']}")
    print(f"Prediction: {pred} ({'MAINTENANCE REQUIRED' if pred else 'NO MAINTENANCE NEEDED'})")
    print(f"Probability of needing maintenance: {prob:.4f}")


if __name__ == "__main__":
    main()
