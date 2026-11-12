"""Tests for the API. Run from the project folder with:

    python -m unittest discover -s tests -v

They use Flask's built-in test client, so no server needs to be
running. A test only passes if the real saved model file loads and
answers, which is what the CI pipeline checks on every push.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from app import FEATURES, app  # noqa: E402


class TestApi(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(base, "examples", "example_request.json")) as f:
            self.example = json.load(f)

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_model_info_lists_features(self):
        response = self.client.get("/model-info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.get_json()["features"]), 18)

    def test_predict_example(self):
        response = self.client.post("/predict", json=self.example)
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertIn(body["needs_maintenance"], [0, 1])
        self.assertTrue(0.0 <= body["probability"] <= 1.0)

    def test_predict_missing_feature_is_400(self):
        bad = dict(self.example)
        bad.pop(FEATURES[0])
        response = self.client.post("/predict", json=bad)
        self.assertEqual(response.status_code, 400)

    def test_predict_non_numeric_is_400(self):
        bad = dict(self.example)
        bad[FEATURES[0]] = "not-a-number"
        response = self.client.post("/predict", json=bad)
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
