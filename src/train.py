"""Train and compare three basic classification models.

What this script does, in order:
  1. Loads data/dataset.csv (1,450 rows).
  2. Drops the 7 constant columns (they never change in FD001, so they
     carry no information): os3, s1, s5, s10, s16, s18, s19.
     That leaves 18 features: cycle, os1, os2, s2, s3, s4, s6, s7, s8,
     s9, s11, s12, s13, s14, s15, s17, s20, s21.
  3. Splits 80/20 with stratification (same % of "needs maintenance"
     in train and test), random_state=42 so the split is identical
     every time. Train = 1,160 rows, test = 290 rows.
  4. Trains Logistic Regression, Decision Tree and Random Forest,
     each inside a Pipeline with a StandardScaler fitted on TRAIN
     data only (fitting the scaler on all data before splitting
     would leak test information into training).
  5. Reports accuracy, precision, recall, F1 and ROC-AUC on the test
     set, plus the confusion matrix, plus 5-fold cross-validation
     accuracy on the full 1,450 rows.
  6. Picks the deployed model by 5-fold CV accuracy (tie-break: ROC-AUC),
     saves it to models/model.pkl, and writes every number to
     models/metrics.json and models/test_predictions.csv.

Run from the project folder:
    python src/train.py
"""

import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data", "dataset.csv")
MODEL_PATH = os.path.join(BASE, "models", "model.pkl")
METRICS_PATH = os.path.join(BASE, "models", "metrics.json")
PREDICTIONS_PATH = os.path.join(BASE, "models", "test_predictions.csv")

# Columns that are constant in FD001 (standard deviation = 0). Verified
# in the report, Appendix C. A constant column cannot help a model.
CONSTANT_COLUMNS = ["os3", "s1", "s5", "s10", "s16", "s18", "s19"]

FEATURES = ["cycle", "os1", "os2", "s2", "s3", "s4", "s6", "s7", "s8",
            "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
}


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel().tolist()
    return {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, prob)), 4),
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "correct": int(tn + tp),
        "total": int(tn + fp + fn + tp),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {DATA_PATH}: {len(df)} rows")
    assert len(df) == 1450, f"Expected 1450 rows, got {len(df)}"
    X = df[FEATURES]
    y = df["needs_maintenance"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"Train: {len(X_train)} rows ({int(y_train.sum())} need maintenance)")
    print(f"Test:  {len(X_test)} rows ({int(y_test.sum())} need maintenance)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    pipelines = {}
    for name, model in MODELS.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        pipe.fit(X_train, y_train)
        pipelines[name] = pipe
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        metrics = evaluate(pipe, X_test, y_test)
        metrics["cv_accuracy_mean"] = round(float(scores.mean()), 4)
        metrics["cv_accuracy_folds"] = [round(float(s), 4) for s in scores]
        results[name] = metrics
        print(f"\n{name}: test accuracy={metrics['accuracy']}, "
              f"precision={metrics['precision']}, recall={metrics['recall']}, "
              f"f1={metrics['f1']}, ROC-AUC={metrics['roc_auc']}, "
              f"CV accuracy={metrics['cv_accuracy_mean']}")
        print(f"  Confusion matrix TN={metrics['confusion_matrix']['TN']} "
              f"FP={metrics['confusion_matrix']['FP']} "
              f"FN={metrics['confusion_matrix']['FN']} "
              f"TP={metrics['confusion_matrix']['TP']}")
        print(classification_report(y_test, pipe.predict(X_test), digits=4))

    # Selection rule, fixed before looking at the test set in spirit:
    # highest 5-fold CV accuracy; ROC-AUC breaks ties. The single
    # hold-out test set is NOT the selector, because one 290-row test
    # set is noisy (Decision Tree wins it by only 2 rows).
    best_name = max(results, key=lambda n: (results[n]["cv_accuracy_mean"],
                                            results[n]["roc_auc"]))
    print(f"\nDeployed model (by CV accuracy, then ROC-AUC): {best_name}")
    best = pipelines[best_name]
    os.makedirs(os.path.join(BASE, "models"), exist_ok=True)
    joblib.dump({"pipeline": best, "features": FEATURES,
                 "model_name": best_name}, MODEL_PATH)

    # Feature importances of the deployed Random Forest, for the report.
    importances = {}
    inner = best.named_steps["model"]
    if hasattr(inner, "feature_importances_"):
        importances = {f: round(float(v), 4)
                       for f, v in sorted(zip(FEATURES, inner.feature_importances_),
                                          key=lambda kv: -kv[1])}

    test_out = X_test.copy()
    test_out["actual"] = y_test.values
    test_out["predicted"] = best.predict(X_test)
    test_out["probability_needs_maintenance"] = best.predict_proba(X_test)[:, 1]
    test_out.to_csv(PREDICTIONS_PATH, index=False)

    payload = {
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_positives": int(y_train.sum()),
        "test_positives": int(y_test.sum()),
        "features": FEATURES,
        "constant_columns_dropped": CONSTANT_COLUMNS,
        "test_split": "80/20, stratified, random_state=42",
        "selection_rule": "Highest 5-fold CV accuracy; ROC-AUC breaks ties",
        "deployed_model": best_name,
        "results": results,
        "feature_importances": importances,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {MODEL_PATH}")
    print(f"Saved {METRICS_PATH}")
    print(f"Saved {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
