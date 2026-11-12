# Predictive Maintenance API with Automated CI/CD Pipeline

A deliberately basic college Machine Learning project. It answers one
question: **given one engine reading, does this engine need maintenance
within the next 30 cycles?**

## What it is, in one paragraph

The data are real NASA turbofan sensor readings (C-MAPSS, FD001 subset).
Each row is one engine at one cycle. Remaining Useful Life (RUL) is
computed from the engine's full run-to-failure history, and a row is
labelled `needs_maintenance = 1` when RUL is 30 cycles or less.
Three simple classifiers are trained and compared (Logistic Regression,
Decision Tree, Random Forest); the best by 5-fold cross-validation is
saved and served by a small Flask API. Every push to GitHub runs the
training and the tests again automatically (GitHub Actions CI).

## Results (exact, reproducible)

- Dataset: **1,450 rows** (first 1,450 rows of NASA `train_FD001.txt`),
  186 need maintenance, 1,264 do not.
- Split: 80/20, stratified, `random_state=42` → train 1,160, test 290.
- Features: 18 (7 constant columns dropped: os3, s1, s5, s10, s16, s18, s19).
- Deployed model: **Random Forest** (200 trees), chosen by 5-fold CV
  accuracy (0.9869), tie-break ROC-AUC (0.9982).
- Test set (290 rows): accuracy **0.9793**, precision **0.8974**,
  recall **0.9459**, F1 **0.9211**, ROC-AUC **0.9982**.
  Confusion matrix: TN 249, FP 4, FN 2, TP 35 (284/290 correct).
- Full comparison is in `models/metrics.json` and in the project report.

## Run it (short version)

```bash
python -m venv venv
# Windows: venv\Scripts\activate   |   Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python src/train.py
python src/app.py
```

Then open http://127.0.0.1:5000/health in a browser, and test a
prediction with the supplied example:

```bash
python src/predict.py --example
```

The full step-by-step guide, written assuming no prior knowledge, is
the separate **Instructions PDF** supplied with this project. The
**Viva PDF** explains every concept an examiner may ask about.

## Project layout

```
data/dataset.csv            1,450-row dataset (NASA C-MAPSS FD001, first 1,450 rows)
data/README.md              data source and column notes
src/prepare_data.py         how dataset.csv was built from the raw NASA file
src/train.py                train + compare + save the model and metrics
src/app.py                  Flask API: /, /health, /model-info, /predict
src/predict.py              command-line prediction demo
tests/test_api.py           5 API tests (run: python -m unittest discover -s tests -v)
examples/example_request.json       engine 1, cycle 192 (RUL 0) -> maintenance
examples/example_healthy.json       engine 1, cycle 1 (RUL 191) -> no maintenance
models/model.pkl            trained pipeline (scaler + Random Forest)
models/metrics.json         every metric, confusion matrix and CV fold
models/test_predictions.csv predictions for all 290 test rows
figures/                    charts used in the report
.github/workflows/ci.yml    GitHub Actions CI pipeline
Dockerfile                  optional container run
```

## API

- `GET /health` → `{"status": "ok", "model": "Random Forest"}`
- `GET /model-info` → model name, features and all test metrics
- `POST /predict` with all 18 features as JSON →
  `{"needs_maintenance": 1, "probability": 1.0, "decision": "MAINTENANCE REQUIRED"}`

Missing or non-numeric features return HTTP 400 with an error message.

## What this project deliberately does NOT claim

- It does **not** reproduce the CAMDE++ LSTM from the research paper.
  The paper's task is RUL regression plus a context-aware decision
  framework; this basic project converts that idea into its simplest
  decision form (a 30-cycle maintenance yes/no) so a viva can be
  defended line by line. The link is explained in the report, Ch. 2.
- Sensor names are **not** invented: the NASA readme gives only
  `s1..s21`, so those names are kept.
- The test split is a random row split, not an engine-wise split, so
  rows from the same engine can appear in both train and test. That
  limitation is stated in the report and is the first suggested
  improvement.

## Source

Saxena, A. and Goebel, K. (2008). Turbofan Engine Degradation
Simulation Data Set. NASA Ames Prognostics Data Repository.
