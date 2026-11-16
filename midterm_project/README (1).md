# Supreme Court Case Outcome Prediction

Midterm project for [ML Zoomcamp 2026](https://github.com/DataTalksClub/machine-learning-zoomcamp) (DataTalks.Club).

A binary classifier that estimates whether the **petitioner wins** a US Supreme Court case, using only information available about the case *before* the ruling. The model is served as a FastAPI web service, and containerized with Docker.

> **Not legal advice.** This is an educational project. The model has modest predictive power (see [Results](#results)) and must not be used to make legal, financial, or professional decisions.

---

## Table of contents

1. [Problem description](#1-problem-description)
2. [Dataset](#2-dataset)
3. [Project structure](#3-project-structure)
4. [Data cleaning and EDA](#4-data-cleaning-and-eda)
5. [Modelling](#5-modelling)
6. [Results](#6-results)
7. [Setup: virtual environment and dependencies](#7-setup-virtual-environment-and-dependencies)
8. [Run the notebook and training script](#8-run-the-notebook-and-training-script)
9. [Run the web service locally](#9-run-the-web-service-locally)
10. [Run with Docker](#10-run-with-docker)
11. [Limitations](#11-limitations)
12. [Citation and acknowledgements](#12-citation-and-acknowledgements)

---

## 1. Problem description

When the US Supreme Court takes a case, one party (the **petitioner**) asks the Court to review a lower court's decision. The Court then either rules in the petitioner's favor (for example by reversing or vacating the lower decision) or against them (for example by affirming it).

**Question:** given the background of a case (who the parties are, which court it came from, why the Court agreed to hear it, how the lower court ruled, and when it was heard), can we estimate the probability that the petitioner wins?

**How the model would be used:** as a decision-support and research tool for law students, legal analysts, and researchers studying judicial behaviour. A user enters a case profile and receives a win probability for the petitioner. The probability is meant to be used for *ranking* and *exploration* (for example, which kinds of cases have historically favoured petitioners), not as a verdict.

**Task type:** binary classification. The target is `partyWinning` (1 = petitioner wins, 0 = petitioner loses).

**Why it is hard:** Supreme Court outcomes depend heavily on case merits and legal arguments, which are not captured by coded background features. A small but real signal is the realistic expectation.

---

## 2. Dataset

**Supreme Court Database (SCDB), Modern Database, 2024 Release 01**, Case Centered data, organized by Supreme Court citation (one row per case). It covers terms 1946 to 2023.

- Release page: https://scdb.la.psu.edu/data/2024-release-01/
- Codebook: https://scdb.la.psu.edu/online-codebook/
- The CSV is committed in this repo at `data/scdb_case_centered.csv`.

**Getting the data yourself (optional):** on the release page, choose *Case Centered Data, Organized by Supreme Court Citation, Download CSV*. The download is a zip archive containing `SCDB_2024_01_caseCentered_Citation.csv`. Unzip it and save the CSV as `data/scdb_case_centered.csv`.

After keeping only cases with a clear outcome (`partyWinning` of 0 or 1), the dataset has **9,252 cases**. The petitioner wins **64.2%** of them.

---

## 3. Project structure

```
.
├── README.md
├── data/
│   └── scdb_case_centered.csv     # SCDB 2024 Release 01 (case centered)
├── notebook.ipynb                 # cleaning, EDA, feature importance, model selection
├── preprocess.py                  # shared preprocessing (used by training and serving)
├── train.py                       # trains the final model and saves model.bin
├── predict.py                     # FastAPI prediction service
├── model.bin                      # trained model, DictVectorizer, rare-category sets
├── requirements.txt               # pinned dependencies
├── Dockerfile
├── .dockerignore
└── images/                        # screenshots of the running service
```

---

## 4. Data cleaning and EDA

Full details are in `notebook.ipynb`.

**Target:** only cases with `partyWinning` of 0 or 1 are kept. Cases coded as "unclear" are dropped.

**Features used** (all known before the ruling): `term`, `naturalCourt`, `chief`, `petitioner`, `respondent`, `jurisdiction`, `adminAction`, `threeJudgeFdc`, `caseOrigin`, `caseSource`, `certReason`, `lcDisagreement`, `lcDisposition`, `lcDispositionDirection`.

**Leakage prevention:** columns that describe the outcome or are only known after the decision were excluded, for example vote counts (`majVotes`, `minVotes`), `decisionDirection`, `decisionType`, `caseDisposition`, `declarationUncon`, `precedentAlteration`, and opinion-writer fields. Identifiers and free text (`caseId`, `usCite`, `caseName`, and similar) were also excluded.

**Cleaning steps:**
- SCDB codes are numeric labels, not quantities, so all coded columns are treated as **categorical** (for example `12.0` becomes `"12"`).
- Missing values become their own category (`"missing"`). Missingness is meaningful here: `adminAction` is missing for 73.0% of cases because it is only coded when a federal agency is involved. Other missing shares: `lcDisposition` 13.5%, `caseOrigin` 3.7%, `caseSource` 2.2%, `lcDispositionDirection` 2.1%.
- Rare categories in `petitioner`, `respondent`, `caseOrigin`, and `caseSource` (fewer than 30 training cases) are grouped as `"other"`. Counts come from the training data only.
- `term` is rescaled to `(term - 1946) / 10` so linear models behave well.

**EDA highlights** (see the notebook for plots):
- Target balance: 64.2% petitioner wins, 35.8% losses. Always predicting "petitioner wins" therefore already gives about 64 to 70% accuracy, so accuracy alone is misleading. **ROC-AUC is the main metric.**
- Petitioner win rate by decade: <!-- TODO: add one sentence describing the trend from your plot -->
- Feature importance (XGBoost importances and mutual information): <!-- TODO: name the top 3 to 5 features from your plot -->

---

## 5. Modelling

**Chronological split (not random)**, to mimic predicting future terms and avoid temporal leakage:

| Split | Terms | Cases |
|---|---|---|
| Train | up to 2009 | 8,264 |
| Validation | 2010 to 2016 | 539 |
| Test | 2017 to 2023 | 449 |

Categorical features are one-hot encoded with scikit-learn's `DictVectorizer`, fitted on training data only.

**Models trained and tuned** (on validation ROC-AUC):

| Model | Parameters tuned | Best validation AUC |
|---|---|---|
| Logistic regression | `C` | 0.614 |
| Decision tree | `max_depth`, `min_samples_leaf` | 0.591 |
| Random forest | `n_estimators`, `max_depth`, `min_samples_leaf` | 0.615 |
| **XGBoost** | `learning_rate`, `max_depth`, `n_estimators` | **0.618** |

**Selected model:** XGBoost (`learning_rate=0.1`, `max_depth=3`, `n_estimators=300`). The top three models are within about 0.004 AUC, which is within the noise of a 539-case validation set, so logistic regression is a nearly-as-good, simpler alternative. XGBoost was chosen as the best validation score. The final model is retrained on train + validation (terms up to 2016) and evaluated **once** on the test set.

---

## 6. Results

Final model on the held-out test set (terms 2017 to 2023, 449 cases):

| Metric | Model | "Always predict petitioner wins" baseline |
|---|---|---|
| ROC-AUC | **0.610** | 0.500 |
| Accuracy | 0.693 | 0.697 |
| F1 | 0.815 | not reported |

**Interpretation:** validation and test AUC agree (0.618 vs 0.610), which suggests the model is not overfit to the validation set. The model ranks cases modestly better than chance, but its accuracy does not beat the trivial baseline, so it should be read as a weak probabilistic signal, not a reliable classifier. With only 449 test cases, the test scores carry roughly a few points of statistical noise.

---

## 7. Setup: virtual environment and dependencies

Tested with **Python 3.14**. Create and activate a virtual environment, then install the pinned dependencies.

**macOS / Linux**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

To run the notebook you will also need Jupyter and a plotting library, for example `pip install jupyter matplotlib`.

---

## 8. Run the notebook and training script

With the virtual environment active:

```bash
# explore: cleaning, EDA, feature importance, model selection
jupyter notebook notebook.ipynb

# train the final model and save model.bin
python train.py
```

`train.py` loads `data/scdb_case_centered.csv`, applies the same preprocessing used by the service (`preprocess.py`), trains XGBoost on terms up to 2016, prints the test-set AUC, F1, and accuracy, and saves `model.bin`.

---

## 9. Run the web service locally

```bash
uvicorn predict:app --port 8000
```

Endpoints:

| Method | Path | Description |
|---|---|---|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| POST | `/predict` | Returns the petitioner win probability |
| GET | `/docs` | Interactive API documentation (Swagger UI) |

**Example request:**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"term": 2020, "naturalCourt": 1706, "chief": "Roberts", "petitioner": 8, "respondent": 126, "jurisdiction": 1, "caseOrigin": 300, "caseSource": 300, "certReason": 1, "lcDisposition": 3, "lcDispositionDirection": 1}'
```

**Example response:**

```json
{"petitioner_win_probability": 0.63, "prediction": "petitioner wins"}
```

(The exact probability depends on the trained model. Only `term` is required, and omitted fields are treated as missing. Field values are the numeric codes defined in the [SCDB codebook](https://scdb.la.psu.edu/online-codebook/).)

Screenshot of the service running locally: `images/<!-- TODO: filename -->`

---

## 10. Run with Docker

**Build the image:**

```bash
docker build -t scotus-predict .
```

**Run the container:**

```bash
docker run -p 8000:8000 scotus-predict
```

Then send the same `curl` request as above to `http://127.0.0.1:8000/predict`. If port 8000 is already in use on your machine, map a different host port, for example `docker run -p 8001:8000 scotus-predict`, and use port 8001 in the request.

The image is based on `python:3.14-slim`, installs the pinned `requirements.txt`, and starts the service with uvicorn.

Screenshot of the container running: `images/<!-- TODO: filename -->`

---

## 11. Limitations

- **Weak signal.** Test AUC is about 0.61 and accuracy does not beat the always-"petitioner wins" baseline. Case outcomes depend on legal merits that these coded features do not capture.
- **Small evaluation sets.** The validation (539) and test (449) sets are small, so differences of about 0.01 AUC between models are within noise.
- **Possible residual leakage.** Some SCDB variables are coded by researchers after reading the opinion, even if they describe the case itself. Only variables believed to be known before the ruling were used, but the coding process cannot be fully audited.
- **Temporal drift.** The Court's composition, docket, and coding practices change over time, so older patterns may not hold for future terms.
- **Coded inputs.** The API expects numeric SCDB codes, which requires consulting the codebook.
- **Not legal advice.** See the note at the top.

---

## 12. Citation and acknowledgements

Data: Harold J. Spaeth, Lee Epstein, Michael J. Nelson, Andrew D. Martin, et al. 2024 Supreme Court Database, Version 2024 Release 1. URL: https://scdb.psu.edu

Course: [ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) by DataTalks.Club.
