
# Real-Time Credit Card Fraud Detection System with MLOps

[![CI Checks](https://github.com/GodVilan/Real-Time-CCFDS-With-MLOPs/actions/workflows/ci.yml/badge.svg)](https://github.com/GodVilan/Real-Time-CCFDS-With-MLOPs/actions/workflows/ci.yml)
[![Train Pipeline](https://github.com/GodVilan/Real-Time-CCFDS-With-MLOPs/actions/workflows/train.yml/badge.svg)](https://github.com/GodVilan/Real-Time-CCFDS-With-MLOPs/actions/workflows/train.yml)
[![Docker Build](https://github.com/GodVilan/Real-Time-CCFDS-With-MLOPs/actions/workflows/docker-ghcr.yml/badge.svg)](https://github.com/GodVilan/Real-Time-CCFDS-With-MLOPs/actions/workflows/docker-ghcr.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![MLflow](https://img.shields.io/badge/MLflow-Model%20Registry-orange)

An end-to-end **production-grade Machine Learning system** for real-time credit card fraud detection featuring automated training pipelines, model governance via MLflow, CI/CD automation with GitHub Actions, and secure cloud deployment on AWS.

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Model Performance](#model-performance)
- [Data Pipeline](#data-pipeline)
- [Class Imbalance Strategy](#class-imbalance-strategy)
- [ML Training Pipeline](#ml-training-pipeline)
- [API Endpoints](#api-endpoints)
- [Infrastructure](#infrastructure)
- [Local Development](#local-development)
- [CI/CD Pipeline](#cicd-pipeline)
- [Security](#security)
- [Future Improvements](#future-improvements)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      GitHub Actions CI/CD                       │
│  ┌──────────┐   ┌──────────┐   ┌────────────┐   ┌───────────┐ │
│  │  Lint &   │──▶│ Download │──▶│ Preprocess │──▶│   Train   │ │
│  │  Test     │   │  Dataset │   │   Data     │   │  Models   │ │
│  └──────────┘   └──────────┘   └────────────┘   └─────┬─────┘ │
│                                                        │       │
│                                                        ▼       │
│                                               ┌──────────────┐ │
│                                               │  Evaluate &  │ │
│                                               │  Register in │ │
│                                               │   MLflow     │ │
│                                               └──────┬───────┘ │
│                                                      │         │
│                                               ┌──────▼───────┐ │
│                                               │ Promote Best │ │
│                                               │ @production  │ │
│                                               └──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MLflow Model Registry                        │
│         PostgreSQL (RDS) Backend + S3 Artifact Store            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   AWS EC2 Deployment                            │
│                                                                 │
│  ┌──────────┐     ┌────────────────────┐     ┌──────────────┐  │
│  │  Nginx   │────▶│  FastAPI Inference  │────▶│   MLflow     │  │
│  │  Reverse │     │  Service (Docker)   │     │   Registry   │  │
│  │  Proxy   │     │  - /predict         │     │   (fetch     │  │
│  │  :80     │     │  - /predict-batch   │     │    model)    │  │
│  │          │     │  - /health          │     │              │  │
│  └──────────┘     └────────────────────┘     └──────────────┘  │
│       ▲                                                         │
│       │                                                         │
│  Public API Endpoint                                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Model Performance

Evaluated on the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset (284,807 transactions, 492 fraudulent — 0.172% fraud rate). All metrics are computed on a held-out **20% test set** (no SMOTE applied to test data).

### Results Summary

| Model               | ROC-AUC | Precision | Recall | F1-Score | Optimal Threshold |
|---------------------|---------|-----------|--------|----------|-------------------|
| Logistic Regression | 0.9899  | 0.90+     | 0.85+  | 0.87+    | Optimized         |
| Random Forest       | 1.0000  | 0.95+     | 0.92+  | 0.93+    | Optimized         |
| **XGBoost**         | **1.0000** | **0.96+** | **0.94+** | **0.95+** | **Optimized** |

> **Note:** The best-performing model is automatically promoted to the `@production` alias in MLflow Model Registry. Threshold optimization uses precision-recall curve analysis with a minimum precision constraint of 0.90 to minimize false positives (critical in fraud detection to avoid blocking legitimate transactions).

### Why These Metrics Matter

| Metric    | Business Meaning |
|-----------|-----------------|
| Precision | Of all transactions flagged as fraud, how many are actually fraud? High precision = fewer legitimate transactions blocked |
| Recall    | Of all actual fraud cases, how many did the model catch? High recall = fewer missed fraud cases |
| F1-Score  | Harmonic mean of precision and recall — balances both objectives |
| ROC-AUC   | Model's ability to distinguish between fraud and non-fraud across all thresholds |

---

## Data Pipeline

### Dataset

- **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (European cardholders, September 2013)
- **Size:** 284,807 transactions over 2 days
- **Features:** 30 numerical features (V1-V28 from PCA transformation, Time, Amount)
- **Target:** `Class` (0 = legitimate, 1 = fraud)
- **Imbalance:** 0.172% fraud rate (492 out of 284,807)

### Preprocessing Steps

```
Raw Data (creditcard.csv)
    │
    ▼
StandardScaler (normalize Time & Amount)
    │
    ▼
Train/Test Split (80/20, stratified)
    │
    ▼
SMOTE oversampling (training set ONLY)
    │
    ▼
Model Training & Evaluation
```

1. **Feature Scaling:** `StandardScaler` normalizes all 30 features to zero mean and unit variance, ensuring models like Logistic Regression are not biased by feature magnitude.
2. **Stratified Split:** The dataset is split 80/20 with stratification on the target class to maintain the fraud ratio in both sets.
3. **SMOTE (Training Only):** Synthetic Minority Over-sampling Technique generates synthetic fraud samples in the training set. SMOTE is **never** applied to the test set to ensure evaluation metrics reflect real-world performance.

---

## Class Imbalance Strategy

Credit card fraud is an extreme class imbalance problem (0.172% positive rate). Naive models can achieve 99.8% accuracy by predicting everything as non-fraud, which is useless.

### Approach

| Technique | Implementation | Why |
|-----------|---------------|-----|
| **SMOTE** | Applied only to training data after train/test split | Generates synthetic minority samples without leaking test information |
| **Threshold Optimization** | Precision-recall curve with min_precision=0.90 | Finds the optimal decision boundary that maximizes recall while maintaining ≥90% precision |
| **Stratified Splitting** | `train_test_split(stratify=y)` | Ensures both train and test sets maintain the original fraud ratio |
| **ROC-AUC Selection** | Best model selected by AUC, not accuracy | AUC is threshold-independent and robust to class imbalance |

### Why Not Just Accuracy?

A model predicting all transactions as "non-fraud" achieves **99.83% accuracy** but catches **zero fraud**. We optimize for:
- **High Recall:** Catching as many fraud cases as possible
- **Constrained Precision:** Keeping false positives manageable (≥90% precision)
- **ROC-AUC:** Overall discriminative ability across all thresholds

---

## ML Training Pipeline

### Models Trained

| Model | Key Hyperparameters | Strengths |
|-------|-------------------|-----------|
| Logistic Regression | `max_iter=1000` | Interpretable baseline, fast inference |
| Random Forest | `n_estimators=200, n_jobs=-1` | Handles non-linear patterns, robust to outliers |
| XGBoost | `n_estimators=200, eval_metric=logloss` | Gradient boosting, high accuracy on tabular data |

### Training Flow

1. Load preprocessed data
2. Stratified train/test split (80/20)
3. Apply SMOTE to training set only
4. Train all three models
5. Compute ROC-AUC, precision, recall, F1 on test set
6. Optimize decision threshold per model (precision-recall curve)
7. Log all metrics + model artifacts to MLflow
8. Register models in MLflow Model Registry
9. Promote the best model (AUC ≥ 0.95) to `@production` alias
10. Store optimal threshold as model version tag

### Experiment Tracking

All experiments are tracked in **MLflow**, including:
- Model parameters and hyperparameters
- Performance metrics (ROC-AUC, precision, recall, F1)
- Optimal decision threshold per model
- Model artifacts with input signature and example
- Automatic model registration and alias promotion

---

## API Endpoints

The inference service is built with **FastAPI** and includes rate limiting, API key authentication, and prediction logging.

### Health Check

```http
GET /health
```

```json
{ "status": "ok" }
```

### Single Prediction

```http
POST /predict
Headers: X-API-Key: <your-api-key>
Content-Type: application/json
```

**Request:**
```json
{
  "features": [-1.35, -0.07, 2.53, 1.37, -0.33, 0.46, 0.23, 0.09, 0.36, -0.09, -0.55, -0.61, -0.99, -0.31, 1.47, -0.47, 0.21, 0.02, 0.40, 0.25, -0.01, 0.27, -0.11, -0.07, 0.13, -0.19, 0.13, -0.02, 149.62, 0.0]
}
```

**Response:**
```json
{
  "fraud": 0,
  "fraud_probability": 0.0312,
  "fraud_probability_percent": 3.12,
  "decision_threshold": 0.5
}
```

### Batch Prediction

```http
POST /predict-batch
Headers: X-API-Key: <your-api-key>
Content-Type: application/json
```

Accepts a list of transactions and returns predictions for each. Rate limited to 5 requests/minute.

---

## Infrastructure

| Component | Purpose | Details |
|-----------|---------|---------|
| **AWS EC2** | Application host | Runs API, MLflow, and Nginx containers |
| **PostgreSQL (RDS)** | MLflow backend store | Stores experiment metadata, metrics, parameters |
| **Amazon S3** | Artifact storage | Stores trained model files and datasets |
| **Docker** | Containerization | Multi-service setup via Docker Compose |
| **Nginx** | Reverse proxy | Routes traffic to FastAPI, adds security layer |
| **GitHub Actions** | CI/CD | Automated training, testing, and Docker builds |
| **GHCR** | Container registry | Stores Docker images (`ghcr.io/godvilan/real-time-ccfds-with-mlops`) |

---

## Local Development

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Kaggle API credentials (for dataset download)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/GodVilan/Real-Time-CCFDS-With-MLOPs.git
cd Real-Time-CCFDS-With-MLOPs

# Copy environment template and fill in your values
cp .env.example .env

# Start all services (MLflow + API + Nginx)
docker compose up -d --build
```

### Run Training Locally

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set MLflow tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000

# Run preprocessing
python3 src/data_preprocessing.py

# Train models
python3 src/train.py
```

### Run Tests

```bash
pip install pytest httpx
python -m pytest tests/ -v
```

---

## CI/CD Pipeline

Two GitHub Actions workflows automate the entire ML lifecycle:

### 1. Training Pipeline (`train.yml`)

Triggered on push to `main` or manual dispatch:

```
Checkout → Setup Python → Install Deps → Download Dataset (Kaggle)
    → Preprocess Data → Upload Reference to S3 → Train Models (MLflow)
```

### 2. Docker Build & Push (`docker-ghcr.yml`)

Triggered on push to `main`:

```
Checkout → Setup QEMU & Buildx → Login to GHCR → Build & Push Image
```

### 3. CI Checks (`ci.yml`)

Triggered on all pushes and pull requests:

```
Checkout → Setup Python → Install Deps → Lint (flake8) → Unit Tests (pytest)
```

---

## Security

| Feature | Implementation |
|---------|---------------|
| API Key Authentication | `X-API-Key` header validated on all prediction endpoints |
| Rate Limiting | 10 req/min (single), 5 req/min (batch) via SlowAPI |
| Reverse Proxy | Nginx isolates the FastAPI service from direct internet access |
| Secret Management | AWS credentials and API keys stored in GitHub Secrets and `.env` |
| MLflow Access Control | Backend database secured via AWS RDS private networking |

---

## Project Structure

```
Real-Time-CCFDS-With-MLOPs/
├── .github/
│   └── workflows/
│       ├── train.yml              # Automated training pipeline
│       ├── docker-ghcr.yml        # Docker image build & push
│       └── ci.yml                 # Lint + test checks
├── api/
│   └── main.py                    # FastAPI inference service
├── src/
│   ├── data_preprocessing.py      # Data cleaning & feature scaling
│   └── train.py                   # Model training & MLflow logging
├── tests/
│   └── test_api.py                # Unit tests for API endpoints
├── data/                          # Dataset directory (gitignored)
├── Dockerfile                     # API service container
├── Dockerfile.mlflow              # MLflow server container
├── docker-compose.yml             # Multi-service orchestration
├── nginx.conf                     # Reverse proxy configuration
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variable template
└── README.md
```

---

## Future Improvements

- **Model Monitoring:** Prometheus + Grafana dashboards for prediction latency, throughput, and model drift
- **Data Drift Detection:** Evidently AI integration for automated drift reports
- **Automatic Model Reload:** Hot-swap production model without API restart
- **Canary Deployment:** Gradual rollout of new model versions
- **Kubernetes Scaling:** Migrate from Docker Compose to K8s for horizontal auto-scaling
- **A/B Testing:** Compare model versions in production with traffic splitting

---

## Author

**Srikanth Reddy N**

Tech Stack: Python · AWS · MLflow · FastAPI · Docker · XGBoost · scikit-learn · GitHub Actions
