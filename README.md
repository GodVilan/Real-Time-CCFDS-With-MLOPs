
# 🚀 Real-Time Credit Card Fraud Detection System (Production-Grade MLOps)

![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![MLflow](https://img.shields.io/badge/MLflow-Model%20Registry-orange)
![AWS](https://img.shields.io/badge/AWS-Deployed-yellow)
![CI/CD](https://img.shields.io/badge/GitHub%20Actions-CI/CD-success)
![Python](https://img.shields.io/badge/Python-3.11-blue)

An end-to-end **production-grade Machine Learning system** for real-time credit card fraud detection featuring automated training pipelines, model governance, CI/CD automation, and secure cloud deployment on AWS.

---

## ⚡ Live System Overview

This project simulates how **real FinTech ML systems** operate in production environments.

✅ Automated training pipeline  
✅ Model registry governance  
✅ Containerized inference API  
✅ Secure cloud deployment  
✅ Continuous delivery of ML models  

---

## 🏗 System Architecture

```
GitHub Actions (CI/CD)
        │
        ▼
Train → Evaluate → Register Model → Promote Alias
        │
        ▼
MLflow Model Registry
(PostgreSQL RDS + S3 Artifacts)
        │
        ▼
FastAPI Inference Service (Docker)
        │
        ▼
Nginx Reverse Proxy
        │
        ▼
Public API Endpoint (AWS EC2)
```

---

## ☁ Infrastructure Components

| Component | Purpose |
|-----------|---------|
| AWS EC2 | Hosts API, MLflow, and Nginx |
| PostgreSQL (RDS) | MLflow backend store |
| Amazon S3 | Model artifact storage |
| Docker | Containerized services |
| GitHub Actions | Automated training & deployment |

---

## 📌 Key Features

- Automated ML training using GitHub Actions
- MLflow Model Registry with alias-based promotion (`@production`)
- Secure FastAPI inference service
- API key authentication
- Reverse proxy using Nginx
- Cloud deployment on AWS
- Automated model versioning
- Production decision threshold management

---

## 🧠 Machine Learning Pipeline

### Models Trained
- Logistic Regression
- Random Forest
- XGBoost

### Training Strategy
- SMOTE applied **only to training data**
- ROC-AUC optimization
- Automatic best model selection
- Experiment tracking with MLflow
- Automatic model registration
- Production alias promotion

---

## 📊 Example Model Performance

| Model | ROC-AUC |
|------|---------|
| Logistic Regression | 0.9899 |
| Random Forest | 1.0000 |
| XGBoost | 1.0000 |

The best-performing model is automatically promoted to:

@production

---

## 🌐 API Endpoints

### ✅ Health Check
GET /health

Response:
{
  "status": "healthy"
}

### 🔍 Single Prediction
POST /predict

Headers:
- X-API-Key: <your-api-key>
- Content-Type: application/json

Request:
{
  "features": [ ... 30 feature values ... ]
}

Response:
{
  "fraud": 0,
  "fraud_probability": 0.15,
  "fraud_probability_percent": 15,
  "decision_threshold": 0.5
}

### 📦 Batch Prediction
POST /predict-batch

---

## 🔐 Security

- API key authentication
- Nginx reverse proxy isolation
- MLflow access control
- Backend database secured via AWS RDS

---

## 🐳 Local Development Setup

Clone Repository:
```bash
git clone https://github.com/GodVilan/Real-Time-CCFDS-With-MLOPs.git
cd Real-Time-CCFDS-With-MLOPs
```

Start Services:
```bash
docker compose up -d --build
```

---

## 🧪 Run Training Locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
```
```bash
python3 src/train.py
```

---

## 📊 Model Governance

Uses MLflow Model Registry aliases:
- @production

---

## 🚀 Deployment

AWS EC2 + Docker Compose + Nginx + RDS + S3

MLflow UI:
[UI Link](http://18.189.20.189:5000/)

Public API:
[API Link](http://18.189.20.189/docs)

---

## 📈 Future Improvements

- Prometheus + Grafana monitoring
- Automatic model reload
- Drift detection
- Canary deployment
- Kubernetes scaling

---

## 🎯 Why This Project Matters

Demonstrates real-world:
- End-to-end ML system design
- Production deployment
- CI/CD automation
- Model governance
- Cloud infrastructure integration

---

## 👤 Author

SRIKANTH REDDY N

Tech Stack: AWS • MLflow • FastAPI • Docker • MLOps
