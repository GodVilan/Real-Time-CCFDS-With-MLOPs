# 🚀 Real-Time Credit Card Fraud Detection System (Production-Grade MLOps)

An end-to-end machine learning system for **real-time credit card fraud
detection** with automated training, model registry governance, CI/CD
integration, and secure containerized deployment on AWS.

------------------------------------------------------------------------

## 📌 Overview

This project demonstrates a **production-ready ML system** featuring:

-   Automated model training via **GitHub Actions**
-   **MLflow Model Registry** with alias-based promotion (`@production`)
-   Dockerized **FastAPI inference service**
-   Secure API access with **API key authentication**
-   Reverse proxy using **Nginx**
-   **AWS EC2 deployment**
-   **PostgreSQL (RDS)** as MLflow backend store
-   **Amazon S3** for artifact storage

The system supports:

-   Real-time single prediction
-   Batch prediction
-   Automated model versioning
-   Production threshold management

------------------------------------------------------------------------

## 🏗 System Architecture

GitHub Actions (CI/CD) │ ▼ Train → Evaluate → Register Model → Promote
Alias │ ▼ MLflow Registry (RDS Backend + S3 Artifacts) │ ▼ FastAPI
Inference Service (Docker) │ ▼ Nginx Reverse Proxy │ ▼ Public API
Endpoint (AWS EC2)

------------------------------------------------------------------------

## ☁ Infrastructure Components

  Component          Purpose
  ------------------ ---------------------------------
  EC2                Hosts API, MLflow, and Nginx
  RDS (PostgreSQL)   MLflow backend store
  S3                 Model artifact storage
  Docker             Containerized services
  GitHub Actions     Automated training & deployment

------------------------------------------------------------------------

## 🧠 Machine Learning Pipeline

### Models Trained

-   Logistic Regression
-   Random Forest
-   XGBoost

### Training Features

-   SMOTE applied **only to training data**
-   ROC-AUC optimization
-   Automatic threshold selection
-   Best-model auto selection
-   MLflow experiment tracking
-   Automatic model registration
-   Production alias assignment

------------------------------------------------------------------------

### 📊 Example Performance

  Model                 ROC-AUC
  --------------------- ---------
  Logistic Regression   0.9899
  Random Forest         1.0000
  XGBoost               1.0000

The best-performing model is automatically promoted to **@production**.

------------------------------------------------------------------------

## 🌐 API Endpoints

### Health Check

GET /health

Response: { "status": "healthy" }

### Single Prediction

POST /predict

Headers: X-API-Key: `<your-api-key>`{=html} Content-Type:
application/json

Body: { "features": \[ ... 30 feature values ... \] }

Response: { "fraud": 0, "fraud_probability": 0.15,
"fraud_probability_percent": 15, "decision_threshold": 0.5 }

### Batch Prediction

POST /predict-batch

------------------------------------------------------------------------

## 🔐 Security

-   API key authentication
-   Reverse proxy isolation via Nginx
-   MLflow allowed-host validation
-   Backend database isolated in AWS RDS

------------------------------------------------------------------------

## 🐳 Local Development Setup

Clone repository: git clone `<repo-url>`{=html} cd
real-time-ccfds-with-mlops

Start services: docker compose up -d --build

------------------------------------------------------------------------

## 🧪 Run Training Locally

python3 -m venv venv source venv/bin/activate pip install -r
requirements.txt

export MLFLOW_TRACKING_URI=http://localhost:5000

python3 src/train.py

------------------------------------------------------------------------

## 📊 Model Governance

Uses MLflow Model Registry aliases: - @production

------------------------------------------------------------------------

## 🚀 Deployment

AWS EC2 + Docker Compose + Nginx + RDS + S3

------------------------------------------------------------------------

## 📈 Future Improvements

-   Prometheus + Grafana monitoring
-   Automatic model reload
-   Drift detection
-   Canary deployment
-   Kubernetes scaling

------------------------------------------------------------------------

## 🎯 Why This Project Matters

Demonstrates real-world: - End-to-end ML system design - Production
deployment - CI/CD automation - Model governance - Cloud infrastructure
integration

------------------------------------------------------------------------

## 👤 Author

SRIKANTH REDDY N

Tech Stack: AWS • MLflow • FastAPI • Docker • MLOps
