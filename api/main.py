from fastapi import FastAPI, HTTPException, Security, UploadFile, File
from fastapi.security import APIKeyHeader
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from pydantic import BaseModel
import mlflow
# import mlflow.pyfunc
import numpy as np
import pandas as pd
import json
from datetime import datetime
import mlflow.sklearn
import joblib

# --------------------------
# MLflow Configuration
# --------------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_registry_uri("sqlite:///mlflow.db")

# MODEL_URI = "models:/FraudDetectionModel@production"
# model = mlflow.sklearn.load_model(MODEL_URI)

model = joblib.load("production_model.pkl")

# Set Threshold for Fraud Classification
import os

THRESHOLD = 0.5  # fallback default

if os.path.exists("production_threshold.txt"):
    with open("production_threshold.txt", "r") as f:
        THRESHOLD = float(f.read().strip())
        print(f"Loaded production threshold: {THRESHOLD}")

# --------------------------
# FastAPI App
# --------------------------
app = FastAPI(title="Credit Card Fraud Detection API")

# --------------------------
# Rate Limiting
# --------------------------
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# --------------------------
# API Key Authentication
# --------------------------
API_KEY = "supersecretkey"
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")

# --------------------------
# Logging Function
# --------------------------
def log_prediction(features, pred, prob):
    log = {
        "timestamp": datetime.utcnow().isoformat(),
        "prediction": int(pred),
        "probability": float(prob),
        "features": features
    }

    with open("predictions.log", "a") as f:
        f.write(json.dumps(log) + "\n")

# --------------------------
# Schemas
# --------------------------
class Transaction(BaseModel):
    features: list[float]

# --------------------------
# Health Check
# --------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

from pydantic import BaseModel

class ThresholdUpdate(BaseModel):
    threshold: float

@app.post("/set-threshold")
def set_threshold(payload: ThresholdUpdate):
    global THRESHOLD

    if not 0 <= payload.threshold <= 1:
        raise HTTPException(status_code=400,
                            detail="Threshold must be between 0 and 1")

    THRESHOLD = payload.threshold
    return {
        "message": "Threshold updated",
        "new_threshold": THRESHOLD
    }

# --------------------------
# Single Prediction
# --------------------------
@app.post("/predict")
@limiter.limit("10/minute")
def predict(
    request: Request,
    transaction: Transaction,
    api_key: str = Security(api_key_header)
):

    verify_api_key(api_key)

    if len(transaction.features) != 30:
        raise HTTPException(status_code=400, detail="Expected 30 features")

    feature_names = model.feature_names_in_
    data = pd.DataFrame([transaction.features], columns=feature_names)


    prob = float(model.predict_proba(data)[0][1])
    pred = 1 if prob >= THRESHOLD else 0

    probs = model.predict_proba(data)
    print("Raw probabilities:", probs)

    log_prediction(transaction.features, pred, prob)

    return {
        "fraud": pred,
        "fraud_probability": prob,
        "fraud_probability_percent": round(prob * 100, 6),
        "decision_threshold": THRESHOLD
    }


# --------------------------
# Batch Prediction
# --------------------------
@app.post("/predict-batch")
@limiter.limit("5/minute")
async def predict_batch(
    request: Request,
    file: UploadFile = File(...),
    api_key: str = Security(api_key_header)
):

    verify_api_key(api_key)

    df = pd.read_csv(file.file)

    if df.shape[1] != 30:
        raise HTTPException(status_code=400,
                            detail="CSV must contain exactly 30 columns")

    preds = model.predict(df.values)
    probs = model.predict_proba(df.values)[:, 1]

    df["fraud"] = preds
    df["fraud_probability"] = probs

    return df.head(10).to_dict(orient="records")
