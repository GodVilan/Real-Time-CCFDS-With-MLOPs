import os
from fastapi import FastAPI, HTTPException, Security, UploadFile, File, Request
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import json
from datetime import datetime
# from dotenv import load_dotenv
# load_dotenv()

# --------------------------
# MLflow Configuration
# --------------------------
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_registry_uri(MLFLOW_TRACKING_URI)

MODEL_NAME = "FraudDetectionModel"
MODEL_URI = f"models:/{MODEL_NAME}@production"

print("🚀 Loading production model from MLflow registry...")
model = mlflow.sklearn.load_model(MODEL_URI)

# --------------------------
# Load Production Threshold from MLflow Tag
# --------------------------
client = mlflow.MlflowClient()

try:
    version_info = client.get_model_version_by_alias(
        MODEL_NAME,
        "production"
    )

    threshold_tag = client.get_model_version(
        MODEL_NAME,
        version_info.version
    ).tags.get("production_threshold")

    THRESHOLD = float(threshold_tag) if threshold_tag else 0.5

    print(f"✅ Loaded production threshold: {THRESHOLD}")

except Exception as e:
    print("⚠ Could not load threshold from registry. Using default 0.5")
    THRESHOLD = 0.5


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


class ThresholdUpdate(BaseModel):
    threshold: float


# --------------------------
# Health Check
# --------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------
# Update Threshold (Runtime Only)
# --------------------------
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
        raise HTTPException(status_code=400,
                            detail="Expected 30 features")

    feature_names = model.feature_names_in_
    data = pd.DataFrame([transaction.features], columns=feature_names)

    prob = float(model.predict_proba(data)[0][1])
    pred = 1 if prob >= THRESHOLD else 0

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

    probs = model.predict_proba(df.values)[:, 1]
    preds = (probs >= THRESHOLD).astype(int)

    df["fraud"] = preds
    df["fraud_probability"] = probs

    return df.head(10).to_dict(orient="records")
