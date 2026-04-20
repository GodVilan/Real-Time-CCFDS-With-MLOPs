import os
import json
import mlflow
import mlflow.sklearn
import pandas as pd
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from pydantic import BaseModel
import joblib
import tempfile

# --------------------------------------------------
# MLflow Configuration
# --------------------------------------------------

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_registry_uri(MLFLOW_TRACKING_URI)

MODEL_NAME = "FraudDetectionModel"
MODEL_ALIAS = "production"

print("🚀 Loading production model from MLflow registry...")
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
model = mlflow.sklearn.load_model(MODEL_URI)

print("✅ Model loaded successfully.")

# --------------------------------------------------
# Load the scaler for input preprocessing
# --------------------------------------------------
# Replace the scaler loading block with this
scaler = None
try:
    client = mlflow.tracking.MlflowClient()
    model_version = client.get_model_version_by_alias(MODEL_NAME, "production")
    run_id = model_version.run_id

    scaler_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="scaler/scaler.pkl"
    )
    import joblib
    scaler = joblib.load(scaler_path)
    print("✅ Scaler loaded successfully.")
except Exception as e:
    print(f"⚠ Scaler not found in this model version, skipping: {e}")
# --------------------------------------------------
# Load Production Threshold from Registry
# --------------------------------------------------

def load_production_threshold():
    try:
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_model_version_by_alias(MODEL_NAME, "production")
        threshold = model_version.tags.get("production_threshold")
        if threshold:
            return float(threshold)
        return 0.5
    except Exception as e:
        print("Threshold load error:", e)
        return 0.5


THRESHOLD = load_production_threshold()
print(f"🎯 Production Threshold: {THRESHOLD}")

# --------------------------------------------------
# FastAPI App
# --------------------------------------------------

app = FastAPI(title="Credit Card Fraud Detection API")

# --------------------------------------------------
# Rate Limiting
# --------------------------------------------------

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# --------------------------------------------------
# API Key Security
# --------------------------------------------------

API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key")


def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")


# --------------------------------------------------
# Prediction Logging
# --------------------------------------------------

LOG_FILE = "prediction_logs.json"


def log_prediction(input_data: dict, prediction: int, probability: float):
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": input_data,
        "prediction": int(prediction),
        "probability": float(probability)
    }

    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print("Logging error:", e)


# --------------------------------------------------
# Schemas
# --------------------------------------------------

class Transaction(BaseModel):
    features: list[float]


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------------------------------
# Single Prediction Endpoint
# --------------------------------------------------

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

    try:
        feature_names = model.feature_names_in_
        input_df = pd.DataFrame([transaction.features], columns=feature_names)

        # ------------------------------
        # Apply scaling before prediction
        # ------------------------------
        if scaler:
            input_df = pd.DataFrame(scaler.transform(input_df), columns=feature_names)

        prob = float(model.predict_proba(input_df)[0][1])
        prediction = 1 if prob >= THRESHOLD else 0

        log_prediction(
            input_data={"features": transaction.features},
            prediction=prediction,
            probability=prob
        )

        return {
            "fraud": prediction,
            "fraud_probability": prob,
            "fraud_probability_percent": round(prob * 100, 4),
            "decision_threshold": THRESHOLD
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------
# Batch Prediction Endpoint
# --------------------------------------------------

@app.post("/predict-batch")
@limiter.limit("5/minute")
async def predict_batch(
    request: Request,
    transactions: list[Transaction],
    api_key: str = Security(api_key_header)
):

    verify_api_key(api_key)

    try:
        data = [t.features for t in transactions]

        if any(len(row) != 30 for row in data):
            raise HTTPException(status_code=400, detail="Each transaction must have 30 features")

        feature_names = model.feature_names_in_
        input_df = pd.DataFrame(data, columns=feature_names)
        # -------------------------------
        # Apply scaling before prediction
        # -------------------------------
        if scaler:
            input_df = pd.DataFrame(scaler.transform(input_df), columns=feature_names)

        probs = model.predict_proba(input_df)[:, 1]
        preds = [1 if p >= THRESHOLD else 0 for p in probs]

        results = []

        for i in range(len(data)):
            log_prediction(
                input_data={"features": data[i]},
                prediction=preds[i],
                probability=float(probs[i])
            )

            results.append({
                "fraud": preds[i],
                "fraud_probability": float(probs[i])
            })

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))