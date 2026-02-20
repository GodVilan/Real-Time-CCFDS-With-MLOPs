import os
import json
from datetime import datetime

import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# -----------------------------
# Configuration
# -----------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_NAME = "FraudDetectionModel"
MODEL_STAGE = "Production"

LOG_FILE = "prediction_logs.json"
THRESHOLD_FILE = "production_threshold.txt"

# ----------------------------
# Load Model
# ----------------------------
model = mlflow.pyfunc.load_model(
    "models:/FraudDetectionModel@production"
)
# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="Fraud Detection API")


class Transaction(BaseModel):
    features: list


@app.post("/predict")
def predict(transaction: Transaction):

    input_df = pd.DataFrame([transaction.features])

    probability = model.predict(input_df)[0]

    with open(THRESHOLD_FILE, "r") as f:
        threshold = float(f.read().strip())

    prediction = 1 if probability >= threshold else 0

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": {"features": transaction.features},
        "prediction": prediction,
        "probability": float(probability)
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {
        "prediction": prediction,
        "probability": float(probability),
        "threshold": threshold
    }