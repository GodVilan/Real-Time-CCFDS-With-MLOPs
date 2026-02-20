import os
import json
import pandas as pd
import boto3
import requests
from io import StringIO

from evidently import Report
from evidently.metrics import DataDriftTable

# -----------------------------
# CONFIG
# -----------------------------

S3_BUCKET = os.getenv("S3_BUCKET")
S3_REFERENCE_KEY = os.getenv("S3_REFERENCE_KEY")  # e.g. "reference/processed_data.csv"

LOG_FILE = "predictions.log"
DRIFT_REPORT_FILE = "drift_report.html"

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = "godvilan/real-time-ccfds-with-mlops"
WORKFLOW_FILE = "train.yml"

# -----------------------------
# LOAD REFERENCE DATA FROM S3
# -----------------------------

def load_reference_data():
    if not S3_BUCKET or not S3_REFERENCE_KEY:
        raise ValueError("S3_BUCKET or S3_REFERENCE_KEY not set.")

    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_REFERENCE_KEY)
    data = obj["Body"].read().decode("utf-8")

    df = pd.read_csv(StringIO(data))

    # Remove target column
    df = df.drop(columns=["Class"], errors="ignore")

    return df


# -----------------------------
# LOAD CURRENT DATA FROM LOGS
# -----------------------------

def load_current_data():
    if not os.path.exists(LOG_FILE):
        print("No prediction logs found.")
        return None

    logs = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            logs.append(json.loads(line))

    if not logs:
        return None

    df = pd.DataFrame([log["features"] for log in logs])
    return df


# -----------------------------
# RUN DRIFT
# -----------------------------

def run_drift_detection():
    reference_data = load_reference_data()
    current_data = load_current_data()

    if current_data is None:
        print("No current data available.")
        return False

    report = Report(metrics=[DataDriftTable()])
    report.run(
        reference_data=reference_data,
        current_data=current_data
    )

    report.save_html(DRIFT_REPORT_FILE)
    print("Drift report generated.")

    results = report.as_dict()
    drift_detected = results["metrics"][0]["result"]["dataset_drift"]

    return drift_detected


# -----------------------------
# RETRAIN TRIGGER
# -----------------------------

def trigger_retraining():
    if not GITHUB_TOKEN:
        print("No GitHub token found. Skipping retraining.")
        return

    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{WORKFLOW_FILE}/dispatches"

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    data = {"ref": "main"}

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 204:
        print("Retraining workflow triggered.")
    else:
        print("Failed to trigger retraining:", response.text)


# -----------------------------
# MAIN
# -----------------------------

if __name__ == "__main__":
    drift_detected = run_drift_detection()

    if drift_detected:
        print("⚠ Drift detected.")
        trigger_retraining()
    else:
        print("✅ No significant drift detected.")
