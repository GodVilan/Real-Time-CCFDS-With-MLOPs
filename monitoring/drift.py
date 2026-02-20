import os
import json
import pandas as pd
import boto3
import requests
from evidently import Report
from evidently.metrics import DataDriftPreset

BUCKET_NAME = "mlflow-artifacts-srikanth"
REFERENCE_KEY = "reference/processed_data.csv"

LOG_FILE = "prediction_logs.json"
DRIFT_REPORT_FILE = "drift_report.html"

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = "godvilan/real-time-ccfds-with-mlops"
WORKFLOW_FILE = "train.yml"


def load_reference_data():
    s3 = boto3.client("s3")
    s3.download_file(BUCKET_NAME, REFERENCE_KEY, "reference.csv")
    return pd.read_csv("reference.csv")


def load_current_data():
    if not os.path.exists(LOG_FILE):
        return None

    logs = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            logs.append(json.loads(line))

    if not logs:
        return None

    df = pd.DataFrame([log["input"]["features"] for log in logs])

    columns = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
    df.columns = columns

    return df


def run_drift_detection():
    reference_data = load_reference_data()
    current_data = load_current_data()

    if current_data is None:
        print("No data for drift detection.")
        return False

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_data.drop(columns=["Class"], errors="ignore"),
        current_data=current_data
    )

    report.save_html(DRIFT_REPORT_FILE)

    result = report.as_dict()
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]

    return drift_detected


def trigger_retraining():
    if not GITHUB_TOKEN:
        return

    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{WORKFLOW_FILE}/dispatches"

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    data = {"ref": "main"}

    requests.post(url, headers=headers, json=data)


if __name__ == "__main__":
    drift = run_drift_detection()

    if drift:
        print("Drift detected. Triggering retraining.")
        trigger_retraining()
    else:
        print("No drift detected.")