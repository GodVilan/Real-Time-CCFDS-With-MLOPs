import os
import json
import pandas as pd
import requests
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

REFERENCE_DATA_PATH = "data/processed/processed_data.csv"
LOG_FILE = "prediction_logs.json"
DRIFT_REPORT_FILE = "drift_report.html"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = "godvilan/real-time-ccfds-with-mlops"
WORKFLOW_FILE = "train.yml"


def load_reference_data():
    return pd.read_csv(REFERENCE_DATA_PATH)


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

    df = pd.DataFrame([log["input"] for log in logs])
    return df


def run_drift_detection():
    reference_data = load_reference_data()
    current_data = load_current_data()

    if current_data is None:
        print("No current data available for drift detection.")
        return False

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_data.drop(columns=["Class"], errors="ignore"),
        current_data=current_data
    )

    report.save_html(DRIFT_REPORT_FILE)
    print("Drift report generated.")

    result = report.as_dict()

    drift_detected = result["metrics"][0]["result"]["dataset_drift"]
    return drift_detected


def trigger_retraining():
    if not GITHUB_TOKEN:
        print("No GitHub token found.")
        return

    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{WORKFLOW_FILE}/dispatches"

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    data = {"ref": "main"}

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 204:
        print("Retraining workflow triggered successfully.")
    else:
        print("Failed to trigger retraining:", response.text)


if __name__ == "__main__":
    drift_detected = run_drift_detection()

    if drift_detected:
        print("⚠ Drift detected! Triggering retraining...")
        trigger_retraining()
    else:
        print("✅ No significant drift detected.")
