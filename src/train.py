import mlflow
import mlflow.sklearn
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import numpy as np

import os
import mlflow

# -------------------------------------
# MLflow Configuration
# -------------------------------------
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://18.118.48.132:5000"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_registry_uri(MLFLOW_TRACKING_URI)

mlflow.set_experiment("Credit Card Fraud Detection")


# -------------------------------------
# Threshold Optimization Function
# -------------------------------------
def find_optimal_threshold(y_true, probabilities, min_precision=0.90):

    precisions, recalls, thresholds = precision_recall_curve(
        y_true,
        probabilities
    )

    optimal_threshold = 0.5
    best_recall = 0

    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        if p >= min_precision and r > best_recall:
            best_recall = r
            optimal_threshold = t

    return optimal_threshold, best_recall


# -------------------------------------
# Training Function
# -------------------------------------
def train_model():

    print("🚀 Training started...")

    df = pd.read_csv("data/processed/processed_data.csv")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # ---------------------------------
    # Split FIRST (Avoid Leakage)
    # ---------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ---------------------------------
    # Apply SMOTE ONLY on training data
    # ---------------------------------
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print("✅ SMOTE applied to training set only")

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            n_estimators=200,
            n_jobs=-1
        )
    }

    best_model_name = None
    best_model_object = None
    best_roc_auc = 0
    best_run_id = None
    best_threshold = 0.5

    for name, model in models.items():

        with mlflow.start_run(run_name=name) as run:

            model.fit(X_train, y_train)

            probs = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, probs)

            # -------- Threshold Optimization --------
            optimal_threshold, best_recall = find_optimal_threshold(
                y_test,
                probs,
                min_precision=0.90
            )

            custom_preds = (probs >= optimal_threshold).astype(int)

            precision = precision_score(y_test, custom_preds)
            recall = recall_score(y_test, custom_preds)
            f1 = f1_score(y_test, custom_preds)

            # -------- Log Everything --------
            mlflow.log_param("model_name", name)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("optimal_threshold", optimal_threshold)
            mlflow.log_metric("custom_precision", precision)
            mlflow.log_metric("custom_recall", recall)
            mlflow.log_metric("custom_f1", f1)

            # Log model with signature + input example
            input_example = X_train.iloc[:1]

            mlflow.sklearn.log_model(
                model,
                name="model",
                registered_model_name="FraudDetectionModel",
                input_example=input_example
            )

            print(f"✅ {name}")
            print(f"   ROC-AUC: {roc_auc:.4f}")
            print(f"   Optimal Threshold: {optimal_threshold:.6f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")

            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_model_name = name
                best_model_object = model
                best_run_id = run.info.run_id
                best_threshold = optimal_threshold

    print(f"\n🏆 Best Model: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})")

    PERFORMANCE_THRESHOLD = 0.95

    if best_roc_auc >= PERFORMANCE_THRESHOLD:

        client = mlflow.MlflowClient()

        versions = client.search_model_versions(
            "name='FraudDetectionModel'"
        )

        best_version = None
        for v in versions:
            if v.run_id == best_run_id:
                best_version = v.version
                break

        if best_version:

            client.set_registered_model_alias(
                name="FraudDetectionModel",
                alias="production",
                version=best_version
            )

            # Store threshold inside MLflow model tags
            client.set_model_version_tag(
                name="FraudDetectionModel",
                version=best_version,
                key="production_threshold",
                value=str(best_threshold)
            )

            print(f"🚀 Model version {best_version} set as @production")
            print(f"🚀 Production threshold stored in registry")

        else:
            print("⚠ Could not determine best model version")

    else:
        print("⚠ Model performance below threshold. Not promoted.")

    print("🎉 Training completed successfully.")


if __name__ == "__main__":
    train_model()
