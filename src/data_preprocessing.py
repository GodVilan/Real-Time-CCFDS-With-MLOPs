import pandas as pd
from sklearn.preprocessing import StandardScaler
import os


def preprocess():

    print("Starting preprocessing...")

    # -----------------------------
    # Ensure raw data exists
    # -----------------------------
    raw_path = "data/raw/creditcard.csv"

    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path}"
        )

    df = pd.read_csv(raw_path)

    print(f"Raw data shape: {df.shape}")
    print(f"Fraud ratio: {df['Class'].mean():.4%}")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # -----------------------------
    # Feature Scaling
    # -----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------
    # Reassemble DataFrame
    # NOTE: SMOTE is applied ONLY during training
    # (in src/train.py) to prevent data leakage.
    # -----------------------------
    processed_df = pd.DataFrame(X_scaled, columns=X.columns)
    processed_df["Class"] = y.values

    # -----------------------------
    # Save processed data
    # -----------------------------
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    output_path = os.path.join(processed_dir, "processed_data.csv")
    processed_df.to_csv(output_path, index=False)

    print(f"Processed data saved at {output_path}")
    print(f"Processed data shape: {processed_df.shape}")


if __name__ == "__main__":
    preprocess()
