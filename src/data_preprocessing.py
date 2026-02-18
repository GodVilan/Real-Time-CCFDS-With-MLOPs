import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os


def preprocess():

    print("🚀 Starting preprocessing...")

    # -----------------------------
    # Ensure raw data exists
    # -----------------------------
    raw_path = "data/raw/creditcard.csv"

    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path}"
        )

    df = pd.read_csv(raw_path)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # -----------------------------
    # Scaling
    # -----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------
    # SMOTE
    # -----------------------------
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    processed_df = pd.DataFrame(X_resampled, columns=X.columns)
    processed_df["Class"] = y_resampled

    # -----------------------------
    # Create processed directory
    # -----------------------------
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    output_path = os.path.join(processed_dir, "processed_data.csv")

    processed_df.to_csv(output_path, index=False)

    print(f"✅ Processed data saved at {output_path}")


if __name__ == "__main__":
    preprocess()
