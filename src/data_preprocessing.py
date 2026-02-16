import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def preprocess():
    df = pd.read_csv("data/raw/creditcard.csv")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    processed_df = pd.DataFrame(X_resampled, columns=X.columns)
    processed_df["Class"] = y_resampled

    processed_df.to_csv("data/processed/processed_data.csv", index=False)

if __name__ == "__main__":
    preprocess()
