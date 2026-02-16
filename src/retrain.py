import os
import pandas as pd
from train import train_model

LOG_FILE = "predictions.log"

def retrain_if_needed():
    if not os.path.exists(LOG_FILE):
        return

    logs = pd.read_json(LOG_FILE, lines=True)

    if len(logs) > 5000:
        print("Retraining triggered...")
        train_model()

if __name__ == "__main__":
    retrain_if_needed()
