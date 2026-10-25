import json
import os
import joblib
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))

df = pd.read_csv("data/processed.csv")
X = df.drop(columns=["target"])
y = df["target"]

# Same split as train.py (same test_size and random_state)
_, X_test, _, y_test = train_test_split(
    X, y,
    test_size=params["train"]["test_size"],
    random_state=params["train"]["random_state"],
    stratify=y,
)

model = joblib.load("models/model.pkl")
preds = model.predict(X_test)

metrics = {
    "accuracy": round(accuracy_score(y_test, preds), 4),
    "precision": round(precision_score(y_test, preds), 4),
    "recall": round(recall_score(y_test, preds), 4),
    "f1_score": round(f1_score(y_test, preds), 4),
}

os.makedirs("metrics", exist_ok=True)
with open("metrics/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
print(metrics)
