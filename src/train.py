import os
import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))

df = pd.read_csv("data/processed.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=params["train"]["test_size"],
    random_state=params["train"]["random_state"],
    stratify=y,
)

model = RandomForestClassifier(
    n_estimators=params["model"]["n_estimators"],
    max_depth=params["model"]["max_depth"],
    random_state=params["train"]["random_state"],
)
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
print("Model trained and saved")
