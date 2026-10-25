import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/dataset.csv")

# Data cleaning
df = df.dropna().drop_duplicates()

# Feature engineering: scale all features
X = df.drop(columns=["target"])
y = df["target"]
X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
X_scaled["target"] = y.values

X_scaled.to_csv("data/processed.csv", index=False)
print("Processed rows:", len(X_scaled))
