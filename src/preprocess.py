import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(subset=["TotalCharges"], inplace=True)
df.drop("customerID", axis=1, inplace=True)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

bins = [0, 12, 24, 48, 72]
labels = ["0-12", "13-24", "25-48", "49-72"]
df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels, include_lowest=True)

df["charges_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)

y = df["Churn"]
X = df.drop("Churn", axis=1)

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

continuous_cols = ["tenure", "MonthlyCharges", "TotalCharges", "charges_per_tenure"]
scaler = StandardScaler()
X[continuous_cols] = scaler.fit_transform(X[continuous_cols])

feature_list = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
with open(os.path.join(OUTPUT_DIR, "feature_list.pkl"), "wb") as f:
    pickle.dump(feature_list, f)
with open(os.path.join(OUTPUT_DIR, "train_test_data.pkl"), "wb") as f:
    pickle.dump((X_train_sm, X_test, y_train_sm, y_test), f)

print(f"Original training set: {X_train.shape[0]} samples")
print(f"After SMOTE: {X_train_sm.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Features: {len(feature_list)}")
print(f"\nSaved: scaler.pkl, feature_list.pkl, train_test_data.pkl")
print("Preprocessing complete.")
