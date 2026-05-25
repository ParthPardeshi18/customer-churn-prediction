import pickle
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    roc_curve, confusion_matrix
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

with open(os.path.join(OUTPUT_DIR, "train_test_data.pkl"), "rb") as f:
    X_train, X_test, y_train, y_test = pickle.load(f)
with open(os.path.join(OUTPUT_DIR, "feature_list.pkl"), "rb") as f:
    feature_list = pickle.load(f)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        use_label_encoder=False, eval_metric="logloss", random_state=42
    ),
}

print("=" * 70)
print("MODEL COMPARISON (5-Fold Cross-Validation)")
print("=" * 70)

results = []
for name, model in models.items():
    auc_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
    f1_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
    prec_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="precision")
    rec_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="recall")
    results.append({
        "Model": name,
        "AUC": f"{auc_scores.mean():.4f}",
        "F1": f"{f1_scores.mean():.4f}",
        "Precision": f"{prec_scores.mean():.4f}",
        "Recall": f"{rec_scores.mean():.4f}",
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print("\n" + "=" * 70)
print("TRAINING BEST MODEL (XGBoost) ON FULL TRAINING SET")
print("=" * 70)

best_model = models["XGBoost"]
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

test_auc = roc_auc_score(y_test, y_prob)
test_f1 = f1_score(y_test, y_pred)
test_prec = precision_score(y_test, y_pred)
test_rec = recall_score(y_test, y_pred)

print(f"Test AUC:       {test_auc:.4f}")
print(f"Test F1:        {test_f1:.4f}")
print(f"Test Precision: {test_prec:.4f}")
print(f"Test Recall:    {test_rec:.4f}")

# Chart 1 — ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"XGBoost (AUC = {test_auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — XGBoost", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=150)
plt.close()
print("\nSaved: roc_curve.png")

# Chart 2 — Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix — XGBoost", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print("Saved: confusion_matrix.png")

# Chart 3 — Top 15 feature importance
importances = best_model.feature_importances_
feat_imp = pd.Series(importances, index=feature_list).sort_values(ascending=True).tail(15)
fig, ax = plt.subplots(figsize=(10, 7))
feat_imp.plot(kind="barh", color="#2196F3", edgecolor="black", ax=ax)
ax.set_title("Top 15 Feature Importances — XGBoost", fontsize=14, fontweight="bold")
ax.set_xlabel("Importance")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
plt.close()
print("Saved: feature_importance.png")

# Save model
with open(os.path.join(OUTPUT_DIR, "churn_model.pkl"), "wb") as f:
    pickle.dump(best_model, f)
print("Saved: churn_model.pkl")

print("\nTraining complete.")
