import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(f"Shape: {df.shape}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
if df.isnull().sum().sum() == 0:
    print("  (none)")
print(f"\nChurn rate: {df['Churn'].value_counts(normalize=True)['Yes']:.2%}")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print(f"\nTotalCharges blanks converted to NaN: {df['TotalCharges'].isnull().sum()}")
df.dropna(subset=["TotalCharges"], inplace=True)
print(f"Shape after dropping nulls: {df.shape}")

# Chart 1 — Churn rate by Contract type
fig, ax = plt.subplots(figsize=(8, 5))
churn_by_contract = df.groupby("Contract")["Churn"].apply(lambda x: (x == "Yes").mean())
churn_by_contract.plot(kind="bar", color=["#2196F3", "#4CAF50", "#FF9800"], edgecolor="black", ax=ax)
ax.set_title("Churn Rate by Contract Type", fontsize=14, fontweight="bold")
ax.set_ylabel("Churn Rate")
ax.set_xlabel("Contract Type")
ax.set_ylim(0, 1)
for i, v in enumerate(churn_by_contract):
    ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontweight="bold")
plt.xticks(rotation=0)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "churn_by_contract.png"), dpi=150)
plt.close()
print("\nSaved: churn_by_contract.png")

# Chart 2 — Churn rate by tenure group
bins = [0, 12, 24, 48, 72]
labels = ["0-12", "13-24", "25-48", "49-72"]
df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels, include_lowest=True)
fig, ax = plt.subplots(figsize=(8, 5))
churn_by_tenure = df.groupby("tenure_group", observed=False)["Churn"].apply(lambda x: (x == "Yes").mean())
churn_by_tenure.plot(kind="bar", color=["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71"], edgecolor="black", ax=ax)
ax.set_title("Churn Rate by Tenure Group (months)", fontsize=14, fontweight="bold")
ax.set_ylabel("Churn Rate")
ax.set_xlabel("Tenure Group")
ax.set_ylim(0, 1)
for i, v in enumerate(churn_by_tenure):
    ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontweight="bold")
plt.xticks(rotation=0)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "churn_by_tenure.png"), dpi=150)
plt.close()
print("Saved: churn_by_tenure.png")

# Chart 3 — Monthly charges distribution by churn status
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df, palette={"No": "#2196F3", "Yes": "#e74c3c"}, ax=ax)
ax.set_title("Monthly Charges Distribution by Churn Status", fontsize=14, fontweight="bold")
ax.set_ylabel("Monthly Charges ($)")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "monthly_charges_boxplot.png"), dpi=150)
plt.close()
print("Saved: monthly_charges_boxplot.png")

print("\nEDA complete.")
