"""
06_anomaly_detection.py
Financial Fraud Detection Analytics Platform
Student : Abotula Yaswanth Kumar
Roll No : A23126552191
Case Study 11.2 - Big Data Analytics Lab

Step 6: Unsupervised anomaly detection using Isolation Forest (Scikit-Learn)
        Trained only on clean LEGITIMATE transactions to detect novel/zero-day fraud
"""

import pandas as pd
import numpy as np
import os
import json
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

CLEANED_CSV  = "results/cleaned_data/creditcard_cleaned.csv"
ANOMALY_PATH = "results/anomalies"
os.makedirs(ANOMALY_PATH, exist_ok=True)

print("=" * 60)
print("  FINANCIAL FRAUD DETECTION - ISOLATION FOREST ANOMALY")
print("  Student : Abotula Yaswanth Kumar | A23126552191")
print("=" * 60)

# ── Load cleaned data ─────────────────────────────────────────
print("\n[1] Loading cleaned dataset...")
df = pd.read_csv(CLEANED_CSV)
print(f"    Total records: {len(df):,}")

feature_cols = [c for c in df.columns if c != 'Class']
X = df[feature_cols].values
y = df['Class'].values

# ── Baseline: train only on LEGITIMATE transactions ───────────
print("\n[2] Extracting clean baseline (legitimate transactions only)...")
legit_mask  = df['Class'] == 0
X_legit     = X[legit_mask]
legit_count = len(X_legit)
print(f"    Legitimate baseline records: {legit_count:,}")

# ── Train Isolation Forest ────────────────────────────────────
print("\n[3] Training Isolation Forest on legitimate baseline...")
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.002,   # ~0.17% fraud rate in dataset
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_legit)
print("    Training complete!")

# ── Predict on a sample of 150,000 records ────────────────────
print("\n[4] Running anomaly detection on 150,000 transaction sample...")
np.random.seed(42)
sample_idx = np.random.choice(len(X), size=150000, replace=False)
X_sample   = X[sample_idx]
y_sample   = y[sample_idx]

# IsolationForest returns: -1 = anomaly, 1 = normal
raw_preds  = iso_forest.predict(X_sample)
scores     = iso_forest.score_samples(X_sample)

# Convert: -1 → 1 (fraud), 1 → 0 (legitimate)
y_pred     = (raw_preds == -1).astype(int)

anomaly_count = int(y_pred.sum())
print(f"    Anomalies detected: {anomaly_count:,} out of 150,000")

# ── Metrics ───────────────────────────────────────────────────
print("\n[5] Evaluation Metrics:")
precision = precision_score(y_sample, y_pred, zero_division=0)
recall    = recall_score(y_sample, y_pred, zero_division=0)
f1        = f1_score(y_sample, y_pred, zero_division=0)
accuracy  = accuracy_score(y_sample, y_pred)

print(f"\n    ┌─────────────────────────────────┐")
print(f"    │  Isolation Forest Results       │")
print(f"    ├─────────────────────────────────┤")
print(f"    │  Accuracy  : {accuracy*100:>6.2f}%            │")
print(f"    │  Precision : {precision*100:>6.2f}%            │")
print(f"    │  Recall    : {recall*100:>6.2f}%            │")
print(f"    │  F1 Score  : {f1*100:>6.2f}%            │")
print(f"    │  Anomalies : {anomaly_count:>6,}             │")
print(f"    └─────────────────────────────────┘")

# ── Save detected anomalies ───────────────────────────────────
print("\n[6] Saving detected anomalies...")
anomaly_mask = y_pred == 1
anomaly_df   = pd.DataFrame(X_sample[anomaly_mask], columns=feature_cols)
anomaly_df['anomaly_score'] = scores[anomaly_mask]
anomaly_df['true_label']    = y_sample[anomaly_mask]
anomaly_df.to_csv(f"{ANOMALY_PATH}/detected_anomalies.csv", index=False)
print(f"    Saved {len(anomaly_df):,} anomalies → {ANOMALY_PATH}/detected_anomalies.csv")

# ── Save metrics ──────────────────────────────────────────────
metrics = {
    "model"                  : "Isolation Forest (Scikit-Learn)",
    "baseline_training_records": int(legit_count),
    "sample_records_tested"  : 150000,
    "anomalies_detected"     : int(anomaly_count),
    "accuracy_pct"           : round(accuracy  * 100, 2),
    "precision_pct"          : round(precision * 100, 2),
    "recall_pct"             : round(recall    * 100, 2),
    "f1_score_pct"           : round(f1        * 100, 2),
    "contamination_param"    : 0.002,
    "n_estimators"           : 100
}
with open(f"{ANOMALY_PATH}/anomaly_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"    Metrics saved → {ANOMALY_PATH}/anomaly_metrics.json")
print("\n[DONE] Anomaly detection complete.")
print("=" * 60)
