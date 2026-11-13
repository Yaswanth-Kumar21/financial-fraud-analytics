"""
06_anomaly_detection.py
Financial Fraud Detection Analytics Platform
Student : Abotula Yaswanth Kumar
Roll No : A23126552191
Case Study 11.2 - Big Data Analytics Lab

Step 6: Unsupervised Anomaly Detection — Algorithm 3
        Algorithm 3 : Isolation Forest (Scikit-Learn)
        Trained on LEGITIMATE transactions only → detects novel fraud patterns
"""

import pandas as pd
import numpy as np
import os
import json
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

CLEANED_CSV  = "results/cleaned_data/ieee_fraud_cleaned.csv"
ANOMALY_PATH = "results/anomalies"
os.makedirs(ANOMALY_PATH, exist_ok=True)

print("=" * 65)
print("  FINANCIAL FRAUD DETECTION - ISOLATION FOREST ANOMALY")
print("  Algorithm 3 : Isolation Forest (Scikit-Learn)")
print("  Student : Abotula Yaswanth Kumar | A23126552191")
print("=" * 65)

# ── Load cleaned data ─────────────────────────────────────────
print("\n[1] Loading cleaned IEEE-CIS dataset...")
df = pd.read_csv(CLEANED_CSV)
print(f"    Total records: {len(df):,}")
print(f"    Columns      : {len(df.columns)}")

# Target column is 'isFraud' in cleaned CSV
target_col = 'isFraud' if 'isFraud' in df.columns else 'Class'
feature_cols = [c for c in df.columns if c != target_col]

# Keep only numeric columns for Isolation Forest
numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
print(f"    Numeric features used: {len(numeric_cols)}")

X = df[numeric_cols].values
y = df[target_col].values

# ── Baseline: legitimate transactions only ─────────────────────
print("\n[2] Extracting legitimate baseline for training...")
legit_mask  = y == 0
X_legit     = X[legit_mask]
legit_count = len(X_legit)
print(f"    Legitimate records: {legit_count:,}")

# ── Train Isolation Forest ─────────────────────────────────────
print("\n[3] Training Isolation Forest on legitimate baseline...")
fraud_rate = (y == 1).sum() / len(y)
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=round(float(fraud_rate), 4),
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_legit)
print(f"    Training complete! (contamination={round(float(fraud_rate),4)})")

# ── Predict on a 100K sample ───────────────────────────────────
sample_size = min(100000, len(X))
print(f"\n[4] Running anomaly detection on {sample_size:,} transaction sample...")
np.random.seed(42)
sample_idx = np.random.choice(len(X), size=sample_size, replace=False)
X_sample   = X[sample_idx]
y_sample   = y[sample_idx]

raw_preds  = iso_forest.predict(X_sample)
scores     = iso_forest.score_samples(X_sample)

# -1 = anomaly → 1 (fraud), 1 = normal → 0
y_pred        = (raw_preds == -1).astype(int)
anomaly_count = int(y_pred.sum())
print(f"    Anomalies detected: {anomaly_count:,} out of {sample_size:,}")

# ── Metrics ───────────────────────────────────────────────────
print("\n[5] Evaluation Metrics:")
precision = precision_score(y_sample, y_pred, zero_division=0)
recall    = recall_score(y_sample, y_pred, zero_division=0)
f1        = f1_score(y_sample, y_pred, zero_division=0)
accuracy  = accuracy_score(y_sample, y_pred)

print(f"\n    ┌─────────────────────────────────────┐")
print(f"    │  Isolation Forest Results           │")
print(f"    ├─────────────────────────────────────┤")
print(f"    │  Accuracy  : {accuracy*100:>6.2f}%              │")
print(f"    │  Precision : {precision*100:>6.2f}%              │")
print(f"    │  Recall    : {recall*100:>6.2f}%              │")
print(f"    │  F1 Score  : {f1*100:>6.2f}%              │")
print(f"    │  Anomalies : {anomaly_count:>6,}               │")
print(f"    └─────────────────────────────────────┘")

# ── Save anomalies ─────────────────────────────────────────────
print("\n[6] Saving detected anomalies...")
anomaly_mask = y_pred == 1
anomaly_df   = pd.DataFrame(X_sample[anomaly_mask], columns=numeric_cols)
anomaly_df['anomaly_score'] = scores[anomaly_mask]
anomaly_df['true_label']    = y_sample[anomaly_mask]
anomaly_df.to_csv(f"{ANOMALY_PATH}/detected_anomalies.csv", index=False)
print(f"    Saved {len(anomaly_df):,} anomalies → {ANOMALY_PATH}/detected_anomalies.csv")

# ── Save metrics ───────────────────────────────────────────────
metrics = {
    "model"                     : "Isolation Forest (Scikit-Learn)",
    "algorithm_number"          : 3,
    "baseline_training_records" : int(legit_count),
    "sample_records_tested"     : int(sample_size),
    "anomalies_detected"        : int(anomaly_count),
    "accuracy_pct"              : round(accuracy   * 100, 2),
    "precision_pct"             : round(precision  * 100, 2),
    "recall_pct"                : round(recall     * 100, 2),
    "f1_score_pct"              : round(f1         * 100, 2),
    "contamination_param"       : round(float(fraud_rate), 4),
    "n_estimators"              : 100,
}
with open(f"{ANOMALY_PATH}/anomaly_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"    Metrics saved → {ANOMALY_PATH}/anomaly_metrics.json")
print("\n[DONE] Anomaly detection (Isolation Forest) complete.")
print("=" * 65)
