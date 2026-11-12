"""
01_dataset_exploration.py
Financial Fraud Detection Analytics Platform
Student : Abotula Yaswanth Kumar
Roll No : A23126552191
Case Study 11.2 - Big Data Analytics Lab

Step 1: Explore the raw dataset - schema, class distribution, basic statistics
Dataset : Credit Card Fraud Detection (Kaggle - mlg-ulb/creditcardfraud)
Records : 284,807 transactions | 30 features (V1-V28 PCA + Time + Amount)
"""

import pandas as pd
import numpy as np
import os
import json

DATA_PATH    = "data/creditcard.csv"
RESULTS_PATH = "results/analytics"
os.makedirs(RESULTS_PATH, exist_ok=True)

print("=" * 60)
print("  FINANCIAL FRAUD DETECTION - DATASET EXPLORATION")
print("  Student : Abotula Yaswanth Kumar | A23126552191")
print("=" * 60)

# ── Load ──────────────────────────────────────────────────────
print("\n[1] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"    Rows    : {len(df):,}")
print(f"    Columns : {len(df.columns)}")
print(f"    Memory  : {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

# ── Schema ────────────────────────────────────────────────────
print("\n[2] Schema & Data Types:")
print(f"    {'Column':<14} {'DType':<12} {'Non-Null':>10} {'Null%':>8}")
print(f"    {'-'*48}")
for col in df.columns:
    null_pct = df[col].isnull().sum() / len(df) * 100
    print(f"    {col:<14} {str(df[col].dtype):<12} "
          f"{df[col].notnull().sum():>10,} {null_pct:>7.2f}%")

# ── Class distribution ────────────────────────────────────────
print("\n[3] Class Distribution (Fraud vs Legitimate):")
counts = df['Class'].value_counts()
pct    = df['Class'].value_counts(normalize=True) * 100
print(f"    Legitimate (0) : {counts[0]:>7,}  ({pct[0]:.3f}%)")
print(f"    Fraud      (1) : {counts[1]:>7,}  ({pct[1]:.3f}%)")
print(f"    Imbalance Ratio: {counts[0]/counts[1]:.1f} : 1")

# ── Stats ─────────────────────────────────────────────────────
print("\n[4] Key Feature Statistics (Time, Amount):")
print(df[['Time', 'Amount']].describe().to_string())

# ── Amount by class ───────────────────────────────────────────
print("\n[5] Transaction Amount by Class:")
for cls, label in [(0, 'Legitimate'), (1, 'Fraud')]:
    s = df[df['Class'] == cls]['Amount']
    print(f"    {label}: Min=${s.min():.2f}  Max=${s.max():.2f}  "
          f"Mean=${s.mean():.2f}  Median=${s.median():.2f}")

# ── Save summary ──────────────────────────────────────────────
summary = {
    "total_records"    : int(len(df)),
    "total_features"   : int(len(df.columns)),
    "legitimate_count" : int(counts[0]),
    "fraud_count"      : int(counts[1]),
    "fraud_percentage" : float(round(pct[1], 3)),
    "imbalance_ratio"  : float(round(counts[0] / counts[1], 1)),
    "null_values"      : int(df.isnull().sum().sum()),
    "amount_max"       : float(round(df['Amount'].max(), 2)),
    "amount_mean"      : float(round(df['Amount'].mean(), 2))
}
with open(f"{RESULTS_PATH}/dataset_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n[6] Summary saved → {RESULTS_PATH}/dataset_summary.json")
print("\n[DONE] Dataset exploration complete.")
print("=" * 60)
