"""
02_data_preprocessing.py
Financial Fraud Detection Analytics Platform
Student : Abotula Yaswanth Kumar
Roll No : A23126552191
Case Study 11.2 - Big Data Analytics Lab

Step 2: Clean, deduplicate, normalize and save the dataset for Spark processing
"""

import pandas as pd
import numpy as np
import os
import json

DATA_PATH    = "data/creditcard.csv"
CLEANED_PATH = "results/cleaned_data"
RESULTS_PATH = "results/analytics"
os.makedirs(CLEANED_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

print("=" * 60)
print("  FINANCIAL FRAUD DETECTION - DATA PREPROCESSING")
print("  Student : Abotula Yaswanth Kumar | A23126552191")
print("=" * 60)

# ── Load ──────────────────────────────────────────────────────
print("\n[1] Loading raw dataset...")
df = pd.read_csv(DATA_PATH)
original_count = len(df)
print(f"    Records loaded: {original_count:,}")

# ── Null check ────────────────────────────────────────────────
print("\n[2] Checking for null values...")
total_nulls = int(df.isnull().sum().sum())
if total_nulls == 0:
    print("    No null values found — dataset is clean.")
else:
    df.fillna(df.median(numeric_only=True), inplace=True)
    print(f"    {total_nulls} nulls imputed with column medians.")

# ── Deduplication ─────────────────────────────────────────────
print("\n[3] Removing duplicate rows...")
dup_count = int(df.duplicated().sum())
if dup_count > 0:
    df.drop_duplicates(inplace=True)
    print(f"    Removed {dup_count:,} duplicates. Remaining: {len(df):,}")
else:
    print(f"    No duplicates found.")

# ── Normalization ─────────────────────────────────────────────
print("\n[4] Normalizing Time and Amount features...")
df['Amount_norm'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
df['Time_norm']   = (df['Time']   - df['Time'].mean())   / df['Time'].std()
df.drop(columns=['Time', 'Amount'], inplace=True)
print("    Time  → Time_norm  (z-score normalized)")
print("    Amount → Amount_norm (z-score normalized)")

# ── Feature summary ───────────────────────────────────────────
features = [c for c in df.columns if c != 'Class']
print(f"\n[5] Total features after preprocessing: {len(features)}")

counts = df['Class'].value_counts()
print(f"\n[6] Final class distribution:")
print(f"    Legitimate: {counts[0]:,} | Fraud: {counts[1]:,}")

# ── Save cleaned CSV ──────────────────────────────────────────
cleaned_file = f"{CLEANED_PATH}/creditcard_cleaned.csv"
df.to_csv(cleaned_file, index=False)
size_mb = os.path.getsize(cleaned_file) / 1e6
print(f"\n[7] Cleaned data saved → {cleaned_file}  ({size_mb:.2f} MB)")

# ── Save report ───────────────────────────────────────────────
report = {
    "original_records"            : original_count,
    "cleaned_records"             : int(len(df)),
    "duplicates_removed"          : dup_count,
    "null_values_imputed"         : total_nulls,
    "features_after_preprocessing": len(features),
    "normalization_applied"       : ["Time → Time_norm", "Amount → Amount_norm"],
    "legitimate_count"            : int(counts[0]),
    "fraud_count"                 : int(counts[1])
}
with open(f"{RESULTS_PATH}/preprocessing_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"    Report saved → {RESULTS_PATH}/preprocessing_report.json")
print("\n[DONE] Preprocessing complete.")
print("=" * 60)
