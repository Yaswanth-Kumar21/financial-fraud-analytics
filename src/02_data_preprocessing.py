"""
02_data_preprocessing.py
Financial Fraud Detection Analytics Platform
Student : Abotula Yaswanth Kumar
Roll No : A23126552191
Case Study 11.2 - Big Data Analytics Lab

Step 2: Data Preprocessing
        - Join transaction + identity tables
        - Handle nulls (median/mode imputation)
        - Encode categoricals
        - Normalize TransactionAmt
        - Remove duplicates
        - Save cleaned CSV for downstream steps
"""

import pandas as pd
import numpy as np
import os
import json

TRANSACTION_CSV  = "ieee-fraud-detection/train_transaction.csv"
IDENTITY_CSV     = "ieee-fraud-detection/train_identity.csv"
CLEANED_DIR      = "results/cleaned_data"
RESULTS_PATH     = "results/analytics"
os.makedirs(CLEANED_DIR,  exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

print("=" * 65)
print("  FINANCIAL FRAUD DETECTION - DATA PREPROCESSING")
print("  Student : Abotula Yaswanth Kumar | A23126552191")
print("=" * 65)

# ── Load ───────────────────────────────────────────────────────
print("\n[1] Loading datasets...")
txn = pd.read_csv(TRANSACTION_CSV)
idn = pd.read_csv(IDENTITY_CSV)
print(f"    Transactions : {len(txn):,} rows × {len(txn.columns)} cols")
print(f"    Identity     : {len(idn):,} rows × {len(idn.columns)} cols")

# ── Merge ──────────────────────────────────────────────────────
print("\n[2] Merging on TransactionID (LEFT JOIN)...")
df = txn.merge(idn, on='TransactionID', how='left')
print(f"    Merged shape : {df.shape}")
before_rows = len(df)

# ── Drop columns with >70% nulls ───────────────────────────────
print("\n[3] Dropping columns with >70% null values...")
null_pct    = df.isnull().mean()
drop_cols   = null_pct[null_pct > 0.70].index.tolist()
# Always keep target and key identifiers
keep_always = ['isFraud', 'TransactionID', 'TransactionAmt', 'TransactionDT']
drop_cols   = [c for c in drop_cols if c not in keep_always]
df.drop(columns=drop_cols, inplace=True)
print(f"    Dropped {len(drop_cols)} high-null columns")
print(f"    Remaining columns: {len(df.columns)}")

# ── Remove duplicates ──────────────────────────────────────────
print("\n[4] Removing duplicate rows...")
df.drop_duplicates(inplace=True)
after_rows = len(df)
print(f"    Removed {before_rows - after_rows:,} duplicates")
print(f"    Remaining rows: {after_rows:,}")

# ── Identify column types ──────────────────────────────────────
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in ['isFraud', 'TransactionID']]
print(f"\n[5] Column types:")
print(f"    Numeric    : {len(num_cols)} columns")
print(f"    Categorical: {len(cat_cols)} columns  {cat_cols[:5]}...")

# ── Fill numeric nulls with median ────────────────────────────
print("\n[6] Imputing numeric nulls with median...")
for col in num_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

# ── Encode categoricals (label encoding) ──────────────────────
print("\n[7] Label-encoding categorical columns...")
for col in cat_cols:
    df[col].fillna('unknown', inplace=True)
    df[col] = df[col].astype('category').cat.codes

# ── Normalize TransactionAmt ───────────────────────────────────
print("\n[8] Normalizing TransactionAmt (log1p + z-score)...")
df['TransactionAmt_log'] = np.log1p(df['TransactionAmt'])
amt_mean = df['TransactionAmt_log'].mean()
amt_std  = df['TransactionAmt_log'].std()
df['TransactionAmt_norm'] = (df['TransactionAmt_log'] - amt_mean) / amt_std
print(f"    Mean after norm: {df['TransactionAmt_norm'].mean():.4f}")
print(f"    Std after norm : {df['TransactionAmt_norm'].std():.4f}")

# ── Fill any remaining nulls with 0 ──────────────────────────
print("\n[9] Filling any remaining nulls with 0...")
df.fillna(0, inplace=True)
remaining_nulls = df.isnull().sum().sum()
print(f"    Final null count: {remaining_nulls}")

# ── Class distribution after cleaning ─────────────────────────
counts = df['isFraud'].value_counts()
print(f"\n[10] Final class distribution:")
print(f"    Legitimate (0) : {counts[0]:,}")
print(f"    Fraud      (1) : {counts[1]:,}")
print(f"    Fraud Rate     : {counts[1]/len(df)*100:.3f}%")

# ── Save cleaned data ──────────────────────────────────────────
print(f"\n[11] Saving cleaned dataset...")
out_path = f"{CLEANED_DIR}/ieee_fraud_cleaned.csv"
df.drop(columns=['TransactionID', 'TransactionAmt_log'], errors='ignore').to_csv(out_path, index=False)
print(f"    Saved → {out_path}")
print(f"    Final shape: {df.shape}")

# ── Save preprocessing report ─────────────────────────────────
report = {
    "original_transaction_rows": int(len(txn)),
    "original_identity_rows"   : int(len(idn)),
    "after_merge_rows"         : int(before_rows),
    "after_cleaning_rows"      : int(after_rows),
    "final_columns"            : int(len(df.columns)),
    "dropped_high_null_cols"   : int(len(drop_cols)),
    "duplicates_removed"       : int(before_rows - after_rows),
    "categorical_encoded"      : int(len(cat_cols)),
    "numeric_imputed"          : int(len(num_cols)),
    "remaining_nulls"          : int(remaining_nulls),
    "fraud_count_final"        : int(counts[1]),
    "legitimate_count_final"   : int(counts[0]),
    "fraud_rate_pct"           : float(round(counts[1]/len(df)*100, 3)),
}
with open(f"{RESULTS_PATH}/preprocessing_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"    Report saved → {RESULTS_PATH}/preprocessing_report.json")
print("\n[DONE] Data preprocessing complete.")
print("=" * 65)
