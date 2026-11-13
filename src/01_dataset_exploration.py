"""
01_dataset_exploration.py
Financial Fraud Detection Analytics Platform
Student : Abotula Yaswanth Kumar
Roll No : A23126552191
Case Study 11.2 - Big Data Analytics Lab

Step 1: Explore the IEEE-CIS Fraud Detection dataset
Dataset : IEEE-CIS Fraud Detection (Kaggle / Vesta Corporation)
Files   : train_transaction.csv (590K rows, 394 cols)
          train_identity.csv    (144K rows, 41 cols)
Total   : ~1.3 GB across all files
"""

import pandas as pd
import numpy as np
import os
import json

TRANSACTION_CSV = "ieee-fraud-detection/train_transaction.csv"
IDENTITY_CSV    = "ieee-fraud-detection/train_identity.csv"
RESULTS_PATH    = "results/analytics"
os.makedirs(RESULTS_PATH, exist_ok=True)

print("=" * 65)
print("  FINANCIAL FRAUD DETECTION - DATASET EXPLORATION")
print("  Student : Abotula Yaswanth Kumar | A23126552191")
print("  Dataset : IEEE-CIS Fraud Detection (Vesta Corporation)")
print("=" * 65)

# ── Load transaction table ────────────────────────────────────
print("\n[1] Loading train_transaction.csv ...")
txn = pd.read_csv(TRANSACTION_CSV)
print(f"    Rows    : {len(txn):,}")
print(f"    Columns : {len(txn.columns)}")
print(f"    Memory  : {txn.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# ── Load identity table ───────────────────────────────────────
print("\n[2] Loading train_identity.csv ...")
idn = pd.read_csv(IDENTITY_CSV)
print(f"    Rows    : {len(idn):,}")
print(f"    Columns : {len(idn.columns)}")
print(f"    Memory  : {idn.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# ── Merge on TransactionID ────────────────────────────────────
print("\n[3] Merging transaction + identity on TransactionID (LEFT JOIN)...")
df = txn.merge(idn, on='TransactionID', how='left')
print(f"    Merged rows    : {len(df):,}")
print(f"    Merged columns : {len(df.columns)}")
print(f"    Memory         : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# ── Class distribution ────────────────────────────────────────
print("\n[4] Class Distribution (isFraud):")
counts = df['isFraud'].value_counts()
pct    = df['isFraud'].value_counts(normalize=True) * 100
print(f"    Legitimate (0) : {counts[0]:>7,}  ({pct[0]:.3f}%)")
print(f"    Fraud      (1) : {counts[1]:>7,}  ({pct[1]:.3f}%)")
print(f"    Imbalance Ratio: {counts[0]/counts[1]:.1f} : 1")

# ── Schema overview ────────────────────────────────────────────
print("\n[5] Column Groups Overview:")
txn_cols = [c for c in txn.columns if c not in ['TransactionID', 'isFraud']]
idn_cols = [c for c in idn.columns if c != 'TransactionID']

col_groups = {
    "Transaction (amount/time)" : [c for c in txn_cols if c in ['TransactionDT','TransactionAmt','ProductCD']],
    "Card features"             : [c for c in txn_cols if c.startswith('card')],
    "Address/Distance"          : [c for c in txn_cols if c.startswith('addr') or c.startswith('dist')],
    "Email domains"             : [c for c in txn_cols if 'email' in c.lower()],
    "C-features (count)"        : [c for c in txn_cols if c.startswith('C')],
    "D-features (time delta)"   : [c for c in txn_cols if c.startswith('D')],
    "M-features (match)"        : [c for c in txn_cols if c.startswith('M')],
    "V-features (Vesta)"        : [c for c in txn_cols if c.startswith('V')],
    "Identity features"         : idn_cols,
}
for grp, cols in col_groups.items():
    print(f"    {grp:<35} : {len(cols)} cols")

# ── Null analysis ──────────────────────────────────────────────
print("\n[6] Null Value Summary (top 10 highest null %):")
null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
for col, pct_val in null_pct.head(10).items():
    print(f"    {col:<20} : {pct_val:.1f}% null")

# ── Transaction amount stats ───────────────────────────────────
print("\n[7] TransactionAmt Statistics:")
print(df[['TransactionAmt']].describe().to_string())

print("\n[8] TransactionAmt by Class:")
for cls, label in [(0, 'Legitimate'), (1, 'Fraud')]:
    s = df[df['isFraud'] == cls]['TransactionAmt']
    print(f"    {label}: Min=${s.min():.2f}  Max=${s.max():.2f}  "
          f"Mean=${s.mean():.2f}  Median=${s.median():.2f}")

# ── Product code distribution ──────────────────────────────────
print("\n[9] ProductCD Distribution:")
print(df.groupby('ProductCD')['isFraud'].agg(['count','sum'])
        .rename(columns={'count':'Total','sum':'Fraud'})
        .assign(FraudRate=lambda x: (x['Fraud']/x['Total']*100).round(2))
        .sort_values('Total', ascending=False)
        .to_string())

# ── Save summary ───────────────────────────────────────────────
summary = {
    "dataset"          : "IEEE-CIS Fraud Detection (Vesta Corporation / Kaggle)",
    "total_records"    : int(len(txn)),
    "identity_records" : int(len(idn)),
    "merged_records"   : int(len(df)),
    "total_features"   : int(len(df.columns)),
    "legitimate_count" : int(counts[0]),
    "fraud_count"      : int(counts[1]),
    "fraud_percentage" : float(round(pct[1], 3)),
    "legitimate_percentage": float(round(pct[0], 3)),
    "imbalance_ratio"  : float(round(counts[0] / counts[1], 1)),
    "null_values_total": int(df.isnull().sum().sum()),
    "amount_max"       : float(round(df['TransactionAmt'].max(), 2)),
    "amount_mean"      : float(round(df['TransactionAmt'].mean(), 2)),
    "amount_median"    : float(round(df['TransactionAmt'].median(), 2)),
    "total_transactions": int(len(txn)),
}
with open(f"{RESULTS_PATH}/dataset_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n[10] Summary saved → {RESULTS_PATH}/dataset_summary.json")
print("\n[DONE] Dataset exploration complete.")
print("=" * 65)
