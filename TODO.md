# 📋 Project TODO — Financial Fraud Detection Analytics Platform
**Student:** Abotula Yaswanth Kumar | Roll No: A23126552191
**Case Study:** 11.2 — Big Data Analytics Lab

---

## 🎯 Goal
Replace old 144MB Kaggle credit card dataset with the new **IEEE-CIS Fraud Detection dataset (1.3 GB)**
and upgrade ML to use **3 algorithms** as required by sir.

---

## ✅ Dataset
- [x] Downloaded IEEE-CIS Fraud Detection dataset (1.3 GB total)
  - `ieee-fraud-detection/train_transaction.csv` — 652 MB, 590K rows, 394 cols
  - `ieee-fraud-detection/train_identity.csv`   — 26 MB,  144K rows, 41 cols
  - `ieee-fraud-detection/test_transaction.csv` — 585 MB
  - `ieee-fraud-detection/test_identity.csv`    — 25 MB
- [x] Virtual environment created (`venv/` with Python 3.11)

---

## 🔧 Scripts to Rewrite (all 7 steps)

| # | Script | Status | Notes |
|---|--------|--------|-------|
| 1 | `src/01_dataset_exploration.py` | ⬜ TODO | New columns: isFraud, TransactionAmt, card*, id_* etc |
| 2 | `src/02_data_preprocessing.py` | ⬜ TODO | Join transaction + identity on TransactionID |
| 3 | `src/03_spark_processing.py`   | ⬜ TODO | Spark ETL on merged dataset → Parquet |
| 4 | `src/04_spark_sql_analytics.py`| ⬜ TODO | SQL queries on new feature set |
| 5 | `src/05_fraud_classification.py`| ⬜ TODO | Add **3 ML models**: Random Forest + GBT + Logistic Regression |
| 6 | `src/06_anomaly_detection.py`  | ⬜ TODO | Isolation Forest on new features |
| 7 | `src/07_unified_threat_detection.py` | ⬜ TODO | Combine all 3 model outputs + new dashboard |

---

## 🤖 ML Algorithms (3 required by sir)

| # | Algorithm | Library | Type |
|---|-----------|---------|------|
| 1 | **Random Forest Classifier** | PySpark MLlib | Supervised |
| 2 | **Gradient Boosted Trees (GBT)** | PySpark MLlib | Supervised |
| 3 | **Isolation Forest** | Scikit-Learn | Unsupervised / Anomaly Detection |

---

## 📦 Dependencies to Install

- [x] `kagglehub` — installed
- [ ] `pyspark==3.5.0`
- [ ] `pandas>=2.2`
- [ ] `numpy>=2.0`
- [ ] `scikit-learn>=1.5`
- [ ] `matplotlib>=3.8`
- [ ] `seaborn>=0.13`

---

## 📂 GitHub Tasks

- [ ] Add GitHub remote URL (share your repo URL)
- [ ] Update README.md with new dataset info
- [ ] Push all updated source files
- [ ] Update Google Classroom submission link with new dataset reference

---

## 📊 Dashboard Updates

- [ ] Show all 3 ML model comparison chart
- [ ] Update dataset info (IEEE-CIS, 1.3 GB, 590K rows)
- [ ] Update KPI cards with new fraud stats

---

## 🚀 Execution Order

```
Step 1 → 01_dataset_exploration.py     ← explore IEEE-CIS merged dataset
Step 2 → 02_data_preprocessing.py      ← clean + join transaction+identity
Step 3 → 03_spark_processing.py        ← Spark ETL → Parquet
Step 4 → 04_spark_sql_analytics.py     ← Spark SQL analytics
Step 5 → 05_fraud_classification.py    ← RF + GBT classifiers
Step 6 → 06_anomaly_detection.py       ← Isolation Forest
Step 7 → 07_unified_threat_detection.py← Combined dashboard
```

---

## 📝 Notes
- Dataset path: `ieee-fraud-detection/` (inside project folder)
- Target column: `isFraud` (was `Class` in old dataset)
- Amount column: `TransactionAmt` (was `Amount`)
- No PCA features — real features: card info, email domain, device, C/D/M/V columns
- Friend's reference: CICIDS2017 + Random Forest → 99.55% accuracy
- Our target: IEEE-CIS + RF + GBT + IsolationForest → aim for 98%+
