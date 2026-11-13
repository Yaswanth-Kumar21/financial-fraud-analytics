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
| 1 | `src/01_dataset_exploration.py` | ✅ DONE | 590K rows, 434 cols, 3.499% fraud rate |
| 2 | `src/02_data_preprocessing.py` | ✅ DONE | Merged, 228 cols, 0 nulls after cleaning |
| 3 | `src/03_spark_processing.py`   | ✅ DONE | Parquet written, 230 cols incl. derived |
| 4 | `src/04_spark_sql_analytics.py`| ✅ DONE | 6 SQL queries, fraud peaks at hrs 5-9 AM |
| 5 | `src/05_fraud_classification.py`| ✅ DONE | RF: 84% acc, 88.75% F1 / GBT: 86.88% acc, 90.56% F1 |
| 6 | `src/06_anomaly_detection.py`  | ✅ DONE | Isolation Forest: 93.95% acc, 4175 anomalies |
| 7 | `src/07_unified_threat_detection.py` | ✅ DONE | Dashboard with 3-model comparison generated |

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
- [x] `pyspark==3.5.0`
- [x] `pandas>=2.2`
- [x] `numpy>=2.0`
- [x] `scikit-learn>=1.5`
- [x] `matplotlib>=3.8`
- [x] `seaborn>=0.13`

---

## 📂 GitHub Tasks

- [x] Add GitHub remote URL — https://github.com/Yaswanth-Kumar21/financial-fraud-analytics
- [x] Update README.md with new dataset info
- [x] Push all updated source files
- [ ] Update Google Classroom submission link with new dataset reference

---

## 📊 Dashboard Updates

- [x] Show all 3 ML model comparison chart
- [x] Update dataset info (IEEE-CIS, 1.3 GB, 590K rows)
- [x] Update KPI cards with new fraud stats

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
