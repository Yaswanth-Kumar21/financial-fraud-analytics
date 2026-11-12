# Financial Fraud Detection Analytics Platform

**Student:** Abotula Yaswanth Kumar  
**Roll No:** A23126552191  
**Case Study:** 11.2 — Financial Fraud Detection Analytics Platform  
**Course:** Big Data Analytics Lab

---

## About the Project

An end-to-end Big Data & Machine Learning system for detecting financial fraud in large-scale credit card transaction data. Built with Apache Spark (PySpark) for distributed processing and Scikit-Learn for unsupervised anomaly detection.

**Dataset:** [Credit Card Fraud Detection — Kaggle (mlg-ulb)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
284,807 transactions | 492 fraud cases | 30 features (V1–V28 PCA + Time + Amount)

---

## Architecture

```
Raw Dataset (creditcard.csv)
        │
        ▼
01_dataset_exploration.py     ← Schema profiling, class distribution
        │
        ▼
02_data_preprocessing.py      ← Cleaning, dedup, normalization
        │
        ▼
03_spark_processing.py        ← Apache Spark ETL → Snappy Parquet
        │
        ▼
04_spark_sql_analytics.py     ← Distributed SQL queries & aggregations
        │
        ▼
05_fraud_classification.py    ← Random Forest (PySpark MLlib)
        │
        ▼
06_anomaly_detection.py       ← Isolation Forest (Scikit-Learn)
        │
        ▼
07_unified_threat_detection.py ← Severity scoring + HTML Dashboard
```

---

## Key Results

### Supervised Classification — Random Forest (PySpark MLlib)
| Metric    | Score  |
|-----------|--------|
| Accuracy  | ~99%   |
| Precision | ~99%   |
| Recall    | ~99%   |
| F1 Score  | ~99%   |
| AUC-ROC   | ~0.99  |

### Unsupervised Anomaly Detection — Isolation Forest (Scikit-Learn)
| Metric             | Value   |
|--------------------|---------|
| Baseline (Legit)   | 227,451 |
| Sample Tested      | 150,000 |
| Anomalies Detected | ~2,000+ |

---

## Tech Stack

| Component              | Technology              |
|------------------------|-------------------------|
| Distributed Processing | Apache Spark (PySpark)  |
| Storage Format         | Apache Parquet (Snappy) |
| SQL Analytics          | Spark SQL               |
| Supervised ML          | PySpark MLlib — Random Forest |
| Unsupervised ML        | Scikit-Learn — Isolation Forest |
| Dashboard              | HTML5 + Chart.js        |
| Language               | Python 3.x              |

---

## Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/Yaswanth-Kumar21/financial-fraud-analytics.git
cd financial-fraud-analytics

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place dataset
# Download creditcard.csv from Kaggle and put it in data/

# 5. Run the pipeline step by step
python src/01_dataset_exploration.py
python src/02_data_preprocessing.py
python src/03_spark_processing.py
python src/04_spark_sql_analytics.py
python src/05_fraud_classification.py
python src/06_anomaly_detection.py
python src/07_unified_threat_detection.py

# 6. Open dashboard
xdg-open dashboard.html
```

---

## Project Structure

```
financial-fraud-analytics/
├── data/                    # Dataset directory (CSV not committed)
├── src/
│   ├── 01_dataset_exploration.py
│   ├── 02_data_preprocessing.py
│   ├── 03_spark_processing.py
│   ├── 04_spark_sql_analytics.py
│   ├── 05_fraud_classification.py
│   ├── 06_anomaly_detection.py
│   └── 07_unified_threat_detection.py
├── results/
│   ├── analytics/           # Spark SQL query outputs
│   ├── ml/                  # Random Forest metrics
│   └── anomalies/           # Isolation Forest metrics
├── screenshots/             # Terminal and dashboard screenshots
├── dashboard.html           # Interactive SOC dashboard
├── requirements.txt
└── README.md
```

---

## References

- Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Apache Spark: [spark.apache.org](https://spark.apache.org)
- PySpark MLlib: [spark.apache.org/docs/latest/ml-guide.html](https://spark.apache.org/docs/latest/ml-guide.html)
