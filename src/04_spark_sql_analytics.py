"""
04_spark_sql_analytics.py
Financial Fraud Detection Analytics Platform
Student : Abotula Yaswanth Kumar
Roll No : A23126552191
Case Study 11.2 - Big Data Analytics Lab

Step 4: Spark SQL Analytics
        - Distributed SQL queries on IEEE-CIS Parquet data
        - Fraud by product, card type, amount bucket, hour
        - Save analytics JSON outputs for dashboard
"""

import os
import json
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

PARQUET_PATH = "results/parquet"
RESULTS_PATH = "results/analytics"
os.makedirs(RESULTS_PATH, exist_ok=True)

print("=" * 65)
print("  FINANCIAL FRAUD DETECTION - SPARK SQL ANALYTICS")
print("  Student : Abotula Yaswanth Kumar | A23126552191")
print("=" * 65)

# ── Spark Session ─────────────────────────────────────────────
print("\n[1] Starting Spark session...")
spark = SparkSession.builder \
    .appName("IEEEFraudDetection_SQL") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# ── Load Parquet ──────────────────────────────────────────────
print("\n[2] Loading Parquet data...")
df = spark.read.parquet(PARQUET_PATH)
df.createOrReplaceTempView("transactions")
total = df.count()
print(f"    Loaded {total:,} rows, {len(df.columns)} columns")

# ── Query 1: Overall traffic summary ─────────────────────────
print("\n[3] Query 1: Overall traffic summary...")
q1 = spark.sql("""
    SELECT
        COUNT(*)                                   AS total_transactions,
        SUM(CASE WHEN Class=0 THEN 1 ELSE 0 END)  AS legitimate,
        SUM(CASE WHEN Class=1 THEN 1 ELSE 0 END)  AS fraud_cases,
        ROUND(AVG(TransactionAmt), 2)              AS avg_amount,
        ROUND(SUM(TransactionAmt), 2)              AS total_volume,
        ROUND(SUM(CASE WHEN Class=1 THEN TransactionAmt ELSE 0 END), 2) AS fraud_volume,
        ROUND(SUM(Class) * 100.0 / COUNT(*), 3)   AS fraud_rate_pct
    FROM transactions
""")
q1.show(truncate=False)
q1_row = q1.collect()[0]

traffic_summary = {
    "total_transactions" : int(q1_row["total_transactions"]),
    "legitimate_count"   : int(q1_row["legitimate"]),
    "fraud_count"        : int(q1_row["fraud_cases"]),
    "avg_transaction_amt": float(q1_row["avg_amount"]),
    "total_volume_usd"   : float(q1_row["total_volume"]),
    "fraud_volume_usd"   : float(q1_row["fraud_volume"]),
    "fraud_rate_pct"     : float(q1_row["fraud_rate_pct"]),
}
with open(f"{RESULTS_PATH}/traffic_summary.json", "w") as f:
    json.dump(traffic_summary, f, indent=2)
print(f"    Saved → {RESULTS_PATH}/traffic_summary.json")

# ── Query 2: Fraud by product code ───────────────────────────
print("\n[4] Query 2: Fraud rate by ProductCD...")
q2 = spark.sql("""
    SELECT
        ProductCD,
        COUNT(*)                              AS total,
        SUM(Class)                            AS fraud,
        ROUND(SUM(Class)*100.0/COUNT(*), 2)  AS fraud_rate_pct,
        ROUND(AVG(TransactionAmt), 2)         AS avg_amount
    FROM transactions
    GROUP BY ProductCD
    ORDER BY fraud_rate_pct DESC
""")
q2.show(truncate=False)

# ── Query 3: Fraud by amount bucket ──────────────────────────
print("\n[5] Query 3: Fraud by transaction amount bucket...")
q3 = spark.sql("""
    SELECT
        amount_bucket,
        COUNT(*)                              AS total,
        SUM(Class)                            AS fraud,
        ROUND(SUM(Class)*100.0/COUNT(*), 2)  AS fraud_rate_pct
    FROM transactions
    GROUP BY amount_bucket
    ORDER BY
        CASE amount_bucket
            WHEN 'LOW' THEN 1
            WHEN 'MEDIUM' THEN 2
            WHEN 'HIGH' THEN 3
            ELSE 4 END
""")
q3.show(truncate=False)
q3_rows = q3.collect()
amount_dist = {
    r["amount_bucket"]: {
        "total": int(r["total"]),
        "fraud": int(r["fraud"]),
        "fraud_rate_pct": float(r["fraud_rate_pct"])
    }
    for r in q3_rows
}
with open(f"{RESULTS_PATH}/amount_distribution.json", "w") as f:
    json.dump(amount_dist, f, indent=2)
print(f"    Saved → {RESULTS_PATH}/amount_distribution.json")

# ── Query 4: Fraud by hour of day ────────────────────────────
print("\n[6] Query 4: Fraud pattern by hour of day...")
q4 = spark.sql("""
    SELECT
        hour_of_day,
        COUNT(*)           AS total,
        SUM(Class)         AS fraud,
        ROUND(SUM(Class)*100.0/COUNT(*), 2) AS fraud_rate_pct
    FROM transactions
    GROUP BY hour_of_day
    ORDER BY hour_of_day
""")
q4.show(24, truncate=False)

# ── Query 5: Feature averages by class ───────────────────────
print("\n[7] Query 5: Feature averages by fraud class...")
q5 = spark.sql("""
    SELECT
        Class,
        ROUND(AVG(TransactionAmt), 2)      AS avg_amount,
        ROUND(AVG(TransactionAmt_norm), 4) AS avg_amount_norm,
        COUNT(*)                           AS count
    FROM transactions
    GROUP BY Class
    ORDER BY Class
""")
q5.show(truncate=False)
q5_rows = q5.collect()
feat_avgs = {
    str(int(r["Class"])): {
        "avg_amount"      : float(r["avg_amount"]),
        "avg_amount_norm" : float(r["avg_amount_norm"]),
        "count"           : int(r["count"]),
    }
    for r in q5_rows
}
with open(f"{RESULTS_PATH}/feature_averages_by_class.json", "w") as f:
    json.dump(feat_avgs, f, indent=2)
print(f"    Saved → {RESULTS_PATH}/feature_averages_by_class.json")

# ── Query 6: Top fraud hours ──────────────────────────────────
print("\n[8] Query 6: Top 5 fraud-heavy hours...")
q6 = spark.sql("""
    SELECT hour_of_day, SUM(Class) AS fraud_count
    FROM transactions
    GROUP BY hour_of_day
    ORDER BY fraud_count DESC
    LIMIT 5
""")
q6.show(truncate=False)

# ── Save combined analytics summary ───────────────────────────
analytics_summary = {
    "total_transactions"    : traffic_summary["total_transactions"],
    "legitimate_count"      : traffic_summary["legitimate_count"],
    "fraud_count"           : traffic_summary["fraud_count"],
    "fraud_percentage"      : traffic_summary["fraud_rate_pct"],
    "legitimate_percentage" : round(100 - traffic_summary["fraud_rate_pct"], 3),
    "avg_transaction_amt"   : traffic_summary["avg_transaction_amt"],
    "total_volume_usd"      : traffic_summary["total_volume_usd"],
    "fraud_volume_usd"      : traffic_summary["fraud_volume_usd"],
    "dataset"               : "IEEE-CIS Fraud Detection (Vesta Corporation)",
    "source"                : "https://www.kaggle.com/competitions/ieee-fraud-detection/data",
}
with open(f"{RESULTS_PATH}/analytics_summary.json", "w") as f:
    json.dump(analytics_summary, f, indent=2)
print(f"\n    Analytics summary saved → {RESULTS_PATH}/analytics_summary.json")

spark.stop()
print("\n[DONE] Spark SQL analytics complete.")
print("=" * 65)
