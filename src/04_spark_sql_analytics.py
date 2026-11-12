"""
04_spark_sql_analytics.py
Financial Fraud Detection Analytics Platform
Student : Abotula Yaswanth Kumar
Roll No : A23126552191
Case Study 11.2 - Big Data Analytics Lab

Step 4: Distributed Spark SQL analytics - aggregations, fraud patterns,
        traffic ratios, amount distributions, hourly trends
"""

import os
import json
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

PARQUET_PATH = "results/parquet"
ANALYTICS    = "results/analytics"
os.makedirs(ANALYTICS, exist_ok=True)

print("=" * 60)
print("  FINANCIAL FRAUD DETECTION - SPARK SQL ANALYTICS")
print("  Student : Abotula Yaswanth Kumar | A23126552191")
print("=" * 60)

# ── Spark Session ─────────────────────────────────────────────
print("\n[1] Starting Spark session...")
spark = SparkSession.builder \
    .appName("FinancialFraudDetection_SQL") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# ── Load Parquet ──────────────────────────────────────────────
print("[2] Loading Parquet data...")
df = spark.read.parquet(PARQUET_PATH)
df.createOrReplaceTempView("transactions")
total = df.count()
print(f"    Total records loaded: {total:,}")

# ── Q1: Fraud vs Legitimate distribution ─────────────────────
print("\n[3] Query 1 — Fraud vs Legitimate distribution:")
q1 = spark.sql("""
    SELECT fraud_label,
           COUNT(*)                              AS total_transactions,
           ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 3) AS percentage
    FROM transactions
    GROUP BY fraud_label
    ORDER BY total_transactions DESC
""")
q1.show()
q1_data = [row.asDict() for row in q1.collect()]
with open(f"{ANALYTICS}/traffic_summary.json", "w") as f:
    json.dump(q1_data, f, indent=2)

# ── Q2: Amount bucket distribution ───────────────────────────
print("[4] Query 2 — Transaction amount bucket breakdown:")
q2 = spark.sql("""
    SELECT amount_bucket,
           fraud_label,
           COUNT(*) AS count,
           ROUND(AVG(Amount_norm), 4) AS avg_norm_amount
    FROM transactions
    GROUP BY amount_bucket, fraud_label
    ORDER BY amount_bucket, fraud_label
""")
q2.show()
q2_data = [row.asDict() for row in q2.collect()]
with open(f"{ANALYTICS}/amount_distribution.json", "w") as f:
    json.dump(q2_data, f, indent=2)

# ── Q3: Top 10 most extreme V1 fraud patterns ────────────────
print("[5] Query 3 — Top fraud records by V1 feature anomaly:")
q3 = spark.sql("""
    SELECT ROUND(V1, 4) AS V1,
           ROUND(V2, 4) AS V2,
           ROUND(V3, 4) AS V3,
           ROUND(Amount_norm, 4) AS Amount_norm,
           fraud_label
    FROM transactions
    WHERE fraud_label = 'FRAUD'
    ORDER BY ABS(V1) DESC
    LIMIT 10
""")
q3.show(truncate=False)

# ── Q4: Feature averages by class ────────────────────────────
print("[6] Query 4 — Average feature values by class:")
q4 = spark.sql("""
    SELECT fraud_label,
           ROUND(AVG(V1),  4) AS avg_V1,
           ROUND(AVG(V2),  4) AS avg_V2,
           ROUND(AVG(V3),  4) AS avg_V3,
           ROUND(AVG(V4),  4) AS avg_V4,
           ROUND(AVG(V14), 4) AS avg_V14,
           ROUND(AVG(Amount_norm), 4) AS avg_Amount_norm
    FROM transactions
    GROUP BY fraud_label
""")
q4.show(truncate=False)
q4_data = [row.asDict() for row in q4.collect()]
with open(f"{ANALYTICS}/feature_averages_by_class.json", "w") as f:
    json.dump(q4_data, f, indent=2)

# ── Q5: Time-based fraud frequency (binned) ──────────────────
print("[7] Query 5 — Fraud frequency by time bin:")
q5 = spark.sql("""
    SELECT CAST(FLOOR(Time_norm * 10) AS INT) AS time_bin,
           SUM(CASE WHEN Class=1 THEN 1 ELSE 0 END) AS fraud_count,
           SUM(CASE WHEN Class=0 THEN 1 ELSE 0 END) AS legit_count
    FROM transactions
    GROUP BY time_bin
    ORDER BY time_bin
    LIMIT 15
""")
q5.show()

# ── Summary stats ─────────────────────────────────────────────
fraud_count = df.filter(F.col("Class") == 1).count()
legit_count = df.filter(F.col("Class") == 0).count()

analytics_summary = {
    "total_transactions"   : total,
    "legitimate_count"     : legit_count,
    "fraud_count"          : fraud_count,
    "fraud_percentage"     : round(fraud_count / total * 100, 3),
    "legitimate_percentage": round(legit_count / total * 100, 3)
}
with open(f"{ANALYTICS}/analytics_summary.json", "w") as f:
    json.dump(analytics_summary, f, indent=2)

print(f"\n[8] Analytics results saved to {ANALYTICS}/")
print(f"    Total: {total:,} | Legitimate: {legit_count:,} | Fraud: {fraud_count:,}")
print(f"    Fraud rate: {fraud_count/total*100:.3f}%")

spark.stop()
print("\n[DONE] Spark SQL analytics complete.")
print("=" * 60)
