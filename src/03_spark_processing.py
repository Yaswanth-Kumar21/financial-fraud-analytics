"""
03_spark_processing.py
Financial Fraud Detection Analytics Platform
Student : Abotula Yaswanth Kumar
Roll No : A23126552191
Case Study 11.2 - Big Data Analytics Lab

Step 3: Apache Spark ETL
        - Load cleaned CSV into Spark
        - Add derived columns (fraud_label, amount_bucket, hour_of_day)
        - Save as Snappy-compressed Parquet for downstream ML
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

CLEANED_CSV  = "results/cleaned_data/ieee_fraud_cleaned.csv"
PARQUET_PATH = "results/parquet"
os.makedirs(PARQUET_PATH, exist_ok=True)

print("=" * 65)
print("  FINANCIAL FRAUD DETECTION - SPARK ETL PROCESSING")
print("  Student : Abotula Yaswanth Kumar | A23126552191")
print("=" * 65)

# ── Spark Session ─────────────────────────────────────────────
print("\n[1] Starting Apache Spark session...")
spark = SparkSession.builder \
    .appName("IEEEFraudDetection_ETL") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
print(f"    Spark version : {spark.version}")

# ── Load CSV ──────────────────────────────────────────────────
print("\n[2] Loading cleaned CSV into Spark...")
df = spark.read.csv(CLEANED_CSV, header=True, inferSchema=True)
total = df.count()
print(f"    Total rows    : {total:,}")
print(f"    Total columns : {len(df.columns)}")

fraud = df.filter(F.col("isFraud") == 1).count()
legit = df.filter(F.col("isFraud") == 0).count()
print(f"    Legitimate    : {legit:,}")
print(f"    Fraud         : {fraud:,}")

# ── Rename target column to 'Class' for compatibility ────────
print("\n[3] Renaming isFraud → Class for pipeline compatibility...")
df = df.withColumnRenamed("isFraud", "Class")

# ── Derived columns ───────────────────────────────────────────
print("\n[4] Adding derived feature columns...")

# fraud label string
df = df.withColumn("fraud_label",
    F.when(F.col("Class") == 1, "FRAUD").otherwise("LEGITIMATE"))

# Amount bucket
df = df.withColumn("amount_bucket",
    F.when(F.col("TransactionAmt") < 50,   "LOW")
     .when(F.col("TransactionAmt") < 500,  "MEDIUM")
     .when(F.col("TransactionAmt") < 5000, "HIGH")
     .otherwise("VERY_HIGH"))

# Hour of day from TransactionDT (seconds offset, modulo day)
df = df.withColumn("hour_of_day",
    (F.col("TransactionDT") / 3600 % 24).cast(IntegerType()))

# Weekend flag (TransactionDT / seconds in a week)
df = df.withColumn("day_of_week",
    ((F.col("TransactionDT") / 86400) % 7).cast(IntegerType()))

print(f"    Derived columns added: fraud_label, amount_bucket, hour_of_day, day_of_week")

# ── Summary stats ─────────────────────────────────────────────
print("\n[5] Amount bucket distribution:")
df.groupBy("amount_bucket", "Class").count() \
  .orderBy("amount_bucket", "Class") \
  .show(truncate=False)

print("[6] Fraud by hour of day (top 5 fraud hours):")
df.filter(F.col("Class") == 1) \
  .groupBy("hour_of_day").count() \
  .orderBy(F.col("count").desc()) \
  .show(5)

# ── Write to Parquet ──────────────────────────────────────────
print(f"\n[7] Writing to Snappy-compressed Parquet → {PARQUET_PATH} ...")
df.write.mode("overwrite") \
        .option("compression", "snappy") \
        .parquet(PARQUET_PATH)

# verify
pq_df    = spark.read.parquet(PARQUET_PATH)
pq_count = pq_df.count()
print(f"    Parquet rows  : {pq_count:,}")
print(f"    Parquet schema: {len(pq_df.columns)} columns")

spark.stop()
print("\n[DONE] Spark ETL complete.")
print("=" * 65)
