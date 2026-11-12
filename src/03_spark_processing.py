"""
03_spark_processing.py
Financial Fraud Detection Analytics Platform
Student : Abotula Yaswanth Kumar
Roll No : A23126552191
Case Study 11.2 - Big Data Analytics Lab

Step 3: Apache Spark distributed ETL - load cleaned CSV, convert to Parquet
        Uses Hadoop distributed partitioning (part-00000 shards + _SUCCESS markers)
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

CLEANED_CSV  = "results/cleaned_data/creditcard_cleaned.csv"
PARQUET_PATH = "results/parquet"
os.makedirs(PARQUET_PATH, exist_ok=True)

print("=" * 60)
print("  FINANCIAL FRAUD DETECTION - SPARK ETL & PARQUET")
print("  Student : Abotula Yaswanth Kumar | A23126552191")
print("=" * 60)

# ── Spark Session ─────────────────────────────────────────────
print("\n[1] Starting Apache Spark session...")
spark = SparkSession.builder \
    .appName("FinancialFraudDetection_ETL") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
print(f"    Spark version : {spark.version}")
print(f"    App name      : FinancialFraudDetection_ETL")

# ── Load cleaned CSV ──────────────────────────────────────────
print(f"\n[2] Loading cleaned CSV into Spark DataFrame...")
df = spark.read.csv(CLEANED_CSV, header=True, inferSchema=True)
total = df.count()
print(f"    Total records  : {total:,}")
print(f"    Total columns  : {len(df.columns)}")
print(f"    Partitions     : {df.rdd.getNumPartitions()}")

print("\n[3] Schema:")
df.printSchema()

# ── Basic Spark transformations ───────────────────────────────
print("[4] Running distributed transformations...")

# Add fraud label as string for readability
df = df.withColumn("fraud_label",
    F.when(F.col("Class") == 1, "FRAUD").otherwise("LEGITIMATE"))

# Add transaction size bucket
df = df.withColumn("amount_bucket",
    F.when(F.col("Amount_norm") < -0.5, "LOW")
     .when(F.col("Amount_norm") < 0.5,  "MEDIUM")
     .otherwise("HIGH"))

print("    Added: fraud_label, amount_bucket columns")
print(f"    Final schema columns: {len(df.columns)}")

# ── Show sample ───────────────────────────────────────────────
print("\n[5] Sample records:")
df.select("V1", "V2", "Amount_norm", "Class", "fraud_label", "amount_bucket") \
  .show(5, truncate=False)

# ── Write Parquet ─────────────────────────────────────────────
print(f"[6] Writing to Snappy-compressed Parquet partitions...")
df.write \
  .mode("overwrite") \
  .option("compression", "snappy") \
  .partitionBy("Class") \
  .parquet(PARQUET_PATH)

# Check output
parquet_files = []
for root, dirs, files in os.walk(PARQUET_PATH):
    for file in files:
        parquet_files.append(os.path.join(root, file))

print(f"    Parquet files written: {len(parquet_files)}")
print(f"    Output path: {PARQUET_PATH}/")
for f in parquet_files[:6]:
    print(f"      {f.replace(PARQUET_PATH+'/', '')}")

# ── Verify Parquet read-back ──────────────────────────────────
print("\n[7] Verifying Parquet read-back...")
df_verify = spark.read.parquet(PARQUET_PATH)
print(f"    Records verified: {df_verify.count():,}")
print(f"    Columns         : {len(df_verify.columns)}")

spark.stop()
print("\n[DONE] Spark ETL complete. Data stored as Parquet.")
print("=" * 60)
