"""
05_fraud_classification.py
Financial Fraud Detection Analytics Platform
Student : Abotula Yaswanth Kumar
Roll No : A23126552191
Case Study 11.2 - Big Data Analytics Lab

Step 5: Supervised ML - Random Forest Classifier using PySpark MLlib
        Trains on 80% data, evaluates on 20%, handles class imbalance with weights
"""

import os
import json
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                   MulticlassClassificationEvaluator)

PARQUET_PATH = "results/parquet"
ML_PATH      = "results/ml"
os.makedirs(ML_PATH, exist_ok=True)

print("=" * 60)
print("  FINANCIAL FRAUD DETECTION - RANDOM FOREST CLASSIFIER")
print("  Student : Abotula Yaswanth Kumar | A23126552191")
print("=" * 60)

# ── Spark Session ─────────────────────────────────────────────
print("\n[1] Starting Spark session...")
spark = SparkSession.builder \
    .appName("FinancialFraudDetection_RF") \
    .config("spark.driver.memory", "3g") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# ── Load Data ─────────────────────────────────────────────────
print("[2] Loading Parquet data...")
df = spark.read.parquet(PARQUET_PATH)
total = df.count()
fraud = df.filter(F.col("Class") == 1).count()
legit = df.filter(F.col("Class") == 0).count()
print(f"    Total: {total:,}  |  Legitimate: {legit:,}  |  Fraud: {fraud:,}")

# ── Class weights to handle imbalance ────────────────────────
print("\n[3] Computing class weights for imbalance...")
ratio = legit / fraud
df = df.withColumn("classWeight",
    F.when(F.col("Class") == 1, ratio).otherwise(1.0))
print(f"    Fraud weight: {ratio:.2f}x  (balances the {ratio:.0f}:1 imbalance)")

# ── Feature assembly ──────────────────────────────────────────
print("\n[4] Assembling feature vector...")
feature_cols = [c for c in df.columns
                if c not in ['Class', 'fraud_label', 'amount_bucket',
                              'classWeight', 'features', 'scaledFeatures']]
print(f"    Using {len(feature_cols)} features")

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
scaler    = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                           withMean=True, withStd=True)

# ── Train/Test split ──────────────────────────────────────────
print("\n[5] Splitting data: 80% train / 20% test...")
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print(f"    Training records : {train_df.count():,}")
print(f"    Testing records  : {test_df.count():,}")

# ── Random Forest model ───────────────────────────────────────
print("\n[6] Training Random Forest Classifier (100 trees)...")
rf = RandomForestClassifier(
    labelCol="Class",
    featuresCol="scaledFeatures",
    weightCol="classWeight",
    numTrees=100,
    maxDepth=10,
    seed=42
)

pipeline = Pipeline(stages=[assembler, scaler, rf])
print("    Training in progress... (this takes ~2-3 minutes)")
model = pipeline.fit(train_df)
print("    Training complete!")

# ── Predictions ───────────────────────────────────────────────
print("\n[7] Running predictions on test set...")
predictions = model.transform(test_df)

# ── Evaluation ───────────────────────────────────────────────
print("\n[8] Evaluating model performance...")

binary_eval = BinaryClassificationEvaluator(
    labelCol="Class", rawPredictionCol="rawPrediction")
auc = binary_eval.evaluate(predictions,
      {binary_eval.metricName: "areaUnderROC"})

mc_eval = MulticlassClassificationEvaluator(
    labelCol="Class", predictionCol="prediction")

accuracy  = mc_eval.evaluate(predictions, {mc_eval.metricName: "accuracy"})
precision = mc_eval.evaluate(predictions, {mc_eval.metricName: "weightedPrecision"})
recall    = mc_eval.evaluate(predictions, {mc_eval.metricName: "weightedRecall"})
f1        = mc_eval.evaluate(predictions, {mc_eval.metricName: "f1"})

print(f"\n    ┌─────────────────────────────────┐")
print(f"    │  Random Forest Results          │")
print(f"    ├─────────────────────────────────┤")
print(f"    │  Accuracy  : {accuracy*100:>6.2f}%            │")
print(f"    │  Precision : {precision*100:>6.2f}%            │")
print(f"    │  Recall    : {recall*100:>6.2f}%            │")
print(f"    │  F1 Score  : {f1*100:>6.2f}%            │")
print(f"    │  AUC-ROC   : {auc:>8.4f}            │")
print(f"    └─────────────────────────────────┘")

# ── Confusion matrix ──────────────────────────────────────────
print("\n[9] Confusion Matrix:")
predictions.groupBy("Class", "prediction").count().orderBy("Class").show()

# ── Feature importance ────────────────────────────────────────
print("[10] Top 10 Most Important Features:")
rf_model      = model.stages[-1]
importances   = rf_model.featureImportances.toArray()
feat_imp      = sorted(zip(feature_cols, importances),
                       key=lambda x: x[1], reverse=True)
for i, (feat, imp) in enumerate(feat_imp[:10], 1):
    print(f"     {i:>2}. {feat:<16} {imp:.4f}")

# ── Save metrics ─────────────────────────────────────────────
metrics = {
    "model"               : "Random Forest Classifier (PySpark MLlib)",
    "num_trees"           : 100,
    "max_depth"           : 10,
    "total_records"       : total,
    "training_records"    : train_df.count(),
    "testing_records"     : test_df.count(),
    "feature_count"       : len(feature_cols),
    "accuracy_pct"        : round(accuracy  * 100, 2),
    "precision_pct"       : round(precision * 100, 2),
    "recall_pct"          : round(recall    * 100, 2),
    "f1_score_pct"        : round(f1        * 100, 2),
    "auc_roc"             : round(auc,  4),
    "top_features"        : [f for f, _ in feat_imp[:10]]
}
with open(f"{ML_PATH}/classification_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n    Metrics saved → {ML_PATH}/classification_metrics.json")
spark.stop()
print("\n[DONE] Fraud classification complete.")
print("=" * 60)
