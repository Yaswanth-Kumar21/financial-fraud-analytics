"""
05_fraud_classification.py
Financial Fraud Detection Analytics Platform
Student : Abotula Yaswanth Kumar
Roll No : A23126552191
Case Study 11.2 - Big Data Analytics Lab

Step 5: Supervised ML Classification — 2 Algorithms
        Algorithm 1 : Random Forest Classifier      (PySpark MLlib)
        Algorithm 2 : Gradient Boosted Trees (GBT)  (PySpark MLlib)
        Both trained on 80/20 split with class imbalance weighting
"""

import os
import json
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import (RandomForestClassifier,
                                       GBTClassifier)
from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                   MulticlassClassificationEvaluator)

PARQUET_PATH = "results/parquet"
ML_PATH      = "results/ml"
os.makedirs(ML_PATH, exist_ok=True)

print("=" * 65)
print("  FINANCIAL FRAUD DETECTION - ML CLASSIFICATION")
print("  Algorithms: Random Forest + Gradient Boosted Trees")
print("  Student : Abotula Yaswanth Kumar | A23126552191")
print("=" * 65)

# ── Spark Session ─────────────────────────────────────────────
print("\n[1] Starting Spark session...")
spark = SparkSession.builder \
    .appName("IEEEFraudDetection_ML") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# ── Load Parquet ──────────────────────────────────────────────
print("\n[2] Loading Parquet data...")
df = spark.read.parquet(PARQUET_PATH)
total = df.count()
fraud = df.filter(F.col("Class") == 1).count()
legit = df.filter(F.col("Class") == 0).count()
print(f"    Total: {total:,}  |  Legitimate: {legit:,}  |  Fraud: {fraud:,}")

# ── Class weights ─────────────────────────────────────────────
print("\n[3] Computing class weights for imbalance handling...")
ratio = legit / fraud
df = df.withColumn("classWeight",
    F.when(F.col("Class") == 1, ratio).otherwise(1.0))
print(f"    Fraud weight: {ratio:.2f}x  ({legit:,} / {fraud:,})")

# ── Feature selection ─────────────────────────────────────────
print("\n[4] Selecting numeric feature columns...")
# Use only numeric columns — skip non-feature cols
skip_cols = {'Class', 'isFraud', 'TransactionID', 'classWeight',
             'fraud_label', 'amount_bucket', 'features', 'scaledFeatures'}
feature_cols = [
    f.name for f in df.schema.fields
    if f.name not in skip_cols
    and str(f.dataType) in ('DoubleType()', 'FloatType()', 'IntegerType()',
                            'LongType()', 'ShortType()')
]
print(f"    Using {len(feature_cols)} numeric features")

# ── Feature assembly ──────────────────────────────────────────
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features",
                            handleInvalid="skip")
scaler    = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                           withMean=True, withStd=True)

# ── Train/Test split ──────────────────────────────────────────
print("\n[5] Train/Test split 80% / 20%...")
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
train_count = train_df.count()
test_count  = test_df.count()
print(f"    Training : {train_count:,}")
print(f"    Testing  : {test_count:,}")

# ─────────────────────────────────────────────────────────────
# ALGORITHM 1 — RANDOM FOREST
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  ALGORITHM 1 : RANDOM FOREST CLASSIFIER")
print("─" * 60)
print("[6a] Training Random Forest (100 trees, maxDepth=8)...")

rf = RandomForestClassifier(
    labelCol="Class",
    featuresCol="scaledFeatures",
    weightCol="classWeight",
    numTrees=100,
    maxDepth=8,
    seed=42
)
rf_pipeline = Pipeline(stages=[assembler, scaler, rf])
rf_model    = rf_pipeline.fit(train_df)
rf_preds    = rf_model.transform(test_df)
print("    Training complete!")

# Evaluate RF
binary_eval = BinaryClassificationEvaluator(
    labelCol="Class", rawPredictionCol="rawPrediction")
mc_eval = MulticlassClassificationEvaluator(
    labelCol="Class", predictionCol="prediction")

rf_auc  = binary_eval.evaluate(rf_preds, {binary_eval.metricName: "areaUnderROC"})
rf_acc  = mc_eval.evaluate(rf_preds, {mc_eval.metricName: "accuracy"})
rf_prec = mc_eval.evaluate(rf_preds, {mc_eval.metricName: "weightedPrecision"})
rf_rec  = mc_eval.evaluate(rf_preds, {mc_eval.metricName: "weightedRecall"})
rf_f1   = mc_eval.evaluate(rf_preds, {mc_eval.metricName: "f1"})

print(f"\n    ┌─────────────────────────────────────┐")
print(f"    │  Random Forest Results              │")
print(f"    ├─────────────────────────────────────┤")
print(f"    │  Accuracy  : {rf_acc*100:>6.2f}%              │")
print(f"    │  Precision : {rf_prec*100:>6.2f}%              │")
print(f"    │  Recall    : {rf_rec*100:>6.2f}%              │")
print(f"    │  F1 Score  : {rf_f1*100:>6.2f}%              │")
print(f"    │  AUC-ROC   : {rf_auc:>8.4f}              │")
print(f"    └─────────────────────────────────────┘")

# RF feature importances
rf_stage   = rf_model.stages[-1]
rf_imps    = rf_stage.featureImportances.toArray()
feat_imp   = sorted(zip(feature_cols, rf_imps), key=lambda x: x[1], reverse=True)
print("\n    Top 10 Important Features (RF):")
for i, (feat, imp) in enumerate(feat_imp[:10], 1):
    print(f"    {i:>2}. {feat:<20} {imp:.4f}")

# ─────────────────────────────────────────────────────────────
# ALGORITHM 2 — GRADIENT BOOSTED TREES
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  ALGORITHM 2 : GRADIENT BOOSTED TREES (GBT)")
print("─" * 60)
print("[6b] Training GBT Classifier (50 iterations, maxDepth=5)...")

gbt = GBTClassifier(
    labelCol="Class",
    featuresCol="scaledFeatures",
    weightCol="classWeight",
    maxIter=50,
    maxDepth=5,
    seed=42
)
gbt_pipeline = Pipeline(stages=[assembler, scaler, gbt])
gbt_model    = gbt_pipeline.fit(train_df)
gbt_preds    = gbt_model.transform(test_df)
print("    Training complete!")

# Evaluate GBT
gbt_auc  = binary_eval.evaluate(gbt_preds, {binary_eval.metricName: "areaUnderROC"})
gbt_acc  = mc_eval.evaluate(gbt_preds, {mc_eval.metricName: "accuracy"})
gbt_prec = mc_eval.evaluate(gbt_preds, {mc_eval.metricName: "weightedPrecision"})
gbt_rec  = mc_eval.evaluate(gbt_preds, {mc_eval.metricName: "weightedRecall"})
gbt_f1   = mc_eval.evaluate(gbt_preds, {mc_eval.metricName: "f1"})

print(f"\n    ┌─────────────────────────────────────┐")
print(f"    │  Gradient Boosted Trees Results     │")
print(f"    ├─────────────────────────────────────┤")
print(f"    │  Accuracy  : {gbt_acc*100:>6.2f}%              │")
print(f"    │  Precision : {gbt_prec*100:>6.2f}%              │")
print(f"    │  Recall    : {gbt_rec*100:>6.2f}%              │")
print(f"    │  F1 Score  : {gbt_f1*100:>6.2f}%              │")
print(f"    │  AUC-ROC   : {gbt_auc:>8.4f}              │")
print(f"    └─────────────────────────────────────┘")

# ── Confusion matrix for best model ──────────────────────────
best_preds = gbt_preds if gbt_f1 >= rf_f1 else rf_preds
print("\n[7] Confusion Matrix (best model):")
best_preds.groupBy("Class", "prediction").count().orderBy("Class").show()

# ── Save metrics ──────────────────────────────────────────────
metrics = {
    "dataset"              : "IEEE-CIS Fraud Detection (Vesta Corporation)",
    "total_records"        : int(total),
    "training_records"     : int(train_count),
    "testing_records"      : int(test_count),
    "feature_count"        : int(len(feature_cols)),
    "random_forest": {
        "model"       : "Random Forest Classifier (PySpark MLlib)",
        "num_trees"   : 100,
        "max_depth"   : 8,
        "accuracy_pct": round(rf_acc  * 100, 2),
        "precision_pct": round(rf_prec * 100, 2),
        "recall_pct"  : round(rf_rec  * 100, 2),
        "f1_score_pct": round(rf_f1   * 100, 2),
        "auc_roc"     : round(rf_auc,  4),
        "top_features": [f for f, _ in feat_imp[:10]],
    },
    "gradient_boosted_trees": {
        "model"       : "Gradient Boosted Trees (PySpark MLlib)",
        "max_iter"    : 50,
        "max_depth"   : 5,
        "accuracy_pct": round(gbt_acc  * 100, 2),
        "precision_pct": round(gbt_prec * 100, 2),
        "recall_pct"  : round(gbt_rec  * 100, 2),
        "f1_score_pct": round(gbt_f1   * 100, 2),
        "auc_roc"     : round(gbt_auc,  4),
    },
    # flat fields for dashboard compatibility
    "accuracy_pct" : round(rf_acc  * 100, 2),
    "precision_pct": round(rf_prec * 100, 2),
    "recall_pct"   : round(rf_rec  * 100, 2),
    "f1_score_pct" : round(rf_f1   * 100, 2),
    "auc_roc"      : round(rf_auc,  4),
    "num_trees"    : 100,
    "max_depth"    : 8,
}
with open(f"{ML_PATH}/classification_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n    Metrics saved → {ML_PATH}/classification_metrics.json")
spark.stop()
print("\n[DONE] ML classification complete (RF + GBT).")
print("=" * 65)
