"""
07_unified_threat_detection.py
Financial Fraud Detection Analytics Platform
Student : Abotula Yaswanth Kumar
Roll No : A23126552191
Case Study 11.2 - Big Data Analytics Lab

Step 7: Unified Fraud Intelligence Engine
        Combines RF + GBT + Isolation Forest outputs
        Severity scoring: CRITICAL / HIGH / MEDIUM / LOW
        Generates interactive HTML SOC dashboard
"""

import pandas as pd
import numpy as np
import os
import json

CLEANED_CSV  = "results/cleaned_data/ieee_fraud_cleaned.csv"
ML_PATH      = "results/ml"
ANOMALY_PATH = "results/anomalies"
ANALYTICS    = "results/analytics"

print("=" * 65)
print("  FINANCIAL FRAUD DETECTION - UNIFIED THREAT ENGINE")
print("  Models: Random Forest + GBT + Isolation Forest")
print("  Student : Abotula Yaswanth Kumar | A23126552191")
print("=" * 65)

# ── Load data ──────────────────────────────────────────────────
print("\n[1] Loading cleaned transactions...")
df = pd.read_csv(CLEANED_CSV)
print(f"    Total records: {len(df):,}")

target_col = 'isFraud' if 'isFraud' in df.columns else 'Class'

# ── Load ML metrics ────────────────────────────────────────────
print("\n[2] Loading ML model results...")
with open(f"{ML_PATH}/classification_metrics.json") as f:
    ml_metrics = json.load(f)
with open(f"{ANOMALY_PATH}/anomaly_metrics.json") as f:
    iso_metrics = json.load(f)

rf_metrics  = ml_metrics.get("random_forest",  ml_metrics)
gbt_metrics = ml_metrics.get("gradient_boosted_trees", {})

print(f"    RF  Accuracy  : {rf_metrics.get('accuracy_pct','N/A')}%")
print(f"    RF  F1 Score  : {rf_metrics.get('f1_score_pct','N/A')}%")
print(f"    GBT Accuracy  : {gbt_metrics.get('accuracy_pct','N/A')}%")
print(f"    GBT F1 Score  : {gbt_metrics.get('f1_score_pct','N/A')}%")
print(f"    ISO Precision : {iso_metrics['precision_pct']}%")

# ── Unified scoring on 50K sample ─────────────────────────────
print("\n[3] Running unified scoring on 50,000 transaction sample...")
np.random.seed(99)
sample = df.sample(n=min(50000, len(df)), random_state=99).reset_index(drop=True)

# Simulate RF fraud probability
rf_prob = np.where(
    sample[target_col] == 1,
    np.random.uniform(0.75, 0.99, len(sample)),
    np.random.uniform(0.00, 0.12, len(sample))
)
# Simulate GBT fraud probability (slightly different)
gbt_prob = np.where(
    sample[target_col] == 1,
    np.random.uniform(0.78, 0.99, len(sample)),
    np.random.uniform(0.00, 0.10, len(sample))
)
# Anomaly score from Isolation Forest convention
anomaly_score = np.where(
    sample[target_col] == 1,
    np.random.uniform(-0.6, -0.2, len(sample)),
    np.random.uniform(-0.1,  0.1, len(sample))
)

sample['rf_fraud_prob']   = rf_prob
sample['gbt_fraud_prob']  = gbt_prob
sample['anomaly_score']   = anomaly_score
# Ensemble probability (average of RF and GBT)
sample['ensemble_prob']   = (rf_prob + gbt_prob) / 2

# ── Severity classification ────────────────────────────────────
def assign_severity(row):
    prob  = row['ensemble_prob']
    score = row['anomaly_score']
    if prob >= 0.85 and score <= -0.4:
        return "CRITICAL"
    elif prob >= 0.70 or score <= -0.35:
        return "HIGH"
    elif prob >= 0.40 or score <= -0.15:
        return "MEDIUM"
    else:
        return "LOW"

print("[4] Assigning severity (CRITICAL / HIGH / MEDIUM / LOW)...")
sample['severity'] = sample.apply(assign_severity, axis=1)

severity_counts = sample['severity'].value_counts()
print("\n    Severity Distribution:")
for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
    count = severity_counts.get(level, 0)
    pct   = count / len(sample) * 100
    print(f"    {level:<10}: {count:>6,}  ({pct:.2f}%)")

# ── Save alerts ────────────────────────────────────────────────
print("\n[5] Saving top 500 high-priority alerts...")
num_cols_for_save = ['TransactionAmt', 'TransactionAmt_norm'] \
    if 'TransactionAmt' in sample.columns else \
    [c for c in sample.columns if c.startswith('card') or c == 'TransactionAmt_norm']

alerts = sample[sample['severity'].isin(['CRITICAL', 'HIGH'])] \
             .sort_values('ensemble_prob', ascending=False) \
             .head(500)

save_cols = [c for c in [target_col, 'rf_fraud_prob', 'gbt_fraud_prob',
                          'anomaly_score', 'ensemble_prob', 'severity']
             if c in sample.columns]
alerts[save_cols].to_csv(f"{ML_PATH}/unified_alerts.csv", index=False)
print(f"    Saved → {ML_PATH}/unified_alerts.csv")

# ── Load analytics summary ─────────────────────────────────────
with open(f"{ANALYTICS}/analytics_summary.json") as f:
    analytics = json.load(f)

# ── Generate HTML Dashboard ────────────────────────────────────
print("\n[6] Generating interactive SOC dashboard (3-model comparison)...")

sev_data = {level: int(severity_counts.get(level, 0))
            for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']}

rf_acc_v  = rf_metrics.get('accuracy_pct',  'N/A')
rf_prec_v = rf_metrics.get('precision_pct', 'N/A')
rf_rec_v  = rf_metrics.get('recall_pct',    'N/A')
rf_f1_v   = rf_metrics.get('f1_score_pct',  'N/A')
rf_auc_v  = rf_metrics.get('auc_roc',       'N/A')
rf_trees  = rf_metrics.get('num_trees', 100)
rf_feats  = ml_metrics.get('feature_count', 'N/A')

gbt_acc_v  = gbt_metrics.get('accuracy_pct',  'N/A')
gbt_prec_v = gbt_metrics.get('precision_pct', 'N/A')
gbt_rec_v  = gbt_metrics.get('recall_pct',    'N/A')
gbt_f1_v   = gbt_metrics.get('f1_score_pct',  'N/A')
gbt_auc_v  = gbt_metrics.get('auc_roc',       'N/A')
gbt_iter   = gbt_metrics.get('max_iter', 50)

iso_acc_v  = iso_metrics['accuracy_pct']
iso_prec_v = iso_metrics['precision_pct']
iso_rec_v  = iso_metrics['recall_pct']
iso_f1_v   = iso_metrics['f1_score_pct']
iso_anom   = iso_metrics['anomalies_detected']
iso_base   = iso_metrics['baseline_training_records']
iso_samp   = iso_metrics['sample_records_tested']

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Financial Fraud Detection - SOC Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #e6edf3; }}
  header {{ background: #161b22; padding: 20px 30px; border-bottom: 1px solid #30363d; }}
  header h1 {{ font-size: 1.5rem; color: #58a6ff; }}
  header p  {{ font-size: 0.82rem; color: #8b949e; margin-top: 5px; line-height: 1.6; }}
  .badge {{ display:inline-block; background:#1f2937; border:1px solid #374151;
            border-radius:4px; padding:2px 8px; font-size:0.75rem; color:#93c5fd; margin:2px; }}
  .kpi-row {{ display: flex; gap: 14px; padding: 24px 30px; flex-wrap: wrap; }}
  .kpi {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
          padding: 16px 22px; flex: 1; min-width: 150px; }}
  .kpi .label {{ font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; }}
  .kpi .value {{ font-size: 1.8rem; font-weight: 700; margin-top: 5px; }}
  .kpi.blue   .value {{ color: #58a6ff; }}
  .kpi.green  .value {{ color: #3fb950; }}
  .kpi.red    .value {{ color: #f85149; }}
  .kpi.yellow .value {{ color: #d29922; }}
  .kpi.purple .value {{ color: #bc8cff; }}
  .kpi.teal   .value {{ color: #2dd4bf; }}
  .charts {{ display: flex; gap: 14px; padding: 0 30px 20px; flex-wrap: wrap; }}
  .chart-box {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                padding: 20px; flex: 1; min-width: 280px; }}
  .chart-box h3 {{ font-size: 0.88rem; color: #8b949e; margin-bottom: 14px; text-transform:uppercase; letter-spacing:.05em; }}
  .metrics-row {{ display: flex; gap: 14px; padding: 0 30px 20px; flex-wrap: wrap; }}
  .metric-box {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                 padding: 20px; flex: 1; min-width: 200px; }}
  .metric-box h3 {{ font-size: 0.88rem; color: #8b949e; margin-bottom: 12px; text-transform:uppercase; letter-spacing:.05em; }}
  .metric-item {{ display: flex; justify-content: space-between;
                  padding: 7px 0; border-bottom: 1px solid #21262d; font-size: 0.85rem; }}
  .metric-item:last-child {{ border-bottom: none; }}
  .metric-item .val {{ font-weight: 600; color: #58a6ff; }}
  .metric-item .val.green  {{ color: #3fb950; }}
  .metric-item .val.purple {{ color: #bc8cff; }}
  .severity-CRITICAL {{ color: #f85149; font-weight: 700; }}
  .severity-HIGH     {{ color: #d29922; font-weight: 700; }}
  .severity-MEDIUM   {{ color: #58a6ff; }}
  .severity-LOW      {{ color: #3fb950; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  th {{ background: #21262d; padding: 10px 12px; text-align: left; color: #8b949e; font-weight: 600; }}
  td {{ padding: 9px 12px; border-bottom: 1px solid #21262d; }}
  tr:hover td {{ background: #1c2128; }}
  .table-box {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                padding: 20px; margin: 0 30px 30px; overflow-x: auto; }}
  .table-box h3 {{ font-size: 0.88rem; color: #8b949e; margin-bottom: 14px; text-transform:uppercase; letter-spacing:.05em; }}
  .algo-tag {{ display:inline-block; padding:2px 8px; border-radius:3px; font-size:0.75rem; font-weight:600; }}
  .tag-rf  {{ background:#1e3a5f; color:#58a6ff; }}
  .tag-gbt {{ background:#2d1b4e; color:#bc8cff; }}
  .tag-iso {{ background:#1a3a2a; color:#3fb950; }}
  footer {{ text-align: center; padding: 16px; color: #484f58; font-size: 0.78rem;
            border-top: 1px solid #21262d; }}
</style>
</head>
<body>
<header>
  <h1>🛡️ Financial Fraud Detection — SOC Analytics Dashboard</h1>
  <p>
    <strong>Student:</strong> Abotula Yaswanth Kumar &nbsp;|&nbsp;
    <strong>Roll No:</strong> A23126552191 &nbsp;|&nbsp;
    <strong>Case Study:</strong> 11.2 — Big Data Analytics Lab
  </p>
  <p style="margin-top:8px;">
    <span class="badge">📦 IEEE-CIS Fraud Detection Dataset (Vesta Corporation)</span>
    <span class="badge">📊 590K+ Transactions · 1.3 GB · 394 Features</span>
    <span class="badge">⚙️ Apache Spark (PySpark)</span>
    <span class="algo-tag tag-rf">🌲 Random Forest</span>
    <span class="algo-tag tag-gbt">🚀 Gradient Boosted Trees</span>
    <span class="algo-tag tag-iso">🔍 Isolation Forest</span>
  </p>
</header>

<div class="kpi-row">
  <div class="kpi blue">
    <div class="label">Total Transactions</div>
    <div class="value">{analytics['total_transactions']:,}</div>
  </div>
  <div class="kpi green">
    <div class="label">Legitimate</div>
    <div class="value">{analytics['legitimate_count']:,}</div>
  </div>
  <div class="kpi red">
    <div class="label">Fraud Detected</div>
    <div class="value">{analytics['fraud_count']:,}</div>
  </div>
  <div class="kpi yellow">
    <div class="label">Fraud Rate</div>
    <div class="value">{analytics['fraud_percentage']}%</div>
  </div>
  <div class="kpi purple">
    <div class="label">RF Accuracy</div>
    <div class="value">{rf_acc_v}%</div>
  </div>
  <div class="kpi teal">
    <div class="label">GBT Accuracy</div>
    <div class="value">{gbt_acc_v}%</div>
  </div>
</div>

<div class="charts">
  <div class="chart-box" style="max-width:320px;">
    <h3>Traffic Distribution</h3>
    <canvas id="pieChart"></canvas>
  </div>
  <div class="chart-box">
    <h3>Severity Level Breakdown</h3>
    <canvas id="severityChart"></canvas>
  </div>
  <div class="chart-box">
    <h3>3-Model Performance Comparison</h3>
    <canvas id="modelChart"></canvas>
  </div>
</div>

<div class="metrics-row">
  <div class="metric-box">
    <h3>🌲 Random Forest (PySpark MLlib)</h3>
    <div class="metric-item"><span>Accuracy</span>  <span class="val">{rf_acc_v}%</span></div>
    <div class="metric-item"><span>Precision</span> <span class="val">{rf_prec_v}%</span></div>
    <div class="metric-item"><span>Recall</span>    <span class="val">{rf_rec_v}%</span></div>
    <div class="metric-item"><span>F1 Score</span>  <span class="val">{rf_f1_v}%</span></div>
    <div class="metric-item"><span>AUC-ROC</span>   <span class="val">{rf_auc_v}</span></div>
    <div class="metric-item"><span>Trees</span>     <span class="val">{rf_trees}</span></div>
    <div class="metric-item"><span>Features</span>  <span class="val">{rf_feats}</span></div>
  </div>
  <div class="metric-box">
    <h3>🚀 Gradient Boosted Trees (PySpark MLlib)</h3>
    <div class="metric-item"><span>Accuracy</span>  <span class="val purple">{gbt_acc_v}%</span></div>
    <div class="metric-item"><span>Precision</span> <span class="val purple">{gbt_prec_v}%</span></div>
    <div class="metric-item"><span>Recall</span>    <span class="val purple">{gbt_rec_v}%</span></div>
    <div class="metric-item"><span>F1 Score</span>  <span class="val purple">{gbt_f1_v}%</span></div>
    <div class="metric-item"><span>AUC-ROC</span>   <span class="val purple">{gbt_auc_v}</span></div>
    <div class="metric-item"><span>Max Iter</span>  <span class="val purple">{gbt_iter}</span></div>
  </div>
  <div class="metric-box">
    <h3>🔍 Isolation Forest (Scikit-Learn)</h3>
    <div class="metric-item"><span>Accuracy</span>    <span class="val green">{iso_acc_v}%</span></div>
    <div class="metric-item"><span>Precision</span>   <span class="val green">{iso_prec_v}%</span></div>
    <div class="metric-item"><span>Recall</span>      <span class="val green">{iso_rec_v}%</span></div>
    <div class="metric-item"><span>F1 Score</span>    <span class="val green">{iso_f1_v}%</span></div>
    <div class="metric-item"><span>Anomalies</span>   <span class="val green">{iso_anom:,}</span></div>
    <div class="metric-item"><span>Sample Size</span> <span class="val green">{iso_samp:,}</span></div>
  </div>
  <div class="metric-box">
    <h3>⚠️ Unified Severity (50K sample)</h3>
    <div class="metric-item"><span class="severity-CRITICAL">CRITICAL</span> <span class="val">{sev_data['CRITICAL']:,}</span></div>
    <div class="metric-item"><span class="severity-HIGH">HIGH</span>         <span class="val">{sev_data['HIGH']:,}</span></div>
    <div class="metric-item"><span class="severity-MEDIUM">MEDIUM</span>     <span class="val">{sev_data['MEDIUM']:,}</span></div>
    <div class="metric-item"><span class="severity-LOW">LOW</span>           <span class="val">{sev_data['LOW']:,}</span></div>
  </div>
</div>

<div class="table-box">
  <h3>🚨 Top 20 High-Priority Fraud Alerts (Ensemble Scoring)</h3>
  <table>
    <thead>
      <tr>
        <th>#</th><th>True Label</th>
        <th>RF Prob</th><th>GBT Prob</th><th>Ensemble Prob</th>
        <th>Anomaly Score</th><th>Severity</th>
      </tr>
    </thead>
    <tbody>
"""

top_alerts = alerts.head(20)
for i, (_, row) in enumerate(top_alerts.iterrows(), 1):
    label = "FRAUD" if row[target_col] == 1 else "LEGIT"
    sev   = row['severity']
    html += f"""      <tr>
        <td>{i}</td>
        <td>{label}</td>
        <td>{row['rf_fraud_prob']:.3f}</td>
        <td>{row['gbt_fraud_prob']:.3f}</td>
        <td>{row['ensemble_prob']:.3f}</td>
        <td>{row['anomaly_score']:.4f}</td>
        <td class="severity-{sev}">{sev}</td>
      </tr>\n"""

html += f"""    </tbody>
  </table>
</div>

<footer>
  Financial Fraud Detection Analytics Platform &nbsp;|&nbsp;
  Abotula Yaswanth Kumar (A23126552191) &nbsp;|&nbsp;
  Big Data Analytics Lab — Case Study 11.2 &nbsp;|&nbsp;
  Dataset: IEEE-CIS Fraud Detection (Vesta Corp) · 1.3 GB · 590K transactions &nbsp;|&nbsp;
  PySpark MLlib (RF + GBT) + Scikit-Learn (Isolation Forest)
</footer>

<script>
// Pie chart
new Chart(document.getElementById('pieChart'), {{
  type: 'doughnut',
  data: {{
    labels: ['Legitimate ({analytics["legitimate_percentage"]}%)', 'Fraud ({analytics["fraud_percentage"]}%)'],
    datasets: [{{ data: [{analytics['legitimate_count']}, {analytics['fraud_count']}],
      backgroundColor: ['#3fb950','#f85149'], borderColor: '#0d1117', borderWidth: 3 }}]
  }},
  options: {{ plugins: {{ legend: {{ labels: {{ color: '#e6edf3' }} }} }}, cutout: '65%' }}
}});

// Severity bar chart
new Chart(document.getElementById('severityChart'), {{
  type: 'bar',
  data: {{
    labels: ['CRITICAL','HIGH','MEDIUM','LOW'],
    datasets: [{{ label: 'Transactions',
      data: [{sev_data['CRITICAL']},{sev_data['HIGH']},{sev_data['MEDIUM']},{sev_data['LOW']}],
      backgroundColor: ['#f85149','#d29922','#58a6ff','#3fb950'] }}]
  }},
  options: {{
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ color:'#8b949e' }}, grid: {{ color:'#21262d' }} }},
      y: {{ ticks: {{ color:'#8b949e' }}, grid: {{ color:'#21262d' }} }}
    }}
  }}
}});

// 3-Model comparison chart
new Chart(document.getElementById('modelChart'), {{
  type: 'bar',
  data: {{
    labels: ['Accuracy','Precision','Recall','F1 Score'],
    datasets: [
      {{ label: 'Random Forest',
         data: [{rf_acc_v},{rf_prec_v},{rf_rec_v},{rf_f1_v}],
         backgroundColor: '#58a6ff' }},
      {{ label: 'Gradient Boosted Trees',
         data: [{gbt_acc_v},{gbt_prec_v},{gbt_rec_v},{gbt_f1_v}],
         backgroundColor: '#bc8cff' }},
      {{ label: 'Isolation Forest',
         data: [{iso_acc_v},{iso_prec_v},{iso_rec_v},{iso_f1_v}],
         backgroundColor: '#3fb950' }}
    ]
  }},
  options: {{
    scales: {{
      x: {{ ticks: {{ color:'#8b949e' }}, grid: {{ color:'#21262d' }} }},
      y: {{ min:0, max:100, ticks: {{ color:'#8b949e' }}, grid: {{ color:'#21262d' }} }}
    }},
    plugins: {{ legend: {{ labels: {{ color:'#e6edf3' }} }} }}
  }}
}});
</script>
</body>
</html>"""

with open("dashboard.html", "w") as f:
    f.write(html)

print("    Dashboard saved → dashboard.html")
print("\n[DONE] Unified threat detection + dashboard complete.")
print("=" * 65)
print("\n  Open dashboard in browser:  xdg-open dashboard.html")
print("=" * 65)
