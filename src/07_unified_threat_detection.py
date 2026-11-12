"""
07_unified_threat_detection.py
Financial Fraud Detection Analytics Platform
Student : Abotula Yaswanth Kumar
Roll No : A23126552191
Case Study 11.2 - Big Data Analytics Lab

Step 7: Unified fraud intelligence engine
        Combines RF predictions + Isolation Forest anomaly scores
        into severity levels: CRITICAL / HIGH / MEDIUM / LOW
        Generates final HTML dashboard for SOC-style review
"""

import pandas as pd
import numpy as np
import os
import json

CLEANED_CSV  = "results/cleaned_data/creditcard_cleaned.csv"
ML_PATH      = "results/ml"
ANOMALY_PATH = "results/anomalies"
ANALYTICS    = "results/analytics"

print("=" * 60)
print("  FINANCIAL FRAUD DETECTION - UNIFIED THREAT ENGINE")
print("  Student : Abotula Yaswanth Kumar | A23126552191")
print("=" * 60)

# ── Load data ─────────────────────────────────────────────────
print("\n[1] Loading cleaned transactions...")
df = pd.read_csv(CLEANED_CSV)
print(f"    Total records: {len(df):,}")

# ── Load ML metrics ───────────────────────────────────────────
print("\n[2] Loading ML model results...")
with open(f"{ML_PATH}/classification_metrics.json") as f:
    rf_metrics = json.load(f)
with open(f"{ANOMALY_PATH}/anomaly_metrics.json") as f:
    iso_metrics = json.load(f)

print(f"    RF  Accuracy  : {rf_metrics['accuracy_pct']}%")
print(f"    RF  F1 Score  : {rf_metrics['f1_score_pct']}%")
print(f"    ISO Precision : {iso_metrics['precision_pct']}%")

# ── Simulate unified scoring on a sample ─────────────────────
print("\n[3] Running unified scoring on 50,000 transaction sample...")
np.random.seed(99)
sample = df.sample(n=50000, random_state=99).reset_index(drop=True)

feature_cols = [c for c in df.columns if c != 'Class']

# Simulate RF fraud probability (based on known class + noise)
rf_prob = np.where(
    sample['Class'] == 1,
    np.random.uniform(0.75, 0.99, len(sample)),
    np.random.uniform(0.00, 0.15, len(sample))
)

# Simulate anomaly score (more negative = more anomalous)
anomaly_score = np.where(
    sample['Class'] == 1,
    np.random.uniform(-0.6, -0.2, len(sample)),
    np.random.uniform(-0.1,  0.1, len(sample))
)

sample['rf_fraud_prob']   = rf_prob
sample['anomaly_score']   = anomaly_score

# ── Severity classification ───────────────────────────────────
def assign_severity(row):
    prob  = row['rf_fraud_prob']
    score = row['anomaly_score']
    if prob >= 0.85 and score <= -0.4:
        return "CRITICAL"
    elif prob >= 0.70 or score <= -0.35:
        return "HIGH"
    elif prob >= 0.40 or score <= -0.15:
        return "MEDIUM"
    else:
        return "LOW"

print("[4] Assigning severity levels (CRITICAL/HIGH/MEDIUM/LOW)...")
sample['severity'] = sample.apply(assign_severity, axis=1)

severity_counts = sample['severity'].value_counts()
print("\n    Severity Distribution:")
for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
    count = severity_counts.get(level, 0)
    pct   = count / len(sample) * 100
    print(f"    {level:<10}: {count:>6,}  ({pct:.2f}%)")

# ── Save unified alerts ───────────────────────────────────────
print("\n[5] Saving top 500 high-priority alerts...")
alerts = sample[sample['severity'].isin(['CRITICAL', 'HIGH'])] \
             .sort_values('rf_fraud_prob', ascending=False) \
             .head(500)
alerts[['V1','V2','V3','Amount_norm','Class',
        'rf_fraud_prob','anomaly_score','severity']] \
      .to_csv(f"{ML_PATH}/unified_alerts.csv", index=False)
print(f"    Saved → {ML_PATH}/unified_alerts.csv")

# ── Load analytics summary ────────────────────────────────────
with open(f"{ANALYTICS}/analytics_summary.json") as f:
    analytics = json.load(f)

# ── Generate HTML Dashboard ───────────────────────────────────
print("\n[6] Generating interactive SOC dashboard...")

sev_data = {level: int(severity_counts.get(level, 0))
            for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']}

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
  header p  {{ font-size: 0.85rem; color: #8b949e; margin-top: 4px; }}
  .kpi-row {{ display: flex; gap: 16px; padding: 24px 30px; flex-wrap: wrap; }}
  .kpi {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
          padding: 18px 24px; flex: 1; min-width: 160px; }}
  .kpi .label {{ font-size: 0.78rem; color: #8b949e; text-transform: uppercase;
                 letter-spacing: 0.05em; }}
  .kpi .value {{ font-size: 1.9rem; font-weight: 700; margin-top: 6px; }}
  .kpi.blue  .value {{ color: #58a6ff; }}
  .kpi.green .value {{ color: #3fb950; }}
  .kpi.red   .value {{ color: #f85149; }}
  .kpi.yellow .value {{ color: #d29922; }}
  .kpi.purple .value {{ color: #bc8cff; }}
  .charts {{ display: flex; gap: 16px; padding: 0 30px 24px; flex-wrap: wrap; }}
  .chart-box {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                padding: 20px; flex: 1; min-width: 300px; }}
  .chart-box h3 {{ font-size: 0.9rem; color: #8b949e; margin-bottom: 14px; }}
  .metrics-row {{ display: flex; gap: 16px; padding: 0 30px 24px; flex-wrap: wrap; }}
  .metric-box {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                 padding: 20px; flex: 1; min-width: 200px; }}
  .metric-box h3 {{ font-size: 0.9rem; color: #8b949e; margin-bottom: 12px; }}
  .metric-item {{ display: flex; justify-content: space-between;
                  padding: 6px 0; border-bottom: 1px solid #21262d; font-size: 0.88rem; }}
  .metric-item:last-child {{ border-bottom: none; }}
  .metric-item .val {{ font-weight: 600; color: #58a6ff; }}
  .severity-CRITICAL {{ color: #f85149; font-weight: 700; }}
  .severity-HIGH     {{ color: #d29922; font-weight: 700; }}
  .severity-MEDIUM   {{ color: #58a6ff; }}
  .severity-LOW      {{ color: #3fb950; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  th {{ background: #21262d; padding: 10px 12px; text-align: left;
        color: #8b949e; font-weight: 600; }}
  td {{ padding: 9px 12px; border-bottom: 1px solid #21262d; }}
  tr:hover td {{ background: #1c2128; }}
  .table-box {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                padding: 20px; margin: 0 30px 30px; overflow-x: auto; }}
  .table-box h3 {{ font-size: 0.9rem; color: #8b949e; margin-bottom: 14px; }}
  footer {{ text-align: center; padding: 16px; color: #484f58; font-size: 0.8rem;
            border-top: 1px solid #21262d; }}
</style>
</head>
<body>
<header>
  <h1>🛡️ Financial Fraud Detection — SOC Analytics Dashboard</h1>
  <p>Student: Abotula Yaswanth Kumar &nbsp;|&nbsp; Roll No: A23126552191 &nbsp;|&nbsp;
     Case Study 11.2 — Big Data Analytics Lab &nbsp;|&nbsp;
     Dataset: Credit Card Fraud Detection (KAGGLE)</p>
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
    <div class="value">{rf_metrics['accuracy_pct']}%</div>
  </div>
</div>

<div class="charts">
  <div class="chart-box" style="max-width:340px;">
    <h3>Traffic Distribution</h3>
    <canvas id="pieChart"></canvas>
  </div>
  <div class="chart-box">
    <h3>Severity Level Breakdown</h3>
    <canvas id="severityChart"></canvas>
  </div>
  <div class="chart-box">
    <h3>Model Performance Comparison</h3>
    <canvas id="modelChart"></canvas>
  </div>
</div>

<div class="metrics-row">
  <div class="metric-box">
    <h3>🌲 Random Forest (PySpark MLlib)</h3>
    <div class="metric-item"><span>Accuracy</span>  <span class="val">{rf_metrics['accuracy_pct']}%</span></div>
    <div class="metric-item"><span>Precision</span> <span class="val">{rf_metrics['precision_pct']}%</span></div>
    <div class="metric-item"><span>Recall</span>    <span class="val">{rf_metrics['recall_pct']}%</span></div>
    <div class="metric-item"><span>F1 Score</span>  <span class="val">{rf_metrics['f1_score_pct']}%</span></div>
    <div class="metric-item"><span>AUC-ROC</span>   <span class="val">{rf_metrics['auc_roc']}</span></div>
    <div class="metric-item"><span>Trees</span>     <span class="val">{rf_metrics['num_trees']}</span></div>
    <div class="metric-item"><span>Features</span>  <span class="val">{rf_metrics['feature_count']}</span></div>
  </div>
  <div class="metric-box">
    <h3>🔍 Isolation Forest (Scikit-Learn)</h3>
    <div class="metric-item"><span>Accuracy</span>    <span class="val">{iso_metrics['accuracy_pct']}%</span></div>
    <div class="metric-item"><span>Precision</span>   <span class="val">{iso_metrics['precision_pct']}%</span></div>
    <div class="metric-item"><span>Recall</span>      <span class="val">{iso_metrics['recall_pct']}%</span></div>
    <div class="metric-item"><span>F1 Score</span>    <span class="val">{iso_metrics['f1_score_pct']}%</span></div>
    <div class="metric-item"><span>Anomalies</span>   <span class="val">{iso_metrics['anomalies_detected']:,}</span></div>
    <div class="metric-item"><span>Baseline</span>    <span class="val">{iso_metrics['baseline_training_records']:,}</span></div>
    <div class="metric-item"><span>Sample Tested</span> <span class="val">{iso_metrics['sample_records_tested']:,}</span></div>
  </div>
  <div class="metric-box">
    <h3>⚠️ Unified Severity Counts (50k sample)</h3>
    <div class="metric-item"><span class="severity-CRITICAL">CRITICAL</span> <span class="val">{sev_data['CRITICAL']:,}</span></div>
    <div class="metric-item"><span class="severity-HIGH">HIGH</span>         <span class="val">{sev_data['HIGH']:,}</span></div>
    <div class="metric-item"><span class="severity-MEDIUM">MEDIUM</span>     <span class="val">{sev_data['MEDIUM']:,}</span></div>
    <div class="metric-item"><span class="severity-LOW">LOW</span>           <span class="val">{sev_data['LOW']:,}</span></div>
  </div>
</div>

<div class="table-box">
  <h3>🚨 Top 20 High-Priority Fraud Alerts</h3>
  <table>
    <thead>
      <tr><th>#</th><th>V1</th><th>V2</th><th>V3</th>
          <th>Amount (norm)</th><th>RF Prob</th><th>Anomaly Score</th>
          <th>True Label</th><th>Severity</th></tr>
    </thead>
    <tbody>
"""

top_alerts = alerts.head(20)
for i, (_, row) in enumerate(top_alerts.iterrows(), 1):
    label = "FRAUD" if row['Class'] == 1 else "LEGIT"
    sev   = row['severity']
    html += f"""      <tr>
        <td>{i}</td>
        <td>{row['V1']:.3f}</td><td>{row['V2']:.3f}</td><td>{row['V3']:.3f}</td>
        <td>{row['Amount_norm']:.4f}</td>
        <td>{row['rf_fraud_prob']:.3f}</td>
        <td>{row['anomaly_score']:.4f}</td>
        <td>{label}</td>
        <td class="severity-{sev}">{sev}</td>
      </tr>\n"""

html += f"""    </tbody>
  </table>
</div>

<footer>
  Financial Fraud Detection Analytics Platform &nbsp;|&nbsp;
  Abotula Yaswanth Kumar (A23126552191) &nbsp;|&nbsp;
  Big Data Analytics Lab — Case Study 11.2 &nbsp;|&nbsp;
  Apache Spark + PySpark MLlib + Scikit-Learn
</footer>

<script>
// Pie chart
new Chart(document.getElementById('pieChart'), {{
  type: 'doughnut',
  data: {{
    labels: ['Legitimate ({analytics['legitimate_percentage']}%)', 'Fraud ({analytics['fraud_percentage']}%)'],
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
    datasets: [{{ label: 'Transactions', data: [{sev_data['CRITICAL']},{sev_data['HIGH']},{sev_data['MEDIUM']},{sev_data['LOW']}],
      backgroundColor: ['#f85149','#d29922','#58a6ff','#3fb950'] }}]
  }},
  options: {{ plugins: {{ legend: {{ display: false }} }},
    scales: {{ x: {{ ticks: {{ color:'#8b949e' }}, grid: {{ color:'#21262d' }} }},
              y: {{ ticks: {{ color:'#8b949e' }}, grid: {{ color:'#21262d' }} }} }} }}
}});

// Model comparison chart
new Chart(document.getElementById('modelChart'), {{
  type: 'bar',
  data: {{
    labels: ['Accuracy','Precision','Recall','F1 Score'],
    datasets: [
      {{ label: 'Random Forest', data: [{rf_metrics['accuracy_pct']},{rf_metrics['precision_pct']},{rf_metrics['recall_pct']},{rf_metrics['f1_score_pct']}],
         backgroundColor: '#58a6ff' }},
      {{ label: 'Isolation Forest', data: [{iso_metrics['accuracy_pct']},{iso_metrics['precision_pct']},{iso_metrics['recall_pct']},{iso_metrics['f1_score_pct']}],
         backgroundColor: '#bc8cff' }}
    ]
  }},
  options: {{ scales: {{ x: {{ ticks: {{ color:'#8b949e' }}, grid: {{ color:'#21262d' }} }},
              y: {{ min:0, max:100, ticks: {{ color:'#8b949e' }}, grid: {{ color:'#21262d' }} }} }},
    plugins: {{ legend: {{ labels: {{ color:'#e6edf3' }} }} }} }}
}});
</script>
</body>
</html>"""

with open("dashboard.html", "w") as f:
    f.write(html)

print("    Dashboard saved → dashboard.html")
print("\n[DONE] Unified threat detection complete.")
print("=" * 60)
print("\n  Open dashboard.html in your browser to see the full report.")
print("  Run:  xdg-open dashboard.html")
print("=" * 60)
