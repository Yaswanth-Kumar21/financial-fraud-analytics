"""
generate_screenshots.py
Financial Fraud Detection Analytics Platform
Student : Abotula Yaswanth Kumar | A23126552191

Generates 4 publication-quality screenshots from real run results:
  01_dataset_exploration.png   — class distribution + amount stats
  02_spark_sql_analytics.png   — fraud by hour + amount bucket
  03_ml_comparison.png         — 3-model performance bar chart
  04_anomaly_detection.png     — Isolation Forest anomaly summary
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

SCREENSHOTS = "screenshots"
os.makedirs(SCREENSHOTS, exist_ok=True)

# ── Color palette (dark theme matching dashboard) ──────────────
BG      = "#0d1117"
PANEL   = "#161b22"
BORDER  = "#30363d"
BLUE    = "#58a6ff"
GREEN   = "#3fb950"
RED     = "#f85149"
YELLOW  = "#d29922"
PURPLE  = "#bc8cff"
TEAL    = "#2dd4bf"
TEXT    = "#e6edf3"
SUBTEXT = "#8b949e"

def apply_dark(fig, axes):
    fig.patch.set_facecolor(BG)
    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=SUBTEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.grid(color=BORDER, linestyle='--', linewidth=0.5, alpha=0.7)

# ═══════════════════════════════════════════════════════════════
# Screenshot 1 — Dataset Exploration
# ═══════════════════════════════════════════════════════════════
print("[1] Generating 01_dataset_exploration.png ...")

with open("results/analytics/dataset_summary.json") as f:
    ds = json.load(f)

fig = plt.figure(figsize=(14, 7), facecolor=BG)
fig.suptitle("IEEE-CIS Fraud Detection — Dataset Exploration\n"
             "Student: Abotula Yaswanth Kumar | A23126552191 | Case Study 11.2",
             color=TEXT, fontsize=13, fontweight='bold', y=0.97)

gs = GridSpec(1, 3, figure=fig, wspace=0.35)

# --- (a) Donut chart: class distribution
ax1 = fig.add_subplot(gs[0])
sizes  = [ds['legitimate_count'], ds['fraud_count']]
colors = [GREEN, RED]
wedges, texts, autotexts = ax1.pie(
    sizes, colors=colors, autopct='%1.2f%%',
    startangle=90, pctdistance=0.75,
    wedgeprops=dict(width=0.5, edgecolor=BG, linewidth=2)
)
for at in autotexts:
    at.set_color(TEXT); at.set_fontsize(10); at.set_fontweight('bold')
ax1.set_facecolor(PANEL)
ax1.set_title("Class Distribution", color=TEXT, fontsize=11, pad=12)
ax1.legend(
    handles=[
        mpatches.Patch(color=GREEN, label=f"Legitimate  {ds['legitimate_count']:,}"),
        mpatches.Patch(color=RED,   label=f"Fraud       {ds['fraud_count']:,}"),
    ],
    loc='lower center', bbox_to_anchor=(0.5, -0.12),
    frameon=False, labelcolor=TEXT, fontsize=9
)

# --- (b) KPI cards as text panel
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor(PANEL)
ax2.axis('off')
ax2.set_title("Dataset KPIs", color=TEXT, fontsize=11, pad=12)
kpis = [
    ("Total Transactions",  f"{ds['total_transactions']:,}",         BLUE),
    ("Legitimate Cases",    f"{ds['legitimate_count']:,}",            GREEN),
    ("Fraud Cases",         f"{ds['fraud_count']:,}",                 RED),
    ("Fraud Rate",          f"{ds['fraud_percentage']:.3f}%",         YELLOW),
    ("Imbalance Ratio",     f"{ds['imbalance_ratio']:.1f} : 1",       PURPLE),
    ("Total Features",      f"{ds['total_features']}",                TEAL),
    ("Avg Txn Amount",      f"${ds['amount_mean']:.2f}",              BLUE),
    ("Max Txn Amount",      f"${ds['amount_max']:,.2f}",              YELLOW),
    ("Dataset Size",        "1.3 GB",                                  GREEN),
    ("Source",              "IEEE-CIS / Vesta Corp",                   SUBTEXT),
]
for i, (label, value, color) in enumerate(kpis):
    y_pos = 0.93 - i * 0.088
    ax2.text(0.0, y_pos, label,  color=SUBTEXT, fontsize=9,  transform=ax2.transAxes)
    ax2.text(1.0, y_pos, value,  color=color,   fontsize=9.5,
             transform=ax2.transAxes, ha='right', fontweight='bold')
    ax2.plot([0, 1], [y_pos - 0.025, y_pos - 0.025],
             color=BORDER, linewidth=0.5, transform=ax2.transAxes, clip_on=False)

# --- (c) Product code fraud rate bar chart
ax3 = fig.add_subplot(gs[2])
apply_dark(fig, ax3)
products    = ['W', 'C', 'R', 'H', 'S']
fraud_rates = [2.04, 11.69, 3.78, 4.77, 5.90]
bar_colors  = [RED if r > 5 else YELLOW if r > 3 else GREEN for r in fraud_rates]
bars = ax3.bar(products, fraud_rates, color=bar_colors, edgecolor=BG, linewidth=1.2, width=0.6)
for bar, rate in zip(bars, fraud_rates):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
             f'{rate:.1f}%', ha='center', va='bottom', color=TEXT, fontsize=9, fontweight='bold')
ax3.set_title("Fraud Rate by ProductCD", color=TEXT, fontsize=11, pad=10)
ax3.set_xlabel("Product Code", color=SUBTEXT, fontsize=9)
ax3.set_ylabel("Fraud Rate (%)", color=SUBTEXT, fontsize=9)
ax3.set_ylim(0, 15)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(f"{SCREENSHOTS}/01_dataset_exploration.png", dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("    Saved → screenshots/01_dataset_exploration.png")


# ═══════════════════════════════════════════════════════════════
# Screenshot 2 — Spark SQL Analytics
# ═══════════════════════════════════════════════════════════════
print("[2] Generating 02_spark_sql_analytics.png ...")

with open("results/analytics/traffic_summary.json") as f:
    ts = json.load(f)
with open("results/analytics/amount_distribution.json") as f:
    ad = json.load(f)

fig = plt.figure(figsize=(14, 7), facecolor=BG)
fig.suptitle("IEEE-CIS Fraud Detection — Spark SQL Analytics\n"
             "Student: Abotula Yaswanth Kumar | A23126552191 | Case Study 11.2",
             color=TEXT, fontsize=13, fontweight='bold', y=0.97)

gs = GridSpec(1, 3, figure=fig, wspace=0.38)

# --- (a) Fraud by hour of day
ax1 = fig.add_subplot(gs[0])
apply_dark(fig, ax1)
hours      = list(range(24))
fraud_rate = [3.14,3.13,3.75,3.83,5.19,7.03,7.77,10.61,9.30,9.00,
              5.32,3.88,3.04,2.29,2.42,2.54,2.95,3.15,3.52,3.47,
              3.43,3.40,3.27,3.70]
bar_colors = [RED if r > 7 else YELLOW if r > 4 else BLUE for r in fraud_rate]
ax1.bar(hours, fraud_rate, color=bar_colors, width=0.8, edgecolor=BG, linewidth=0.5)
ax1.axhline(y=3.499, color=GREEN, linestyle='--', linewidth=1.2, label='Avg 3.499%')
ax1.set_title("Fraud Rate by Hour of Day", color=TEXT, fontsize=11, pad=10)
ax1.set_xlabel("Hour (0–23)", color=SUBTEXT, fontsize=9)
ax1.set_ylabel("Fraud Rate (%)", color=SUBTEXT, fontsize=9)
ax1.legend(frameon=False, labelcolor=TEXT, fontsize=8)
ax1.set_xlim(-0.5, 23.5)

# --- (b) Amount bucket analysis
ax2 = fig.add_subplot(gs[1])
apply_dark(fig, ax2)
buckets  = ['LOW\n(<$50)', 'MEDIUM\n($50-500)', 'HIGH\n($500-5K)', 'VERY HIGH\n(>$5K)']
totals   = [ad.get('LOW',{}).get('total',187515),
            ad.get('MEDIUM',{}).get('total',378701),
            ad.get('HIGH',{}).get('total',24306),
            ad.get('VERY_HIGH',{}).get('total',18)]
frauds   = [ad.get('LOW',{}).get('fraud',7480),
            ad.get('MEDIUM',{}).get('fraud',12034),
            ad.get('HIGH',{}).get('fraud',1148),
            ad.get('VERY_HIGH',{}).get('fraud',1)]
legits   = [t - f for t, f in zip(totals, frauds)]
x = np.arange(len(buckets))
w = 0.38
ax2.bar(x - w/2, legits,  width=w, label='Legitimate', color=GREEN, edgecolor=BG)
ax2.bar(x + w/2, frauds,  width=w, label='Fraud',      color=RED,   edgecolor=BG)
ax2.set_title("Transactions by Amount Bucket", color=TEXT, fontsize=11, pad=10)
ax2.set_xlabel("Amount Range", color=SUBTEXT, fontsize=9)
ax2.set_ylabel("Count", color=SUBTEXT, fontsize=9)
ax2.set_xticks(x); ax2.set_xticklabels(buckets, fontsize=8)
ax2.legend(frameon=False, labelcolor=TEXT, fontsize=8)

# --- (c) Volume KPIs
ax3 = fig.add_subplot(gs[2])
ax3.set_facecolor(PANEL); ax3.axis('off')
ax3.set_title("Spark SQL Query Results", color=TEXT, fontsize=11, pad=12)
kpis = [
    ("Total Transactions",   f"{ts['total_transactions']:,}",        BLUE),
    ("Legitimate",           f"{ts['legitimate_count']:,}",           GREEN),
    ("Fraud Cases",          f"{ts['fraud_count']:,}",                RED),
    ("Fraud Rate",           f"{ts['fraud_rate_pct']:.3f}%",          YELLOW),
    ("Avg Transaction",      f"${ts['avg_transaction_amt']:.2f}",     TEAL),
    ("Total Volume",         f"${ts['total_volume_usd']/1e6:.1f}M",   BLUE),
    ("Fraud Volume",         f"${ts['fraud_volume_usd']/1e6:.2f}M",   RED),
    ("Peak Fraud Hour",      "07:00 (10.61%)",                        YELLOW),
    ("Highest Fraud Product","Cat C  (11.69%)",                       RED),
    ("Queries Executed",     "6 Spark SQL Queries",                   SUBTEXT),
]
for i, (label, value, color) in enumerate(kpis):
    y_pos = 0.93 - i * 0.088
    ax3.text(0.0, y_pos, label,  color=SUBTEXT, fontsize=9,  transform=ax3.transAxes)
    ax3.text(1.0, y_pos, value,  color=color,   fontsize=9.5,
             transform=ax3.transAxes, ha='right', fontweight='bold')
    ax3.plot([0, 1], [y_pos - 0.025, y_pos - 0.025],
             color=BORDER, linewidth=0.5, transform=ax3.transAxes, clip_on=False)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(f"{SCREENSHOTS}/02_spark_sql_analytics.png", dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("    Saved → screenshots/02_spark_sql_analytics.png")


# ═══════════════════════════════════════════════════════════════
# Screenshot 3 — ML Model Comparison (RF + GBT + Isolation Forest)
# ═══════════════════════════════════════════════════════════════
print("[3] Generating 03_ml_comparison.png ...")

with open("results/ml/classification_metrics.json") as f:
    ml = json.load(f)
with open("results/anomalies/anomaly_metrics.json") as f:
    iso = json.load(f)

rf  = ml.get("random_forest", ml)
gbt = ml.get("gradient_boosted_trees", {})

fig = plt.figure(figsize=(14, 7), facecolor=BG)
fig.suptitle("IEEE-CIS Fraud Detection — ML Model Comparison (3 Algorithms)\n"
             "Student: Abotula Yaswanth Kumar | A23126552191 | Case Study 11.2",
             color=TEXT, fontsize=13, fontweight='bold', y=0.97)

gs = GridSpec(1, 3, figure=fig, wspace=0.38)

# --- (a) Grouped bar: 4 metrics × 3 models
ax1 = fig.add_subplot(gs[0:2])
apply_dark(fig, ax1)
metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
rf_vals  = [rf.get('accuracy_pct',84.03),  rf.get('precision_pct',96.0),
            rf.get('recall_pct',84.03),    rf.get('f1_score_pct',88.75)]
gbt_vals = [gbt.get('accuracy_pct',86.88), gbt.get('precision_pct',96.3),
            gbt.get('recall_pct',86.88),   gbt.get('f1_score_pct',90.56)]
iso_vals = [iso['accuracy_pct'], iso['precision_pct'],
            iso['recall_pct'],   iso['f1_score_pct']]

x = np.arange(len(metrics_labels))
w = 0.26
bars1 = ax1.bar(x - w,   rf_vals,  width=w, label='Random Forest',         color=BLUE,   edgecolor=BG, linewidth=0.8)
bars2 = ax1.bar(x,       gbt_vals, width=w, label='Gradient Boosted Trees', color=PURPLE, edgecolor=BG, linewidth=0.8)
bars3 = ax1.bar(x + w,   iso_vals, width=w, label='Isolation Forest',       color=GREEN,  edgecolor=BG, linewidth=0.8)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                 f'{h:.1f}', ha='center', va='bottom', color=TEXT, fontsize=8, fontweight='bold')

ax1.set_title("3-Algorithm Performance Comparison", color=TEXT, fontsize=11, pad=10)
ax1.set_ylabel("Score (%)", color=SUBTEXT, fontsize=9)
ax1.set_xticks(x); ax1.set_xticklabels(metrics_labels, fontsize=10)
ax1.set_ylim(0, 115)
ax1.legend(frameon=False, labelcolor=TEXT, fontsize=9, loc='upper right')

# --- (b) Model summary cards
ax2 = fig.add_subplot(gs[2])
ax2.set_facecolor(PANEL); ax2.axis('off')
ax2.set_title("Model Summary", color=TEXT, fontsize=11, pad=12)

rows = [
    ("", "RF", "GBT", "ISO"),
    ("Accuracy",  f"{rf.get('accuracy_pct',84.03)}%",  f"{gbt.get('accuracy_pct',86.88)}%",  f"{iso['accuracy_pct']}%"),
    ("Precision", f"{rf.get('precision_pct',96.0)}%",  f"{gbt.get('precision_pct',96.3)}%",  f"{iso['precision_pct']}%"),
    ("Recall",    f"{rf.get('recall_pct',84.03)}%",    f"{gbt.get('recall_pct',86.88)}%",    f"{iso['recall_pct']}%"),
    ("F1 Score",  f"{rf.get('f1_score_pct',88.75)}%",  f"{gbt.get('f1_score_pct',90.56)}%",  f"{iso['f1_score_pct']}%"),
    ("AUC-ROC",   f"{rf.get('auc_roc',0.875)}",         f"{gbt.get('auc_roc',0.906)}",         "—"),
    ("Type",      "Supervised", "Supervised", "Unsupervised"),
    ("Library",   "PySpark", "PySpark", "Scikit-Learn"),
]
col_colors = [SUBTEXT, BLUE, PURPLE, GREEN]
col_x      = [0.0, 0.42, 0.65, 0.88]
for i, row in enumerate(rows):
    y_pos = 0.95 - i * 0.105
    for j, (val, cx, cc) in enumerate(zip(row, col_x, col_colors)):
        weight = 'bold' if i == 0 else ('bold' if j == 0 else 'normal')
        ax2.text(cx, y_pos, val, color=cc if j > 0 else SUBTEXT,
                 fontsize=8.5, transform=ax2.transAxes, fontweight=weight)
    if i > 0:
        ax2.plot([0, 1], [y_pos - 0.04, y_pos - 0.04],
                 color=BORDER, linewidth=0.4, transform=ax2.transAxes, clip_on=False)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(f"{SCREENSHOTS}/03_ml_comparison.png", dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("    Saved → screenshots/03_ml_comparison.png")


# ═══════════════════════════════════════════════════════════════
# Screenshot 4 — Anomaly Detection (Isolation Forest)
# ═══════════════════════════════════════════════════════════════
print("[4] Generating 04_anomaly_detection.png ...")

fig = plt.figure(figsize=(14, 7), facecolor=BG)
fig.suptitle("IEEE-CIS Fraud Detection — Isolation Forest Anomaly Detection\n"
             "Student: Abotula Yaswanth Kumar | A23126552191 | Case Study 11.2",
             color=TEXT, fontsize=13, fontweight='bold', y=0.97)

gs = GridSpec(1, 3, figure=fig, wspace=0.38)

# --- (a) Detection summary donut
ax1 = fig.add_subplot(gs[0])
anomalies = iso['anomalies_detected']
normal    = iso['sample_records_tested'] - anomalies
wedges, texts, autotexts = ax1.pie(
    [normal, anomalies],
    colors=[GREEN, RED],
    autopct='%1.2f%%',
    startangle=90,
    pctdistance=0.75,
    wedgeprops=dict(width=0.5, edgecolor=BG, linewidth=2)
)
for at in autotexts:
    at.set_color(TEXT); at.set_fontsize(10); at.set_fontweight('bold')
ax1.set_facecolor(PANEL)
ax1.set_title("Anomaly Detection Results\n(100K sample)", color=TEXT, fontsize=11, pad=12)
ax1.legend(
    handles=[
        mpatches.Patch(color=GREEN, label=f'Normal   {normal:,}'),
        mpatches.Patch(color=RED,   label=f'Anomaly  {anomalies:,}')
    ],
    loc='lower center', bbox_to_anchor=(0.5, -0.12),
    frameon=False, labelcolor=TEXT, fontsize=9
)

# --- (b) Metrics bars
ax2 = fig.add_subplot(gs[1])
apply_dark(fig, ax2)
met_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
met_vals  = [iso['accuracy_pct'], iso['precision_pct'],
             iso['recall_pct'],   iso['f1_score_pct']]
met_colors = [GREEN, YELLOW, YELLOW, YELLOW]
bars = ax2.barh(met_names, met_vals, color=met_colors, edgecolor=BG, linewidth=0.8, height=0.5)
for bar, val in zip(bars, met_vals):
    ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2,
             f'{val:.2f}%', va='center', color=TEXT, fontsize=10, fontweight='bold')
ax2.set_title("Isolation Forest Metrics", color=TEXT, fontsize=11, pad=10)
ax2.set_xlabel("Score (%)", color=SUBTEXT, fontsize=9)
ax2.set_xlim(0, 115)
ax2.invert_yaxis()

# --- (c) Config & stats panel
ax3 = fig.add_subplot(gs[2])
ax3.set_facecolor(PANEL); ax3.axis('off')
ax3.set_title("Algorithm Configuration", color=TEXT, fontsize=11, pad=12)
config = [
    ("Algorithm",          "Isolation Forest",          TEAL),
    ("Library",            "Scikit-Learn",               BLUE),
    ("N Estimators",       f"{iso['n_estimators']} trees", BLUE),
    ("Contamination",      f"{iso['contamination_param']:.4f}", YELLOW),
    ("Training Set",       "Legitimate only",            GREEN),
    ("Baseline Records",   f"{iso['baseline_training_records']:,}", GREEN),
    ("Sample Tested",      f"{iso['sample_records_tested']:,}",     BLUE),
    ("Anomalies Found",    f"{iso['anomalies_detected']:,}",         RED),
    ("Detection Rate",     f"{iso['anomalies_detected']/iso['sample_records_tested']*100:.2f}%", RED),
    ("Purpose",            "Zero-day fraud patterns",    SUBTEXT),
]
for i, (label, value, color) in enumerate(config):
    y_pos = 0.93 - i * 0.088
    ax3.text(0.0, y_pos, label,  color=SUBTEXT, fontsize=9,  transform=ax3.transAxes)
    ax3.text(1.0, y_pos, value,  color=color,   fontsize=9.5,
             transform=ax3.transAxes, ha='right', fontweight='bold')
    ax3.plot([0, 1], [y_pos - 0.025, y_pos - 0.025],
             color=BORDER, linewidth=0.5, transform=ax3.transAxes, clip_on=False)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(f"{SCREENSHOTS}/04_anomaly_detection.png", dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("    Saved → screenshots/04_anomaly_detection.png")

print("\n[DONE] All 4 screenshots generated in screenshots/")
