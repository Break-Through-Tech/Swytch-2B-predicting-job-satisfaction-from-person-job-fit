import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ==============================================================================
# 0. GLOBAL STYLING & OUTPUT DIRECTORY
# ==============================================================================
OUT_DIR = 'presentation_visuals'
os.makedirs(OUT_DIR, exist_ok=True)

NAVY = '#1D3557'
TEAL = '#2A9D8F'
CORAL = '#E76F51'
SLATE = '#457B9D'
GOLD = '#E9C46A'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.edgecolor': '#CCCCCC',
    'axes.linewidth': 0.8,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

def clean_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ==============================================================================
# 1. VISUAL 1: OCCUPATION SATISFACTION (ADVISOR MAIN GOAL)
# ==============================================================================
occ_groups = [
    'Production, Transport\n& Material Moving',
    'Military / Other',
    'Sales & Office',
    'Service Occupations',
    'Management, Business,\nScience & Arts',
    'Natural Resources,\nConstruction & Maint.'
]
group_pcts = [37.36, 40.48, 40.87, 44.19, 50.76, 53.04]

specific_jobs = [
    'Food Prep & Serving Supervisors',
    'Assemblers & Fabricators',
    'Industrial Truck Operators',
    'Couriers & Messengers',
    'Cashiers',
    'Construction Managers',
    'Chief Executives (CEOs)',
    'Construction Trade Supervisors',
    'Hairdressers & Cosmetologists',
    'Recreation & Fitness Workers'
]
specific_pcts = [13.51, 21.57, 21.88, 25.45, 25.66, 64.18, 68.66, 71.11, 72.73, 75.00]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 1.15]})

# Left Panel: Major Groups
bars1 = ax1.barh(occ_groups, group_pcts, color=SLATE, height=0.6)
bars1[-1].set_color(TEAL)
bars1[0].set_color(CORAL)
ax1.set_title('Job Satisfaction by Major Occupation Group\n(Chi-Square = 117.4, p = 1.09e-23)')
ax1.set_xlabel('% Very Satisfied')
ax1.set_xlim(0, 65)
clean_spines(ax1)
ax1.grid(axis='x', linestyle='--', alpha=0.4)
for bar in bars1:
    ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
             f'{bar.get_width():.1f}%', va='center', fontweight='bold', fontsize=10)

# Right Panel: Top 5 vs Bottom 5
colors2 = [CORAL if v < 40 else TEAL for v in specific_pcts]
bars2 = ax2.barh(specific_jobs, specific_pcts, color=colors2, height=0.65)
ax2.set_title('61.5% Spread: Top 5 vs. Bottom 5 Occupations (n >= 30)')
ax2.set_xlabel('% Very Satisfied')
ax2.set_xlim(0, 88)
clean_spines(ax2)
ax2.grid(axis='x', linestyle='--', alpha=0.4)
for bar in bars2:
    ax2.text(bar.get_width() + 1.2, bar.get_y() + bar.get_height()/2,
             f'{bar.get_width():.1f}%', va='center', fontweight='bold', fontsize=9.5)

plt.tight_layout(pad=2.5)
plt.savefig(f'{OUT_DIR}/01_occupation_satisfaction.png', dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# 2. VISUAL 2: MODEL & THRESHOLD EXPERIMENTATION PROGRESSION
# ==============================================================================
experiments = [
    'LogReg\n(Demographics\nOnly)',
    'LogReg\n(O*NET\nMedian Split)',
    'LogReg\n(O*NET\nContinuous)',
    'LogReg\n(O*NET >= 3.0)\n[WINNER]',
    'Random Forest\n(Untuned,\n>= 3.0)',
    'Random Forest\n(Tuned,\nContinuous)',
    'Random Forest\n(Tuned,\n>= 3.0)'
]
auc_scores = [0.6894, 0.6883, 0.6910, 0.6935, 0.6789, 0.6822, 0.6867]
f1_scores  = [np.nan, 0.6330, 0.6330, 0.6330, 0.5525, 0.5999, 0.6057]

x = np.arange(len(experiments))
width = 0.36

fig, ax = plt.subplots(figsize=(13, 5.5))
b_auc = ax.bar(x - width/2, auc_scores, width, label='AUC (ROC)', color=NAVY)
b_f1  = ax.bar(x + width/2, [0 if np.isnan(v) else v for v in f1_scores], width, label='F1 Score', color=TEAL)

b_auc[3].set_color(GOLD)
b_auc[3].set_edgecolor(NAVY)
b_auc[3].set_linewidth(1.5)

ax.set_ylabel('Score')
ax.set_title('Model & Feature Engineering Progression (Thresholds & Hyperparameter Tuning)')
ax.set_xticks(x)
ax.set_xticklabels(experiments, fontsize=9)
ax.set_ylim(0.50, 0.74)
ax.legend(loc='upper right', frameon=True)
clean_spines(ax)
ax.grid(axis='y', linestyle='--', alpha=0.4)

for bar in b_auc:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

for bar, val in zip(b_f1, f1_scores):
    if not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8.5)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/02_model_threshold_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# 3. VISUAL 3: DIVERGING BAR CHART (WINNING LOGISTIC REGRESSION DRIVERS)
# ==============================================================================
lr_labels = [
    'Generally Not Too Happy',
    'Pretty Happy (vs. Very Happy)',
    'Mid-Income ($40k-$50k)',
    'Finds Life Routine',
    'Finds Life Dull',
    'Strong Republican (Party ID)',
    'Upper Class',
    'Middle Class',
    'O*NET: High Independence',
    'Age (Years)'
]
lr_coefs = [-0.4434, -0.3811, -0.2149, -0.1866, -0.1424, 0.0923, 0.1074, 0.1158, 0.2254, 0.2946]

fig, ax = plt.subplots(figsize=(9.5, 5.5))
colors_c = [CORAL if c < 0 else TEAL for c in lr_coefs]
bars_c = ax.barh(lr_labels, lr_coefs, color=colors_c, height=0.65)
ax.axvline(0, color='black', linewidth=0.9)
ax.set_title('Top Drivers of Job Satisfaction & Dissatisfaction (Logistic Regression, >= 3.0)')
ax.set_xlabel('Standardized Coefficient Impact')
ax.set_xlim(-0.55, 0.38)
clean_spines(ax)
ax.grid(axis='x', linestyle='--', alpha=0.4)

for bar in bars_c:
    val = bar.get_width()
    offset = 0.012 if val >= 0 else -0.012
    ha = 'left' if val >= 0 else 'right'
    ax.text(val + offset, bar.get_y() + bar.get_height()/2,
            f'{val:+.3f}', va='center', ha=ha, fontweight='bold', fontsize=9.5)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/03_logreg_drivers.png', dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# 4. VISUAL 4: TOP 10 FEATURES (TUNED RANDOM FOREST, >= 3.0)
# ==============================================================================
rf_labels = [
    'Years of Education',
    'Fair Health',
    'O*NET: High Achievement',
    'Hours Worked / Week',
    'Finds Life Routine',
    'O*NET: High Independence',
    'Never Married',
    'Pretty Happy (vs. Very Happy)',
    'Generally Not Too Happy',
    'Age (Years)'
]
rf_imps = [0.0233, 0.0264, 0.0284, 0.0427, 0.0543, 0.0677, 0.0695, 0.0785, 0.1199, 0.1611]
colors_rf = [TEAL if 'O*NET' in lbl else SLATE for lbl in rf_labels]

fig, ax = plt.subplots(figsize=(9, 5.5))
bars_rf = ax.barh(rf_labels, rf_imps, color=colors_rf, height=0.65)
ax.set_title('Top 10 Predictive Features (Tuned Random Forest, >= 3.0)')
ax.set_xlabel('Feature Importance (Gini)')
ax.set_xlim(0, 0.19)
clean_spines(ax)
ax.grid(axis='x', linestyle='--', alpha=0.4)

for bar in bars_rf:
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
            f'{bar.get_width():.3f}', va='center', fontweight='bold', fontsize=9.5)

legend_elements = [
    Patch(facecolor=TEAL, label='O*NET Occupational Trait'),
    Patch(facecolor=SLATE, label='GSS Survey Demographic')
]
ax.legend(handles=legend_elements, loc='lower right', frameon=True)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/04_rf_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Done! All 4 visuals saved to ./{OUT_DIR}/")