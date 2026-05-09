"""
Figure: Wilcoxon + Permutation Test Results
============================================
2-panel horizontal bar chart:
  Left  — Median annual ΔSuitability per crop (Wilcoxon, purple = sig positive)
  Right — Sen's slope difference Thaw − No-Thaw (Permutation, purple = sig)

Input:
  ./results/permafrost_thaw_impact/thaw_vs_nothaw/figure_exports/
      wilcoxon_results.csv
      permutation_slope_diff.csv

Output:
  ./results/permafrost_thaw_impact/thaw_vs_nothaw/figures/
      fig_thaw_wilcoxon.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR      = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
WILCOXON_CSV  = (r'./results/permafrost_thaw_impact/thaw_vs_nothaw/figure_exports/'
                 r'wilcoxon_results.csv')
PERM_CSV      = (r'./results/permafrost_thaw_impact/thaw_vs_nothaw/figure_exports/'
                 r'permutation_slope_diff.csv')
OUT_DIR       = r'./results/permafrost_thaw_impact/thaw_vs_nothaw/figures'
OUT_PATH      = f'{OUT_DIR}/fig_thaw_wilcoxon.png'
DPI           = 300

FONT      = 'Helvetica'
BOLD_PATH = '/Users/ming-mayhu/Library/Fonts/Helvetica LT 75 Bold.ttf'
REG_PATH  = '/System/Library/Fonts/Helvetica.ttc'

COLOR_SIG = '#8b1889'
COLOR_NS  = '#AAAAAA'

PERIOD = '1979-2018'
METRIC = 'mean_suit'   # use mean suitability score for permutation panel

# Crop order — sorted by Wilcoxon median delta descending
CROP_ORDER = [
    'Spring Rape', 'Oat', 'White Potato', 'Silage Maize',
    'Winter Wheat', 'Winter Barley', 'Dry Pea',
    'Spring Barley', 'Spring Wheat', 'Winter Rape',
]

os.chdir(WORK_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# ── Font setup ────────────────────────────────────────────────────────────────
try:
    fp_bold = FontProperties(fname=BOLD_PATH)
    fp_reg  = FontProperties(fname=REG_PATH)
except Exception:
    fp_bold = FontProperties(weight='bold')
    fp_reg  = FontProperties()

# ── Seaborn theme ─────────────────────────────────────────────────────────────
sns.set_theme(
    style='ticks',
    rc={
        'font.family':       'sans-serif',
        'font.sans-serif':   [FONT],
        'xtick.direction':   'out',
        'ytick.direction':   'out',
        'xtick.major.size':  4,
        'ytick.major.size':  4,
        'axes.edgecolor':    '#000000',
        'axes.linewidth':    0.8,
    }
)

# ── Load and align data ───────────────────────────────────────────────────────
df_w = pd.read_csv(WILCOXON_CSV)
ordered   = [c for c in CROP_ORDER if c in df_w['crop'].values]
remaining = [c for c in df_w['crop'].values if c not in ordered]
df_w = df_w.set_index('crop').reindex(ordered + remaining).reset_index()

# Compute Overall Wilcoxon from the per-crop annual delta CSV
# Overall = mean delta across all crops per year
DELTA_CSV = (r'./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/'
             r'1_delta_maps/annual_delta_suit_all_crops.csv')
if os.path.exists(DELTA_CSV):
    from scipy.stats import wilcoxon as wilcoxon_test
    df_delta    = pd.read_csv(DELTA_CSV)
    Overall_ts  = df_delta.groupby('year')['mean_delta'].mean()
    valid       = Overall_ts.dropna().values
    if len(valid) >= 4:
        _, p_two = wilcoxon_test(valid, alternative='two-sided')
        _, p_pos = wilcoxon_test(valid, alternative='greater')
        Overall_row = pd.DataFrame([{
            'crop'              : 'Overall',
            'median_delta'      : float(np.median(valid)),
            'pct_years_positive': float(np.mean(valid > 0) * 100),
            'p_greater_zero'    : round(p_pos, 4),
            'sig_positive'      : p_pos < 0.05,
        }])
        df_w = pd.concat([df_w, Overall_row], ignore_index=True)
        print(f'  Overall Wilcoxon: median={np.median(valid):.5f}, p(greater)={p_pos:.4f}')
else:
    print(f'  WARNING: {DELTA_CSV} not found — Overall row omitted from Wilcoxon panel')

df_p = pd.read_csv(PERM_CSV)
df_p = df_p[(df_p['period'] == PERIOD) & (df_p['metric'] == METRIC)].copy()
# Deduplicate Overall
df_p = pd.concat([
    df_p[df_p['crop'] == 'Overall'].head(1),
    df_p[df_p['crop'] != 'Overall']
], ignore_index=True)
# Align both panels to same crop order: per-crop + Overall at bottom
crop_order_full = list(df_w[df_w['crop'] != 'Overall']['crop']) + ['Overall']
df_w = df_w.set_index('crop').reindex(crop_order_full).reset_index()
df_p = df_p.set_index('crop').reindex(crop_order_full).reset_index()

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
fig.patch.set_facecolor('white')

# ── Left panel: Wilcoxon median ΔSuitability ──────────────────────────────────
ax1     = axes[0]
colors1 = [(COLOR_SIG if s else COLOR_NS)
           for c, s in zip(df_w['crop'], df_w['sig_positive'])]
bars1   = ax1.barh(df_w['crop'], df_w['median_delta'],
                   color=colors1, edgecolor='white', height=0.6)
ax1.axvline(0, color='black', linewidth=0.8)

# Separator line above Overall
if 'Overall' in df_w['crop'].values:
    Overall_yi = df_w[df_w['crop'] == 'Overall'].index[0]
    ax1.axhline(Overall_yi - 0.5, color='#cccccc', linewidth=0.8, linestyle=':')

for bar, (_, row) in zip(bars1, df_w.iterrows()):
    x   = bar.get_width()
    sig = row['sig_positive']
    ax1.text(
        x + (0.0003 if x >= 0 else -0.0003),
        bar.get_y() + bar.get_height() / 2,
        f'p = {row["p_greater_zero"]:.3f}',
        va='center', fontsize=12,
        ha='left' if x >= 0 else 'right',
        color=(COLOR_SIG if sig else COLOR_NS),
        fontproperties=fp_bold if sig else fp_reg
    )

ax1.set_xlim(-0.01, 0.055)
ax1.set_xlabel('Median Δsuitability', fontsize=12,
               fontproperties=fp_reg)
ax1.set_title('(a) Wilcoxon signed rank', loc='left' ,fontsize=12, pad=8)
ax1.tick_params(labelsize=12)

# ── Right panel: Permutation slope difference ─────────────────────────────────
ax2     = axes[1]
colors2 = [(COLOR_SIG if s else COLOR_NS)
           for c, s in zip(df_p['crop'], df_p['perm_sig'].fillna(False).astype(bool))]
bars2   = ax2.barh(df_p['crop'], df_p['slope_difference'],
                   color=colors2, edgecolor='white', height=0.6)
ax2.axvline(0, color='black', linewidth=0.8)

# Separator line above Overall
if 'Overall' in df_p['crop'].values:
    print(df_p['crop'], df_p['slope_difference'])
    Overall_yi = df_p[df_p['crop'] == 'Overall'].index[0]
    ax2.axhline(Overall_yi - 0.5, color='#cccccc', linewidth=0.8, linestyle=':')

# p-value annotations — inline next to bar, same style as Wilcoxon panel
for bar, (_, row) in zip(bars2, df_p.iterrows()):
    x          = bar.get_width()
    sig        = bool(row['perm_sig']) if not pd.isna(row['perm_sig']) else False
    is_Overall = row['crop'] == 'Overall'
    color      = (COLOR_SIG if sig else COLOR_NS)
    ax2.text(
        x + (0.00002 if x >= 0 else -0.00002),
        bar.get_y() + bar.get_height() / 2,
        f'p = {row["perm_p"]:.3f}' if not pd.isna(row['perm_p']) else '',
        va='center', fontsize=12,
        ha='left' if x >= 0 else 'right',
        color=color,
        fontproperties=fp_bold if sig else fp_reg
    )

ax2.set_xlabel("Sen's slope difference (mean suitability/yr)", fontsize=12,
               fontproperties=fp_reg)
ax2.set_title('(b) Slope permutation', loc='left',
              fontsize=12, pad=8)
ax2.tick_params(labelsize=12)
ax2.set_xlim(-0.0003, 0.0015)

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {OUT_PATH}')