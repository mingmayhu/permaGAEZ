"""
Figure: Wilcoxon Median ΔSuitability — Single Panel
=====================================================
Single horizontal bar chart:
  Median annual ΔSuitability per crop (Wilcoxon, purple = sig positive)

Same width as timeseries figure (16), taller to give bars room.

Input:
  ./results/permafrost_thaw_impact/thaw_vs_nothaw/figure_exports/
      wilcoxon_results.csv

Output:
  ./results/ch6_permafrost_thaw_impact/thaw_vs_nothaw/figures/
      fig_wilcoxon_single.png
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
WORK_DIR     = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
WILCOXON_CSV = (r'./results/ch6_permafrost_thaw_impact/thaw_vs_nothaw/figure_exports/'
                r'wilcoxon_results.csv')
OUT_DIR      = r'./results/ch6_permafrost_thaw_impact/thaw_vs_nothaw/figures'
OUT_PATH     = f'{OUT_DIR}/fig_wilcoxon_single.png'
DPI          = 300

FONT      = 'Helvetica'
BOLD_PATH = '/Users/ming-mayhu/Library/Fonts/Helvetica LT 75 Bold.ttf'
REG_PATH  = '/System/Library/Fonts/Helvetica.ttc'

COLOR_SIG = "#0571b0"
COLOR_NS  = '#AAAAAA'

CROP_ORDER = [
    'Spring rapeseed', 'Spring oat', 'White potato', 'Silage maize',
    'Winter wheat', 'Winter barley', 'Dry pea',
    'Spring barley', 'Spring wheat', 'Winter rapeseed',
]

os.chdir(WORK_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# ── Fonts ─────────────────────────────────────────────────────────────────────
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
        'font.family':      'sans-serif',
        'font.sans-serif':  [FONT],
        'xtick.direction':  'out',
        'ytick.direction':  'out',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'axes.edgecolor':   '#000000',
        'axes.linewidth':   1.5,
    }
)

# ── Load data ─────────────────────────────────────────────────────────────────
df_w = pd.read_csv(WILCOXON_CSV)
print('Crops in CSV:', df_w['crop'].tolist())

# Separate Overall if already in CSV
overall_in_csv = df_w[df_w['crop'] == 'Overall']
df_w_crops     = df_w[df_w['crop'] != 'Overall']

ordered   = [c for c in CROP_ORDER if c in df_w_crops['crop'].values and c != 'Overall']
remaining = [c for c in df_w_crops['crop'].values if c not in ordered]
df_w_crops = df_w_crops.set_index('crop').reindex(ordered + remaining).reset_index()

# Re-attach Overall at bottom if it was in the CSV
if not overall_in_csv.empty:
    df_w = pd.concat([df_w_crops, overall_in_csv], ignore_index=True)
else:
    df_w = df_w_crops

# Add Overall row if available
DELTA_CSV = (r'./results/ch6_permafrost_thaw_impact/thaw_vs_nothaw/outputs/'
             r'1_delta_maps/annual_delta_suit_all_crops.csv')
if os.path.exists(DELTA_CSV):
    from scipy.stats import wilcoxon as wilcoxon_test
    df_delta   = pd.read_csv(DELTA_CSV)
    overall_ts = df_delta.groupby('year')['mean_delta'].mean()
    valid      = overall_ts.dropna().values
    if len(valid) >= 4:
        _, p_pos = wilcoxon_test(valid, alternative='greater')
        overall_row = pd.DataFrame([{
            'crop':               'Overall',
            'median_delta':       float(np.median(valid)),
            'pct_years_positive': float(np.mean(valid > 0) * 100),
            'p_greater_zero':     round(p_pos, 4),
            'sig_positive':       p_pos < 0.05,
        }])
        df_w = pd.concat([df_w, overall_row], ignore_index=True)
else:
    # Compute Overall Wilcoxon across per-crop median deltas
    from scipy.stats import wilcoxon as wilcoxon_test
    crop_medians = df_w[df_w['crop'] != 'Overall']['median_delta'].dropna().values
    _, p_pos = wilcoxon_test(crop_medians, alternative='greater')
    overall_row = pd.DataFrame([{
        'crop':               'Overall',
        'median_delta':       float(np.median(crop_medians)),
        'pct_years_positive': float(np.mean(crop_medians > 0) * 100),
        'p_greater_zero':     round(p_pos, 4),
        'sig_positive':       p_pos < 0.05,
    }])
    df_w = pd.concat([df_w, overall_row], ignore_index=True)

crop_order_full = list(df_w[df_w['crop'] != 'Overall']['crop']) + ['Overall']
df_w = df_w.set_index('crop').reindex(crop_order_full).reset_index()

# ── Figure — same width as timeseries (16), taller ───────────────────────────
fig, ax = plt.subplots(figsize=(9.2, 10))
fig.patch.set_facecolor('white')

colors = [(COLOR_SIG if s else COLOR_NS)
          for s in df_w['sig_positive']]
bars   = ax.barh(df_w['crop'], df_w['median_delta'],
                 color=colors, edgecolor='white', height=0.6)
ax.axvline(0, color='black', linewidth=0.8)

# Separator line above Overall
if 'Overall' in df_w['crop'].values:
    overall_yi = df_w[df_w['crop'] == 'Overall'].index[0]
    ax.axhline(overall_yi - 0.5, color='#cccccc', linewidth=0.8, linestyle=':')

# p-value annotations
for bar, (_, row) in zip(bars, df_w.iterrows()):
    x   = bar.get_width()
    sig = row['sig_positive']
    ax.text(
        x + (0.0003 if x >= 0 else -0.0003),
        bar.get_y() + bar.get_height() / 2,
        f'p = {row["p_greater_zero"]:.3f}',
        va='center', fontsize=16,
        ha='left' if x >= 0 else 'right',
        color=(COLOR_SIG if sig else COLOR_NS),
        fontproperties=fp_bold if sig else fp_reg
    )

ax.set_xlim(-0.012, 0.052)
ax.set_xlabel('Median Δsuitability', fontsize=16, fontproperties=fp_reg)
ax.tick_params(labelsize=16)
ax.set_yticklabels(df_w['crop'], rotation=45, ha='right', fontsize=16)

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {OUT_PATH}')