"""
Figure: Permutation Test on Sen's Slope Difference (Thaw − No-Thaw)
====================================================================
2-panel dot-and-CI plot:
  Left  — Mean suitability score slope difference
  Right — % suitable land slope difference

Each panel:
  - One row per crop + OVERALL (separated by dashed line)
  - Dot = slope difference, whiskers = 95% CI from permutation test
  - Blue = significant (p < 0.05), grey = not significant, black = OVERALL
  - Vertical dashed line at 0
  - p-value annotated per row

Input:
  ./results/permafrost_thaw_impact/thaw_vs_nothaw/figure_exports/
      permutation_slope_diff.csv

Output:
  ./results/permafrost_thaw_impact/thaw_vs_nothaw/figures/
      fig_thaw_permutation.png
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
WORK_DIR = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
CSV_PATH = (r'./results/permafrost_thaw_impact/thaw_vs_nothaw/figure_exports/'
            r'permutation_slope_diff.csv')
OUT_DIR  = r'./results/permafrost_thaw_impact/thaw_vs_nothaw/figures'
OUT_PATH = f'{OUT_DIR}/fig_thaw_permutation.png'
DPI      = 300

FONT      = 'Helvetica'
BOLD_PATH = '/Users/ming-mayhu/Library/Fonts/Helvetica LT 75 Bold.ttf'
REG_PATH  = '/System/Library/Fonts/Helvetica.ttc'

PERIOD = '1979-2018'

# Crop order — OVERALL at top, per-crop below sorted by slope diff
CROP_ORDER = [
    'OVERALL',
    'Winter Barley', 'Spring Barley', 'Winter Wheat', 'Spring Wheat',
    'Silage Maize', 'White Potato', 'Oat', 'Dry Pea',
    'Winter Rape', 'Spring Rape',
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
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'xtick.direction':   'out',
        'ytick.direction':   'out',
        'xtick.major.size':  4,
        'ytick.major.size':  4,
        'axes.edgecolor':    '#000000',
        'axes.linewidth':    0.8,
    }
)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df = df[df['period'] == PERIOD]

# ── Panel definitions ─────────────────────────────────────────────────────────
panels = [
    {
        'metric'  : 'mean_suit',
        'xlabel'  : "Sen's Slope Difference (class/yr)",
        'title'   : 'Mean Suitability Score',
        'ci_lo'   : 'perm_ci_lo',
        'ci_hi'   : 'perm_ci_hi',
        'p_col'   : 'perm_p',
        'sig_col' : 'perm_sig',
    },
    {
        'metric'  : 'pct_ge2',
        'xlabel'  : "Sen's Slope Difference (%/yr)",
        'title'   : '% Suitable Land (≥ Class 2)',
        'ci_lo'   : 'perm_ci_lo',
        'ci_hi'   : 'perm_ci_hi',
        'p_col'   : 'perm_p',
        'sig_col' : 'perm_sig',
    },
]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 8))
fig.patch.set_facecolor('white')

for ax, panel in zip(axes, panels):
    df_m = df[df['metric'] == panel['metric']].copy()

    # Deduplicate OVERALL (merged CSV may contain it twice), keep first
    df_overall = df_m[df_m['crop'] == 'OVERALL'].head(1)
    df_crops   = df_m[df_m['crop'] != 'OVERALL'].sort_values('slope_difference')
    df_plot    = pd.concat([df_overall, df_crops], ignore_index=True)

    y             = np.arange(len(df_plot))
    p_annotations = []   # collect for axis-aware offset after loop
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--', zorder=1)

    # Dashed separator below OVERALL
    ax.axhline(0.5, color='#cccccc', linewidth=0.8, linestyle=':', zorder=1)

    for yi, (_, row) in enumerate(df_plot.iterrows()):
        is_overall = row['crop'] == 'OVERALL'
        sig        = bool(row[panel['sig_col']]) if not pd.isna(row[panel['sig_col']]) else False

        if is_overall:
            color = '#000000'
        elif sig:
            color = '#2166AC'
        else:
            color = '#AAAAAA'

        ci_lo = row[panel['ci_lo']]
        ci_hi = row[panel['ci_hi']]
        val   = row['slope_difference']
        p_val = row[panel['p_col']]
        sig_str = '★' if sig else ''

        # CI whisker
        ax.plot([ci_lo, ci_hi], [yi, yi],
                color=color, linewidth=2, alpha=0.8, zorder=3)

        # Dot
        marker = 'D' if is_overall else 'o'
        ms     = 8 if is_overall else 6
        ax.scatter(val, yi, color=color, zorder=5,
                   s=ms**2, marker=marker)

        # p-value annotation — store for post-loop placement with axis-aware offset
        p_annotations.append((max(ci_hi, val), yi, p_val, sig_str, sig, color))

    # Place p-value annotations — all aligned to same x position
    ax.figure.canvas.draw()
    x_lo, x_hi = ax.get_xlim()
    x_range    = x_hi - x_lo
    # Use a single x position for all labels: rightmost CI + small fixed gap
    max_ci_hi  = max(a[0] for a in p_annotations)
    x_label    = max_ci_hi + x_range * 0.04

    for _, yi, p_val, sig_str, sig, color in p_annotations:
        p_str = f'p={p_val:.3f}{sig_str}' if not pd.isna(p_val) else ''
        ax.text(x_label, yi, p_str,
                va='center', fontsize=8, color=color,
                fontproperties=fp_bold if sig else fp_reg)

    ax.set_yticks(y)
    ax.set_yticklabels(df_plot['crop'], fontsize=10)
    ax.set_xlabel(panel['xlabel'], fontsize=11, fontproperties=fp_reg)
    ax.set_title(panel['title'], fontsize=12, fontproperties=fp_bold, pad=8)
    ax.tick_params(axis='x', labelsize=9)

fig.suptitle(
    "Does Permafrost Thaw Amplify the Suitability Trend?\n"
    "Permutation Test on Sen's Slope Difference (Thaw − No-Thaw)\n"
    "Black = Overall  |  Blue = significant (p < 0.05)  |  Grey = not significant",
    fontsize=12, fontproperties=fp_bold, y=1.01
)

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {OUT_PATH}')