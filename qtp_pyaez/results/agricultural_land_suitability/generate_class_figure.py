"""
Plot Overall Suitability Class Distribution — Chapter 5
=========================================================
Reads pre-computed CSV and produces a stacked area chart
showing classes 2-5 over 1979-2018, with Sen's slope trend
line per class.

Input:
  ./results/agricultural_land_suitability/outputs/csv/overall_class_distribution.csv

Output:
  ./results/agricultural_land_suitability/outputs/overall_class_distribution.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
import seaborn as sns
from pymannkendall import original_test as mk_test

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR  = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
CSV_PATH  = r'./results/agricultural_land_suitability/outputs/csv/overall_class_distribution.csv'
OUT_PATH  = r'./results/agricultural_land_suitability/outputs/overall_class_distribution.png'

ALPHA = 0.05
DPI   = 300

os.chdir(WORK_DIR)

# ── Font setup ────────────────────────────────────────────────────────────────
FONT      = 'Helvetica'
BOLD_PATH = '/Users/ming-mayhu/Library/Fonts/Helvetica LT 75 Bold.ttf'
REG_PATH  = '/System/Library/Fonts/Helvetica.ttc'

# ── Seaborn theme ─────────────────────────────────────────────────────────────
sns.set_theme(
    style='ticks',
    rc={
        'font.family':       'sans-serif',
        'font.sans-serif':   [FONT],
        # 'axes.spines.top':   False,
        # 'axes.spines.right': False,
        'xtick.direction':   'out',
        'ytick.direction':   'out',
        'xtick.major.size':  4,
        'ytick.major.size':  4,
        'axes.edgecolor':    '#000000',
        'axes.linewidth':    0.8,
        'axes.facecolor':     '#fffcce',
    }
)

CLASS_COLORS = {
    2: '#c2e699',
    3: '#78c679',
    4: '#31a354',
    5: '#006837',
    6:  '#fffcce',
}
# Darker versions of each class colour for the trend lines
TREND_COLORS = {
    2: '#7ab84e',
    3: '#3d8a3d',
    4: '#0f6e2a',
    5: '#003d1f',
}
CLASS_LABELS = {
    1: 'Class 1 (not suitable)',  # combined class 0 and 1
    2: 'Class 2 (marginally suitable)',
    3: 'Class 3 (moderately suitable)',
    4: 'Class 4 (suitable)',
    5: 'Class 5 (very suitable)',
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def run_mk(series):
    s     = np.array(series, dtype=float)
    valid = np.isfinite(s)
    if valid.sum() < 4:
        return None
    mk   = mk_test(s[valid])
    line = np.full(len(s), np.nan)
    line[valid] = mk.intercept + mk.slope * np.arange(valid.sum())
    pstr = 'p < 0.001' if mk.p < 0.001 else f'p = {mk.p:.3f}'
    return {
        'slope': mk.slope, 'p': mk.p, 'pstr': pstr,
        'significant': mk.p < ALPHA, 'sen_line': line,
    }

# ── Load data ─────────────────────────────────────────────────────────────────
df        = pd.read_csv(CSV_PATH)
years_arr = df['Year'].values.astype(int)

# ── Run MK per class ──────────────────────────────────────────────────────────
mk_results = {}
for c in [2, 3, 4, 5]:
    mk_results[c] = run_mk(df[f'pct_class_{c}'].values)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor('white')

fp_bold   = FontProperties(fname=BOLD_PATH, size=16)
fp_legend = FontProperties(fname=REG_PATH,  size=14)

# Stacked area — classes 2-5
bottom = np.zeros(len(years_arr))
cumulative = {}  # store cumulative bottom for each class to position trend lines
for c in [2, 3, 4, 5]:
    vals = df[f'pct_class_{c}'].values
    vals = np.where(np.isfinite(vals), vals, 0.0)
    cumulative[c] = bottom + vals / 2  # midpoint of each band for trend line
    ax.fill_between(years_arr, bottom, bottom + vals,
                    color=CLASS_COLORS[c], alpha=0.85)
    bottom += vals

# Trend lines — drawn at the cumulative top of each class band
cumulative_top = np.zeros(len(years_arr))
for c in [2, 3, 4, 5]:
    vals = df[f'pct_class_{c}'].values
    vals = np.where(np.isfinite(vals), vals, 0.0)
    cumulative_top = cumulative_top + vals
    mk = mk_results[c]
    if mk:
        # Offset Sen's line to sit at the top of the stacked band
        # by computing the Sen's line for cumulative_top
        mk_cum = run_mk(cumulative_top)
        if mk_cum:
            ax.plot(years_arr, mk_cum['sen_line'],
                    color=TREND_COLORS[c], linewidth=1.8,
                    linestyle='--', dashes=(5, 3),
                    zorder=5)

# Axis formatting
ax.set_xlim(1979, 2018)
ax.set_ylim(0, None)
xtick_years = years_arr[::5]
ax.set_xticks(xtick_years)
ax.set_xticklabels([str(y) for y in xtick_years],
                   rotation=45, ha='right', fontsize=16)
ax.set_ylabel('% Land with class \u2265 2', fontsize=16, labelpad=6)
ax.set_xlabel('Year', fontsize=16, labelpad=4)
ax.tick_params(which='major', labelsize=16, length=4,
               color='#000000', width=0.8)
ax.tick_params(which='minor', length=2, color='#999999', width=0.6)
ax.spines['left'].set_color('#000000')
ax.spines['bottom'].set_color('#000000')


# ── Legend ────────────────────────────────────────────────────────────────────
# Class fill patches
fill_handles = [mpatches.Patch(color=CLASS_COLORS[c], alpha=0.85,
                               label=CLASS_LABELS[c])
                for c in [2, 3, 4, 5]]

# Trend line entries with slope info
trend_handles = []
for c in [2, 3, 4, 5]:
    mk = mk_results[c]
    if mk:
        lbl = f"Class {c} sen's slope: {mk['slope']:+.4f}%/yr ({mk['pstr']})"
    else:
        lbl = f'Class {c} trend: n/a'
    trend_handles.append(
        Line2D([0], [0], color=TREND_COLORS[c], linewidth=2,
               linestyle='--', dashes=(5, 3), label=lbl)
    )

leg = ax.legend(handles=list(reversed(fill_handles)) + list(reversed(trend_handles)),
                loc='upper left',
                bbox_to_anchor=(0.2, -.2),
                bbox_transform=ax.transAxes,
                framealpha=0.9, edgecolor='none',
                borderpad=0.5, ncol=2)
for text in leg.get_texts():
    text.set_fontproperties(fp_legend)

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=DPI, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f'Saved to {OUT_PATH}')