"""
Figure: Combined Elevation & Slope Profile of ΔSuitability
===========================================================
Single figure, one set of axes, shared x-axis (Mean ΔSuitability):
  Left y-axis  — Elevation (m), black line with diamond markers
  Right y-axis — Slope (degrees), grey line with circle markers

Input:
  ./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/7_elevation_slope/
      elevation_slope_stats.csv

Output:
  ./results/permafrost_thaw_impact/thaw_vs_nothaw/figures/
      fig_thaw_elevation_slope.png
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
CSV_PATH = (r'./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/'
            r'7_elevation_slope/elevation_slope_stats.csv')
OUT_DIR  = r'./results/permafrost_thaw_impact/thaw_vs_nothaw/figures'
OUT_PATH = f'{OUT_DIR}/fig_thaw_elevation_slope.png'
DPI      = 300

FONT      = 'Helvetica'
BOLD_PATH = '/Users/ming-mayhu/Library/Fonts/Helvetica LT 75 Bold.ttf'
REG_PATH  = '/System/Library/Fonts/Helvetica.ttc'

MIN_PIXELS  = 10
COLOR_ELEV  = "#9B4F23"   # blue for elevation
COLOR_SLOPE = "#FE94C0"   # red for slope

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

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df = df[df['n_pixels'] >= MIN_PIXELS]

df_elev  = (df[(df['variable'] == 'elevation') & (df['crop'] == 'OVERALL')]
            .sort_values('bin_mid').dropna(subset=['mean_delta']))
df_slope = (df[(df['variable'] == 'slope') & (df['crop'] == 'OVERALL')]
            .sort_values('bin_mid').dropna(subset=['mean_delta']))

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax_elev = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor('white')

# Zero reference line
ax_elev.axvline(0, color='#aaaaaa', linewidth=0.8, linestyle='--', zorder=1)

# ── Left y-axis: elevation ────────────────────────────────────────────────────
l1, = ax_elev.plot(
    df_elev['mean_delta'].values,
    df_elev['bin_mid'].values,
    color=COLOR_ELEV, linewidth=2.5,
    marker='D', markersize=6,
    label='Elevation', zorder=4
)

ax_elev.set_ylabel('Elevation (m)', fontsize=14,
                   color='black', fontproperties=fp_reg)
ax_elev.tick_params(axis='y', labelcolor=COLOR_ELEV, labelsize=14)
ax_elev.set_yticks(df_elev['bin_mid'].values)
ax_elev.set_yticklabels([f'{int(m)} m' for m in df_elev['bin_mid'].values],
                        fontsize=14)

# ── Right y-axis: slope ───────────────────────────────────────────────────────
ax_slope = ax_elev.twinx()
# Re-enable right spine for the slope axis
ax_slope.spines['right'].set_visible(True)
ax_slope.spines['right'].set_color('black')
ax_slope.spines['right'].set_linewidth(0.8)

l2, = ax_slope.plot(
    df_slope['mean_delta'].values,
    df_slope['bin_mid'].values,
    color=COLOR_SLOPE, linewidth=2.5,
    marker='o', markersize=6,
    label='Slope', zorder=3
)

ax_slope.set_ylabel('Slope (°)', fontsize=14,
                    color='black')
ax_slope.tick_params(axis='y', labelcolor=COLOR_SLOPE, labelsize=14)
ax_slope.set_yticks(df_slope['bin_mid'].values)
ax_slope.set_yticklabels([f'{m:.1f}°' for m in df_slope['bin_mid'].values],
                         fontsize=14)

# ── x-axis ────────────────────────────────────────────────────────────────────
ax_elev.set_xlabel('Mean Δsuitability',
                   fontsize=14, fontproperties=fp_reg)
ax_elev.tick_params(axis='x', labelsize=14)

# ── Legend ────────────────────────────────────────────────────────────────────
ax_elev.legend(
    handles=[l1, l2],
    labels=['Elevation', 'Slope'],
    fontsize=14, frameon=False, loc='upper right'
)

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {OUT_PATH}')