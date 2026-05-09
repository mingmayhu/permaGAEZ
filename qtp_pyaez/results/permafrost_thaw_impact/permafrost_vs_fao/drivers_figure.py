"""
Figure: Permafrost Drivers of Obs vs FAO Difference — Heatmap (horizontal)
============================================================================
Horizontal layout: permafrost variables on y-axis, crops on x-axis.
  - Cells coloured by Spearman r (RdBu diverging, anchored at 0)
  - Black box outline on significant cells (p < 0.05)
  - Vertical line separating per-crop columns from OVERALL
  - Colourbar labelled 'Spearman r'

Input:
  ./results/permafrost_thaw_impact/permafrost_vs_fao/outputs/
      fao_drivers_heatmap_data.csv

Output:
  ./results/permafrost_thaw_impact/permafrost_vs_fao/outputs/
      fig_fao_drivers_heatmap.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
CSV_PATH = (r'./results/permafrost_thaw_impact/permafrost_vs_fao/outputs/'
            r'fao_drivers_heatmap_data.csv')
OUT_DIR  = r'./results/permafrost_thaw_impact/permafrost_vs_fao/outputs'
OUT_PATH = f'{OUT_DIR}/fig_fao_drivers_heatmap.png'
DPI      = 300

FONT      = 'Helvetica'
BOLD_PATH = '/Users/ming-mayhu/Library/Fonts/Helvetica LT 75 Bold.ttf'
REG_PATH  = '/System/Library/Fonts/Helvetica.ttc'

CROP_ORDER = [
    'Winter Barley', 'Spring Barley', 'Winter Wheat', 'Spring Wheat',
    'Silage Maize', 'White Potato', 'Oat', 'Dry Pea',
    'Winter Rape', 'Spring Rape', 'Overall',
]

VAR_ORDER = ['Active Layer Depth', 'Soil Moisture']

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
        'axes.spines.bottom':   False,
        'axes.spines.left': False,
        'axes.edgecolor':    '#000000',
        'axes.linewidth':    0.8,
    }
)

# ── Load and pivot data ───────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df = df[df['crop'].isin(CROP_ORDER) & df['variable'].isin(VAR_ORDER)]

r_mat = df.pivot(index='variable', columns='crop', values='spearman_r').reindex(
    index=VAR_ORDER, columns=CROP_ORDER)
p_mat = df.pivot(index='variable', columns='crop', values='p_value').reindex(
    index=VAR_ORDER, columns=CROP_ORDER)
sig_mat = df.pivot(index='variable', columns='crop', values='significant').reindex(
    index=VAR_ORDER, columns=CROP_ORDER)

r_vals = r_mat.values.astype(float)
p_vals = p_mat.values.astype(float)
s_vals = sig_mat.values.astype(bool)

n_rows, n_cols = r_vals.shape
vmax = 0.35

# ── Figure ────────────────────────────────────────────────────────────────────
fig_w = n_cols * 0.8 + 2
fig_h = n_rows * 0.8 + 2
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.patch.set_facecolor('white')

im = ax.imshow(r_vals, cmap='RdBu', vmin=-vmax, vmax=vmax, aspect='auto')

# Colourbar
# cbar = plt.colorbar(im, ax=ax, shrink=.9, pad=0.02)
# cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02, fraction=0.05)


divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='2%', pad=0.1
                          )
cbar = plt.colorbar(im, cax=cax)
cbar.set_label('Spearman r', fontsize=12, fontproperties=fp_reg)
cbar.ax.tick_params(labelsize=12)

# Cell text and significance box
for i in range(n_rows):
    for j in range(n_cols):
        val = r_vals[i, j]
        if not np.isfinite(val):
            continue
        sig        = s_vals[i, j]
        txt        = f'{val:.2f}'
        text_color = 'white' if abs(val) > 0.35 else 'black'

        ax.text(j, i, txt,
                ha='center', va='center',
                fontsize=12, color=text_color,
                fontproperties=fp_reg)

        # Black rectangle outline on significant cells
        if sig:
            rect = mpatches.FancyBboxPatch(
                (j - 0.47, i - 0.47), .95, .95,
                boxstyle='square,pad=0',
                linewidth=2, edgecolor="#8b1889",facecolor='none',
                zorder=5
            )
            ax.add_patch(rect)

# x-axis: crop labels
ax.set_xticks(range(n_cols))
ax.set_xticklabels(CROP_ORDER, fontsize=12, rotation=30, ha='right')

# y-axis: variable labels
ax.set_yticks(range(n_rows))
ax.set_yticklabels(VAR_ORDER, fontsize=12)

# Vertical line separating per-crop columns from OVERALL
overall_idx = CROP_ORDER.index('Overall')
ax.axvline(overall_idx - 0.5, color='black', linewidth=1.5)

# Thin white grid lines between cells
for x in np.arange(-0.5, n_cols, 1):
    ax.axvline(x, color='white', linewidth=2)
for y in np.arange(-0.5, n_rows, 1):
    ax.axhline(y, color='white', linewidth=2)
sig_patch = mpatches.Patch(
    linewidth=2, edgecolor='#8b1889', facecolor='white'
)
ax.legend(
    handles=[sig_patch],
    labels=['p < 0.05'],
    loc='upper right',
    bbox_to_anchor=(0.59, 1.2),
    bbox_transform=ax.transAxes,
    fontsize=12,
    frameon=False
)

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {OUT_PATH}')