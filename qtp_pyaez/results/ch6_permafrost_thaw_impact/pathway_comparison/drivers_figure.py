"""
Figure: Drivers Heatmap — Spearman r for pathway analyses
With purple significance boxes matching fig_fao_drivers_heatmap style
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

WORK_DIR = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
os.chdir(WORK_DIR)

IN_THAW = r'./results/ch6_permafrost_thaw_impact/pathway_comparison/thaw_nothaw_pathway.csv'
IN_FAO  = r'./results/ch6_permafrost_thaw_impact/pathway_comparison/permagaez_fao_pathway.csv'
OUT_DIR = r'./results/ch6_permafrost_thaw_impact/pathway_comparison'
os.makedirs(OUT_DIR, exist_ok=True)

FONT      = 'Helvetica'
REG_PATH  = '/System/Library/Fonts/Helvetica.ttc'
DPI       = 300
SIG_ALPHA = 0.05
VMAX      = 0.7
SIG_COLOR = '#8b1889'

THAW_PREDICTORS = {
    'fc2': 'Δfc2',
    'lgp': 'ΔLGP',
}
FAO_PREDICTORS = {
    'fc2': 'Δfc2',
    'lgp': 'ΔLGP',
    'fc4': 'Δfc4',
}

CROP_LABELS = {
    'combined_winter_barley': 'Winter barley',
    'combined_spring_barley': 'Spring barley',
    'combined_winter_wheat':  'Winter wheat',
    'combined_spring_wheat':  'Spring wheat',
    'combined_silage_maize':  'Silage maize',
    'combined_white_potato':  'White potato',
    'combined_oat':           'Spring oat',
    'combined_dry_pea':       'Dry pea',
    'combined_winter_rape':   'Winter rapeseed',
    'combined_spring_rape':   'Spring rapeseed',
    'OVERALL':                'Overall',
}

try:
    fp_reg = FontProperties(fname=REG_PATH)
except Exception:
    fp_reg = FontProperties()

sns.set_theme(
    style='ticks',
    rc={
        'font.family':        'sans-serif',
        'font.sans-serif':    [FONT],
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'axes.spines.bottom': False,
        'axes.spines.left':   False,
        'axes.edgecolor':     '#000000',
        'axes.linewidth':     0.8,
    }
)

def load_and_order(path):
    df    = pd.read_csv(path)
    order = [c for c in CROP_LABELS if c != 'OVERALL'] + ['OVERALL']
    df['_sort'] = df['crop'].map({c: i for i, c in enumerate(order)})
    df = df.sort_values('_sort').drop(columns='_sort').reset_index(drop=True)
    return df

def build_matrices(df, predictors):
    row_labels = [CROP_LABELS.get(c, c) for c in df['crop']]
    col_labels = list(predictors.values())
    suffixes   = list(predictors.keys())

    n_rows = len(df)
    n_cols = len(suffixes)
    r_mat  = np.full((n_rows, n_cols), np.nan)
    p_mat  = np.full((n_rows, n_cols), np.nan)

    for j, suf in enumerate(suffixes):
        r_col = f'r_{suf}'
        p_col = f'p_{suf}'
        if r_col in df.columns:
            r_mat[:, j] = df[r_col].values
        if p_col in df.columns:
            p_mat[:, j] = df[p_col].values

    sig_mat = p_mat < SIG_ALPHA
    return r_mat, p_mat, sig_mat, row_labels, col_labels

def draw_heatmap(ax, r_mat, p_mat, sig_mat, row_labels, col_labels,
                 title, fp_reg, vmax=VMAX, show_yticklabels=True):
    n_rows, n_cols = r_mat.shape

    im = ax.imshow(r_mat, cmap='RdBu', vmin=-vmax, vmax=vmax,
                   aspect='auto', origin='upper')

    for i in range(n_rows):
        for j in range(n_cols):
            r_val = r_mat[i, j]
            if np.isnan(r_val):
                continue
            text_color = 'white' if abs(r_val) > vmax * 0.65 else 'black'

            # Only annotate significant cells
            if sig_mat[i, j]:
                ax.text(j, i, f'{r_val:.2f}',
                        ha='center', va='center',
                        fontsize=12, color=text_color,
                        fontproperties=fp_reg)

            # "ns" label on non-significant cells
            if not sig_mat[i, j]:
                ax.text(j, i, 'not significant',
                        ha='center', va='center',
                        fontsize=9, color='#555555',
                        fontproperties=fp_reg, zorder=6)

    # White grid lines between cells
    for x in np.arange(-0.5, n_cols, 1):
        ax.axvline(x, color='white', linewidth=3, zorder=4)
    for y in np.arange(-0.5, n_rows, 1):
        ax.axhline(y, color='white', linewidth=3, zorder=4)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=12, fontproperties=fp_reg)

    if show_yticklabels:
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels, fontsize=12, fontproperties=fp_reg)
    else:
        ax.set_yticks([])

    ax.set_title(title, fontsize=12, fontproperties=fp_reg, pad=8, loc='left')

    return im

# ── Load data ─────────────────────────────────────────────────────────────────
df_thaw = load_and_order(IN_THAW)
df_fao  = load_and_order(IN_FAO)

r_thaw, p_thaw, sig_thaw, rows_thaw, cols_thaw = build_matrices(df_thaw, THAW_PREDICTORS)
r_fao,  p_fao,  sig_fao,  rows_fao,  cols_fao  = build_matrices(df_fao,  FAO_PREDICTORS)

# ── Figure ────────────────────────────────────────────────────────────────────
n_rows = len(rows_thaw)
fig_w  = 10.0
fig_h  = n_rows * 0.7 + 1.5

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(fig_w, fig_h),
    gridspec_kw={'wspace': 0.15, 'width_ratios': [len(cols_fao), len(cols_thaw)]}
)
fig.patch.set_facecolor('white')

im1 = draw_heatmap(
    ax1, r_fao, p_fao, sig_fao, rows_fao, cols_fao,
    title='(a)', fp_reg=fp_reg, show_yticklabels=True,
)
im2 = draw_heatmap(
    ax2, r_thaw, p_thaw, sig_thaw, rows_thaw, cols_thaw,
    title='(b)', fp_reg=fp_reg, show_yticklabels=False,
)

# Shared colorbar
fig.subplots_adjust(right=0.88)
cax  = fig.add_axes([0.91, 0.12, 0.02, 0.72])
cbar = fig.colorbar(im1, cax=cax)
cbar.set_label('Spearman r', fontsize=12, fontproperties=fp_reg)
cbar.ax.tick_params(labelsize=12)
cbar.set_ticks([-0.5, 0, 0.5])

# No legend needed — caption notes only significant correlations shown

out_png = os.path.join(OUT_DIR, 'fig_drivers_heatmap.png')
out_pdf = os.path.join(OUT_DIR, 'fig_drivers_heatmap.pdf')
fig.savefig(out_png, dpi=DPI, bbox_inches='tight', facecolor='white')
fig.savefig(out_pdf, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {out_png}')