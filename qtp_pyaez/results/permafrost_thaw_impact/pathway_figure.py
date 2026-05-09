"""
Figure: LGP and fc2 Pathway Analysis — All 10 Crops
====================================================
Two figures, each with 2 rows × 10 columns:

Figure 1 (fig_pathway_suitability.png):
  Row 1 — ΔLGP vs ΔSuitability       (dots coloured by ΔSuitability, RdBu)
  Row 2 — Δfc2 vs ΔSuitability       (dots coloured by ΔSuitability, RdBu)

Figure 2 (fig_pathway_precipitation.png):
  Row 1 — Mean Precipitation vs ΔLGP  (dots coloured by ΔLGP, coolwarm)
  Row 2 — Mean Precipitation vs Δfc2  (dots coloured by Δfc2, coolwarm)

Each panel:
  - One dot per pixel
  - OLS trend line (black dashed)
  - Spearman r and p in legend
  - Zero reference lines on both axes

Input:
  ./results/permafrost_thaw_impact/pathway_analysis/{crop}_pathway_data.csv

Output:
  ./results/permafrost_thaw_impact/pathway_analysis/figures/
      fig_pathway_suitability.png
      fig_pathway_precipitation.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import spearmanr

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR  = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
DATA_DIR  = r'./results/permafrost_thaw_impact/pathway_analysis'
OUT_DIR   = r'./results/permafrost_thaw_impact/pathway_analysis/figures'
DPI       = 300

FONT      = 'Helvetica'
BOLD_PATH = '/Users/ming-mayhu/Library/Fonts/Helvetica LT 75 Bold.ttf'
REG_PATH  = '/System/Library/Fonts/Helvetica.ttc'

os.chdir(WORK_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# Crop order matches existing thesis figures
CROPS = [
    ('combined_winter_barley', 'Winter Barley'),
    ('combined_spring_barley', 'Spring Barley'),
    ('combined_winter_wheat',  'Winter Wheat'),
    ('combined_spring_wheat',  'Spring Wheat'),
    ('combined_silage_maize',  'Silage Maize'),
    ('combined_white_potato',  'White Potato'),
    ('combined_oat',           'Oat'),
    ('combined_dry_pea',       'Dry Pea'),
    ('combined_winter_rape',   'Winter Rape'),
    ('combined_spring_rape',   'Spring Rape'),
]

# Lightened crop line colours (mixed with white 0.45) — matches thesis style
CROP_COLORS = [
    '#8AADDB',  # Winter Barley
    '#E8A87C',  # Spring Barley
    '#8AC49A',  # Winter Wheat
    '#E88C8C',  # Spring Wheat
    '#B8A0CC',  # Silage Maize
    '#E8C87C',  # White Potato
    '#8AC4C4',  # Oat
    '#C4A08A',  # Dry Pea
    '#A0A0A0',  # Winter Rape
    '#B8CC8A',  # Spring Rape
]

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
        'font.family':      'sans-serif',
        'font.sans-serif':  [FONT],
        'xtick.direction':  'out',
        'ytick.direction':  'out',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'axes.edgecolor':   '#000000',
        'axes.linewidth':   0.8,
    }
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_csv(tag):
    path = f'{DATA_DIR}/{tag}_pathway_data.csv'
    if not os.path.exists(path):
        print(f'  ⚠ Missing CSV: {path}')
        return None
    return pd.read_csv(path)

def ols_line(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return None, None
    z    = np.polyfit(x[valid], y[valid], 1)
    xfit = np.linspace(x[valid].min(), x[valid].max(), 200)
    return xfit, np.polyval(z, xfit)

def sym_vlim(arr, pct=98):
    """Symmetric colour limit based on percentile of absolute values."""
    v = np.nanpercentile(np.abs(arr[np.isfinite(arr)]), pct)
    return max(v, 1e-6)

def add_panel(ax, x, y, c, cmap, norm, xlabel, ylabel, title,
              fp_reg, show_ylabel=True):
    """Draw one scatter panel."""
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    xv, yv, cv = x[valid], y[valid], c[valid]

    r, p = spearmanr(xv, yv)
    p_str = 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'
    sig   = '*' if p < 0.05 else ''

    sc = ax.scatter(xv, yv, c=cv, cmap=cmap, norm=norm,
                    s=10, alpha=0.45, edgecolors='none', zorder=3)

    xfit, yfit = ols_line(xv, yv)
    if xfit is not None:
        ax.plot(xfit, yfit, color='black', linewidth=1.5,
                linestyle='--', zorder=5,
                label=f'r = {r:.3f}{sig} ({p_str})')

    ax.axhline(0, color='#aaaaaa', linewidth=0.7, linestyle='-', zorder=1)
    ax.axvline(0, color='#aaaaaa', linewidth=0.7, linestyle='-', zorder=1)

    ax.set_xlabel(xlabel, fontsize=9, fontproperties=fp_reg)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=9, fontproperties=fp_reg)
    else:
        ax.set_ylabel('')
        ax.set_yticklabels([])

    ax.set_title(title, fontsize=9, pad=4, loc='left',
                 fontproperties=fp_bold)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc='upper right', frameon=False,
              handlelength=1.2)

    return sc

# ── Load all data ─────────────────────────────────────────────────────────────
print('Loading pathway CSVs …')
data = {}
for tag, label in CROPS:
    df = load_csv(tag)
    if df is not None:
        data[tag] = df
        print(f'  {label}: {len(df)} pixels')

# ── Compute shared colour limits across all crops ─────────────────────────────
all_ds   = np.concatenate([data[t]['delta_suit'].values  for t, _ in CROPS if t in data])
all_lgp  = np.concatenate([data[t]['delta_lgp'].values   for t, _ in CROPS if t in data])
all_fc2  = np.concatenate([data[t]['delta_fc2'].values   for t, _ in CROPS if t in data])

vlim_ds  = sym_vlim(all_ds,  pct=98)
vlim_lgp = sym_vlim(all_lgp, pct=98)
vlim_fc2 = sym_vlim(all_fc2, pct=98)

norm_ds  = TwoSlopeNorm(vmin=-vlim_ds,  vcenter=0, vmax=vlim_ds)
norm_lgp = TwoSlopeNorm(vmin=-vlim_lgp, vcenter=0, vmax=vlim_lgp)
norm_fc2 = TwoSlopeNorm(vmin=-vlim_fc2, vcenter=0, vmax=vlim_fc2)

# ── Panel labels ──────────────────────────────────────────────────────────────
ROW_LABELS_FIG1 = ['(a)', '(b)']   # ΔLGP vs ΔSuit, Δfc2 vs ΔSuit
ROW_LABELS_FIG2 = ['(c)', '(d)']   # Precip vs ΔLGP, Precip vs Δfc2

N_CROPS = len(CROPS)

# =============================================================================
# FIGURE 1 — Pathway vs ΔSuitability (coloured by ΔSuitability)
# =============================================================================
print('\nBuilding Figure 1 (pathway vs suitability) …')

fig1, axes1 = plt.subplots(
    2, N_CROPS,
    figsize=(N_CROPS * 2.8, 7),
    sharey='row',
)
fig1.patch.set_facecolor('white')

for col, (tag, label) in enumerate(CROPS):
    if tag not in data:
        for row in range(2):
            axes1[row, col].set_visible(False)
        continue

    df       = data[tag]
    ds       = df['delta_suit'].values
    lgp      = df['delta_lgp'].values
    fc2      = df['delta_fc2'].values
    show_y   = (col == 0)

    # Row 0: ΔLGP vs ΔSuit
    sc1 = add_panel(
        ax       = axes1[0, col],
        x        = lgp,
        y        = ds,
        c        = ds,
        cmap     = 'RdBu',
        norm     = norm_ds,
        xlabel   = 'ΔLGP (days)',
        ylabel   = 'ΔSuitability',
        title    = f'{ROW_LABELS_FIG1[0]} {label}',
        fp_reg   = fp_reg,
        show_ylabel = show_y,
    )

    # Row 1: Δfc2 vs ΔSuit
    sc2 = add_panel(
        ax       = axes1[1, col],
        x        = fc2,
        y        = ds,
        c        = ds,
        cmap     = 'RdBu',
        norm     = norm_ds,
        xlabel   = 'Δfc2',
        ylabel   = 'ΔSuitability',
        title    = f'{ROW_LABELS_FIG1[1]} {label}',
        fp_reg   = fp_reg,
        show_ylabel = show_y,
    )

# Shared colourbar for Figure 1
fig1.subplots_adjust(right=0.88, hspace=0.45, wspace=0.08)
cbar_ax1 = fig1.add_axes([0.90, 0.15, 0.012, 0.7])
sm1 = plt.cm.ScalarMappable(cmap='RdBu', norm=norm_ds)
sm1.set_array([])
cb1 = fig1.colorbar(sm1, cax=cbar_ax1)
cb1.set_label('ΔSuitability', fontsize=10, fontproperties=fp_reg)
cb1.ax.tick_params(labelsize=8)

# Row labels on left margin
for row, row_label in enumerate(['Row (a): ΔLGP vs ΔSuitability',
                                  'Row (b): Δfc2 vs ΔSuitability']):
    fig1.text(0.01, 0.75 - row * 0.5, row_label,
              va='center', ha='left', fontsize=9,
              fontproperties=fp_bold, rotation=90)

out1 = f'{OUT_DIR}/fig_pathway_suitability.png'
fig1.savefig(out1, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close(fig1)
print(f'  Saved: {out1}')

# =============================================================================
# FIGURE 2 — Precipitation vs ΔLGP / Δfc2 (coloured by ΔLGP / Δfc2)
# =============================================================================
print('\nBuilding Figure 2 (precipitation mechanism) …')

fig2, axes2 = plt.subplots(
    2, N_CROPS,
    figsize=(N_CROPS * 2.8, 7),
    sharey='row',
)
fig2.patch.set_facecolor('white')

for col, (tag, label) in enumerate(CROPS):
    if tag not in data:
        for row in range(2):
            axes2[row, col].set_visible(False)
        continue

    df     = data[tag]
    lgp    = df['delta_lgp'].values
    fc2    = df['delta_fc2'].values
    precip = df['mean_precip'].values
    show_y = (col == 0)

    # Row 0: Precip vs ΔLGP, coloured by ΔLGP
    sc3 = add_panel(
        ax       = axes2[0, col],
        x        = precip,
        y        = lgp,
        c        = lgp,
        cmap     = 'coolwarm',
        norm     = norm_lgp,
        xlabel   = 'Mean precip (mm)',
        ylabel   = 'ΔLGP (days)',
        title    = f'{ROW_LABELS_FIG2[0]} {label}',
        fp_reg   = fp_reg,
        show_ylabel = show_y,
    )

    # Row 1: Precip vs Δfc2, coloured by Δfc2
    sc4 = add_panel(
        ax       = axes2[1, col],
        x        = precip,
        y        = fc2,
        c        = fc2,
        cmap     = 'coolwarm',
        norm     = norm_fc2,
        xlabel   = 'Mean precip (mm)',
        ylabel   = 'Δfc2',
        title    = f'{ROW_LABELS_FIG2[1]} {label}',
        fp_reg   = fp_reg,
        show_ylabel = show_y,
    )

# Shared colourbars for Figure 2 (one per row)
fig2.subplots_adjust(right=0.88, hspace=0.45, wspace=0.08)

# Colourbar row 0 — ΔLGP
cbar_ax2a = fig2.add_axes([0.90, 0.54, 0.012, 0.34])
sm2a = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm_lgp)
sm2a.set_array([])
cb2a = fig2.colorbar(sm2a, cax=cbar_ax2a)
cb2a.set_label('ΔLGP (days)', fontsize=10, fontproperties=fp_reg)
cb2a.ax.tick_params(labelsize=8)

# Colourbar row 1 — Δfc2
cbar_ax2b = fig2.add_axes([0.90, 0.12, 0.012, 0.34])
sm2b = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm_fc2)
sm2b.set_array([])
cb2b = fig2.colorbar(sm2b, cax=cbar_ax2b)
cb2b.set_label('Δfc2', fontsize=10, fontproperties=fp_reg)
cb2b.ax.tick_params(labelsize=8)

# Row labels on left margin
for row, row_label in enumerate(['Row (c): Precipitation vs ΔLGP',
                                  'Row (d): Precipitation vs Δfc2']):
    fig2.text(0.01, 0.75 - row * 0.5, row_label,
              va='center', ha='left', fontsize=9,
              fontproperties=fp_bold, rotation=90)

out2 = f'{OUT_DIR}/fig_pathway_precipitation.png'
fig2.savefig(out2, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print(f'  Saved: {out2}')

print('\nDone.')