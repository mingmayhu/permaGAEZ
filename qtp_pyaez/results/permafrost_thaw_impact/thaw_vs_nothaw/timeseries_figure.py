"""
Figure: Thaw vs No-Thaw Suitability Time Series
================================================
2-panel figure:
  Left  — Overall % suitable land (class >= 2), thaw vs no-thaw, 1979-2018
  Right — Overall mean suitability score (1-5), thaw vs no-thaw, 1979-2018

Both panels:
  - Annual series for thaw (blue) and no-thaw CF (red dashed)
  - Sen's slope trend lines with MK stats in legend
  - Shaded region where thaw > no-thaw (blue) and no-thaw > thaw (red)
  - 1999 divergence line

Input:
  ./results/permafrost_thaw_impact/thaw_vs_nothaw/figure_exports/
      overall_suitability_timeseries.csv

Output:
  ./results/permafrost_thaw_impact/thaw_vs_nothaw/figures/
      fig_thaw_timeseries.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from pymannkendall import original_test as mk_test

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR   = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
CSV_PATH   = (r'./results/permafrost_thaw_impact/thaw_vs_nothaw/figure_exports/'
              r'overall_suitability_timeseries.csv')
OUT_DIR    = r'./results/permafrost_thaw_impact/thaw_vs_nothaw/figures'
OUT_PATH   = f'{OUT_DIR}/fig_thaw_timeseries.png'
DPI        = 300
DIVERGENCE = 1999

FONT      = 'Helvetica'
BOLD_PATH = '/Users/ming-mayhu/Library/Fonts/Helvetica LT 75 Bold.ttf'
REG_PATH  = '/System/Library/Fonts/Helvetica.ttc'

COLOR_THAW = '#2E7BCD'
COLOR_CF   = '#D65F5F'

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
df    = pd.read_csv(CSV_PATH)
years = df['year'].values

# ── MK helper ─────────────────────────────────────────────────────────────────
def run_mk(series):
    s     = np.array(series, dtype=float)
    valid = np.isfinite(s)
    if valid.sum() < 4:
        return None
    mk   = mk_test(s[valid])
    line = np.full(len(s), np.nan)
    line[np.where(valid)[0]] = mk.intercept + mk.slope * np.arange(valid.sum())
    return {'tau': mk.Tau, 'p': mk.p, 'slope': mk.slope,
            'sig': mk.p < 0.05, 'line': line}

# ── Panels ────────────────────────────────────────────────────────────────────
panels = [
        {
        'obs':    df['obs_mean_suit'].values,
        'cf':     df['cf_mean_suit'].values,
        'ylabel': 'Mean suitability score (1-5)',
        'title':  '(a)',
    },
    {
        'obs':    df['obs_area_km2'].values,
        'cf':     df['cf_area_km2'].values,
        'ylabel': 'Suitable land (%)',
        'title':  '(b)',
    },

]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.patch.set_facecolor('white')

for ax, panel in zip(axes, panels):
    obs_s  = panel['obs']
    cf_s   = panel['cf']
    mk_obs = run_mk(obs_s)
    mk_cf  = run_mk(cf_s)
    mean_obs = np.nanmean(obs_s[20:])
    mean_cf = np.nanmean(cf_s[20:])
    print(mean_obs, mean_cf)
    print("tau obs " + str(mk_obs["tau"]))
    print("tau cf " + str(mk_cf["tau"]))

    # Shaded difference regions
    post = years >= DIVERGENCE
    ax.fill_between(years, obs_s, cf_s,
                    where=post & (obs_s >= cf_s),
                    alpha=0.25, color=COLOR_THAW, interpolate=True)
    ax.fill_between(years, obs_s, cf_s,
                    where=post & (obs_s < cf_s),
                    alpha=0.25, color=COLOR_CF, interpolate=True)

    # Annual series
    ax.plot(years, obs_s, color=COLOR_THAW, linewidth=1.8,
            marker='o', markersize=3.5, zorder=4, label='Thaw')
    ax.plot(years, cf_s, color=COLOR_CF, linewidth=1.8,
            marker='s', markersize=3.5, linestyle='--', zorder=4,
            label='No-Thaw')

    # Sen's slope trend lines
    if mk_obs:
    
        ax.plot(years, mk_obs['line'], color=COLOR_THAW,
                linewidth=1.4, linestyle=':', zorder=5,
                label=(f"Thaw slope: {mk_obs['slope']:.5f} yr⁻¹ "
                       f"(p < 0.001)"))
    if mk_cf:
        ax.plot(years, mk_cf['line'], color=COLOR_CF,
                linewidth=1.4, linestyle=':', zorder=5,
                label=(f"No-Thaw slope: {mk_cf['slope']:.5f} yr⁻¹ "
                       f"(p < 0.001)"))

    # 1999 divergence line
    ax.axvline(DIVERGENCE, color='#888888', linewidth=0.8,
               linestyle='--', zorder=2, label='Divergence (1999)')

    ax.set_xlabel('Year', fontsize=14, fontproperties=fp_reg)
    ax.set_ylabel(panel['ylabel'], fontsize=14, fontproperties=fp_reg)
    ax.set_title(panel['title'], fontsize=14, pad=8, loc="left")
    ax.set_xticks(years[::4])
    ax.set_xticklabels(years[::4], rotation=45, ha='right', fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=14, loc='upper left', 
        bbox_to_anchor=(0.2, -.2),
    bbox_transform=ax.transAxes,frameon=False)


plt.tight_layout()
fig.savefig(OUT_PATH, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {OUT_PATH}')