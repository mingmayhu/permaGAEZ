"""
Figure: Overall Mean Suitability Comparison (Obs vs FAO)
=========================================================
Reads pre-computed CSV and produces a 2-panel figure:
  Left  — Overall mean suitability score (1-5), obs vs FAO, 1979-2018
  Right — Overall suitable land area (km²), obs vs FAO, 1979-2018

Both panels show:
  - Annual series for obs (blue) and FAO (red dashed)
  - Sen's slope trend lines with MK stats in legend
  - 95% bootstrap CI band around each Sen's slope line
  - Shaded region where obs > FAO (blue) and FAO > obs (red)

Input:
  ./results/permafrost_thaw_impact/permafrost_vs_fao/outputs/
      overall_mean_suitability_timeseries.csv

Output:
  ./results/permafrost_thaw_impact/permafrost_vs_fao/outputs/
      fig_fao_timeseries.png
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
WORK_DIR = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
CSV_PATH = (r'./results/permafrost_thaw_impact/permafrost_vs_fao/outputs/'
            r'overall_mean_suitability_timeseries.csv')
OUT_DIR  = r'./results/permafrost_thaw_impact/permafrost_vs_fao/outputs'
OUT_PATH = f'{OUT_DIR}/fig_fao_timeseries.png'
DPI      = 300

FONT      = 'Helvetica'
BOLD_PATH = '/Users/ming-mayhu/Library/Fonts/Helvetica LT 75 Bold.ttf'
REG_PATH  = '/System/Library/Fonts/Helvetica.ttc'

COLOR_OBS = "#2E7BCD"
COLOR_FAO = "#D65F5F"

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

# ── Helpers ───────────────────────────────────────────────────────────────────
def run_mk(series):
    s = np.array(series, dtype=float)
    valid = np.isfinite(s)
    if valid.sum() < 4:
        return None
    mk = mk_test(s[valid])
    line = np.full(len(s), np.nan)
    idx  = np.where(valid)[0]
    line[idx] = mk.intercept + mk.slope * np.arange(valid.sum())
    print(mk.Tau, mk.p, mk.slope)
    return {
        'tau': mk.Tau, 'p': mk.p, 'slope': mk.slope,
        'intercept': mk.intercept,
        'sig': mk.p < 0.05, 'line': line,
    }

def bootstrap_sen_ci(series, n_boot=1000, ci=95):
    s         = np.array(series, dtype=float)
    valid_idx = np.where(np.isfinite(s))[0]
    if len(valid_idx) < 4:
        return (np.nan, np.nan)
    s_valid = s[valid_idx]
    slopes  = []
    rng     = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = np.sort(rng.choice(len(s_valid), size=len(s_valid), replace=True))
        slopes.append(mk_test(s_valid[idx]).slope)
    lo = np.percentile(slopes, (100 - ci) / 2)
    hi = np.percentile(slopes, 100 - (100 - ci) / 2)
    return (lo, hi)

def draw_ci_band(ax, years, series, mk, ci, color):
    """Draw 95% CI band around Sen's slope line."""
    if mk is None or np.isnan(ci[0]):
        return
    valid = np.isfinite(series)
    x_idx = np.arange(valid.sum())
    lo_line = np.full(len(series), np.nan)
    hi_line = np.full(len(series), np.nan)
    lo_line[valid] = mk['intercept'] + ci[0] * x_idx
    hi_line[valid] = mk['intercept'] + ci[1] * x_idx
    ax.fill_between(years, lo_line, hi_line,
                    color=color, alpha=0.12, zorder=5)

# ── Build panels ──────────────────────────────────────────────────────────────
panels = [
    {
        'obs':    df['obs_mean_suit'].values,
        'fao':    df['fao_mean_suit'].values,
        'ylabel': 'Mean suitability (1–5)',
        'title':  '(a)',
    },
    {
        'obs':    df['obs_area_km2'].values,
        'fao':    df['fao_area_km2'].values,
        'ylabel': 'Suitable land area (km²)',
        'title':  '(b)',
    },
]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.patch.set_facecolor('white')

for ax, panel in zip(axes, panels):
    obs_s = panel['obs']
    fao_s = panel['fao']
    mk_obs = run_mk(obs_s)
    mk_fao = run_mk(fao_s)
    ci_obs = bootstrap_sen_ci(obs_s)
    ci_fao = bootstrap_sen_ci(fao_s)

    # Shaded difference regions
    ax.fill_between(years, obs_s, fao_s,
                    where=(obs_s >= fao_s),
                    alpha=0.15, color=COLOR_OBS, linewidth=0)
    ax.fill_between(years, obs_s, fao_s,
                    where=(obs_s < fao_s),
                    alpha=0.15, color=COLOR_FAO, linewidth=0)

    # Annual series
    ax.plot(years, fao_s, color=COLOR_FAO, linewidth=1.8,
            marker='s', markersize=3.5, zorder=4, linestyle='--',
            label='PyAEZ')
    ax.plot(years, obs_s, color=COLOR_OBS, linewidth=1.8,
            marker='o', markersize=3.5, zorder=4,
            label='PermaGAEZ')

    # Sen's slope trend lines
    if mk_obs:
        ax.plot(years, mk_obs['line'], color=COLOR_OBS,
                linewidth=1.4, linestyle=':', zorder=6,
                label=(f"PermaGAEZ Sen's slope: {mk_obs['slope']:.5f} yr⁻¹ "
                       f"(p < 0.001)"))
    if mk_fao:
        ax.plot(years, mk_fao['line'], color=COLOR_FAO,
                linewidth=1.4, linestyle=':', zorder=6,
                label=(f"PyAEZ Sen's slope: {mk_fao['slope']:.5f} yr⁻¹ "
                       f"(p < 0.001)"))

    # 95% CI bands around Sen's slope lines
    # draw_ci_band(ax, years, obs_s, mk_obs, ci_obs, COLOR_OBS)
    # draw_ci_band(ax, years, fao_s, mk_fao, ci_fao, COLOR_FAO)

    # Axes
    ax.set_xlabel('Year', fontsize=14, fontproperties=fp_reg)
    ax.set_ylabel(panel['ylabel'], fontsize=14, fontproperties=fp_reg)
    ax.set_title(panel['title'], fontsize=14, pad=14, loc='left')
    ax.set_xticks(years[::4])
    ax.set_xticklabels(years[::4], rotation=45, ha='right', fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=14, loc='upper left',
              bbox_to_anchor=(0.06, -0.26),
              bbox_transform=ax.transAxes, frameon=False)

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {OUT_PATH}')