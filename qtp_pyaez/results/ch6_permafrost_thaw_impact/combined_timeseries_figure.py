"""
Figure: Combined Suitable Land Area Time Series
================================================
2-panel figure:
  (a) Suitable land area — PermaGAEZ vs PyAEZ, 1979–2018
  (b) Suitable land area — Thaw vs No-Thaw, 1979–2018

Both panels:
  - Annual series with markers
  - Sen's slope trend lines
  - Shaded difference regions
  - 1999 divergence line (panel b only)

Inputs:
  ./results/ch6_permafrost_thaw_impact/permafrost_vs_fao/outputs/
      overall_mean_suitability_timeseries.csv
  ./results/ch6_permafrost_thaw_impact/thaw_vs_nothaw/figure_exports/
      overall_suitability_timeseries.csv

Output:
  ./results/ch6_permafrost_thaw_impact/figures/fig_combined_timeseries.png
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
WORK_DIR  = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
CSV_FAO   = (r'./results/ch6_permafrost_thaw_impact/permafrost_vs_fao/outputs/'
             r'overall_mean_suitability_timeseries.csv')
CSV_THAW  = (r'./results/ch6_permafrost_thaw_impact/thaw_vs_nothaw/figure_exports/'
             r'overall_suitability_timeseries.csv')
OUT_DIR   = r'./results/ch6_permafrost_thaw_impact/figures'
OUT_PATH  = f'{OUT_DIR}/fig_combined_timeseries.png'
DPI       = 300
DIVERGENCE = 1999

FONT      = 'Helvetica'
BOLD_PATH = '/Users/ming-mayhu/Library/Fonts/Helvetica LT 75 Bold.ttf'
REG_PATH  = '/System/Library/Fonts/Helvetica.ttc'

# Panel (a) colours — PermaGAEZ vs PyAEZ
COLOR_PERMA = '#2E7BCD'
COLOR_PYAEZ = '#D65F5F'

# Panel (b) colours — Thaw vs No-Thaw
COLOR_THAW  = '#2E7BCD'
COLOR_CF    = '#D65F5F'

os.chdir(WORK_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# ── Fonts ─────────────────────────────────────────────────────────────────────
try:
    fp_reg = FontProperties(fname=REG_PATH)
except Exception:
    fp_reg = FontProperties()

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
def run_mk(series):
    s     = np.array(series, dtype=float)
    valid = np.isfinite(s)
    if valid.sum() < 4:
        return None
    mk   = mk_test(s[valid])
    line = np.full(len(s), np.nan)
    line[np.where(valid)[0]] = mk.intercept + mk.slope * np.arange(valid.sum())
    pstr = 'p < 0.001' if mk.p < 0.001 else f'p = {mk.p:.3f}'
    return {'slope': mk.slope, 'p': mk.p, 'pstr': pstr, 'line': line}

def bootstrap_sen_ci(series, n_boot=1000, ci=95):
    s         = np.array(series, dtype=float)
    valid_idx = np.where(np.isfinite(s))[0]
    if len(valid_idx) < 4:
        return (np.nan, np.nan)
    s_valid = s[valid_idx]
    rng     = np.random.default_rng(42)
    slopes  = [mk_test(s_valid[np.sort(
                   rng.choice(len(s_valid), len(s_valid), replace=True))]).slope
               for _ in range(n_boot)]
    return (np.percentile(slopes, (100-ci)/2),
            np.percentile(slopes, 100-(100-ci)/2))

def draw_panel(ax, years, s1, s2, c1, c2,
               label1, label2, slope_label1, slope_label2,
               ylabel, panel_letter, divergence_year=None):
    """Draw one panel with two series, shading, and Sen's slopes."""
    mk1 = run_mk(s1)
    mk2 = run_mk(s2)

    # Shaded difference regions
    if divergence_year is not None:
        post = years >= divergence_year
        ax.fill_between(years, s1, s2,
                        where=post & (s1 >= s2),
                        alpha=0.25, color=c1, interpolate=True)
        ax.fill_between(years, s1, s2,
                        where=post & (s1 < s2),
                        alpha=0.25, color=c2, interpolate=True)
    else:
        ax.fill_between(years, s1, s2,
                        where=(s1 >= s2),
                        alpha=0.15, color=c1, interpolate=True)
        ax.fill_between(years, s1, s2,
                        where=(s1 < s2),
                        alpha=0.15, color=c2, interpolate=True)

    # Annual series
    ax.plot(years, s2, color=c2, linewidth=1.8,
            marker='s', markersize=3.5, linestyle='--', zorder=4,
            label=label2)
    ax.plot(years, s1, color=c1, linewidth=1.8,
            marker='o', markersize=3.5, zorder=5,
            label=label1)

    # Sen's slope lines
    if mk1:
        lbl = f"{slope_label1}: {mk1['slope']:+.1f} km² yr⁻¹ ({mk1['pstr']})"
        ax.plot(years, mk1['line'], color=c1, linewidth=1.4,
                linestyle=':', zorder=6, label=lbl)
    if mk2:
        lbl = f"{slope_label2}: {mk2['slope']:+.1f} km² yr⁻¹ ({mk2['pstr']})"
        ax.plot(years, mk2['line'], color=c2, linewidth=1.4,
                linestyle=':', zorder=6, label=lbl)

    # Divergence line
    if divergence_year is not None:
        ax.axvline(divergence_year, color='#888888', linewidth=0.8,
                   linestyle='--', zorder=2, label=f'Divergence ({divergence_year})')

    # Formatting
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticks(years[::4])
    ax.set_xticklabels(years[::4], rotation=45, ha='right', fontsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.text(-0.08, 1.04, panel_letter, transform=ax.transAxes,
            fontsize=14, va='bottom', ha='left', fontproperties=fp_reg)

    # Build legend: series lines in col 1, slope lines in col 2
    handles, labels = ax.get_legend_handles_labels()
    slope_handles = [h for h, l in zip(handles, labels) if 'slope' in l.lower()]
    slope_labels  = [l for l in labels if 'slope' in l.lower()]
    other_handles = [h for h, l in zip(handles, labels) if 'slope' not in l.lower()]
    other_labels  = [l for l in labels if 'slope' not in l.lower()]

    # Pad shorter list with blanks so columns are equal length
    from matplotlib.lines import Line2D
    blank = Line2D([], [], color='none')
    while len(other_handles) < len(slope_handles):
        other_handles.append(blank); other_labels.append('')
    while len(slope_handles) < len(other_handles):
        slope_handles.append(blank); slope_labels.append('')

    # Stack: all col-1 items then all col-2 items — matplotlib fills column by column
    leg = ax.legend(other_handles + slope_handles,
                    other_labels  + slope_labels,
                    fontsize=12, loc='upper left',
                    bbox_to_anchor=(-0.02, -0.28),
                    bbox_transform=ax.transAxes,
                    frameon=False, ncol=2,
                    handlelength=2.5, columnspacing=1.0)

# ── Load data ─────────────────────────────────────────────────────────────────
df_fao  = pd.read_csv(CSV_FAO)
df_thaw = pd.read_csv(CSV_THAW)

years_fao  = df_fao['year'].values
years_thaw = df_thaw['year'].values

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 6),
                                  gridspec_kw={'wspace': 0.2})
fig.patch.set_facecolor('white')

# Panel (a) — PermaGAEZ vs PyAEZ
draw_panel(
    ax=ax_a,
    years=years_fao,
    s1=df_fao['obs_area_km2'].values,
    s2=df_fao['fao_area_km2'].values,
    c1=COLOR_PERMA, c2=COLOR_PYAEZ,
    label1='PermaGAEZ', label2='PyAEZ',
    slope_label1='PermaGAEZ slope', slope_label2='PyAEZ slope',
    ylabel='Suitable land area (km²)',
    panel_letter='(a)',
    divergence_year=None,
)

# Panel (b) — Thaw vs No-Thaw
draw_panel(
    ax=ax_b,
    years=years_thaw,
    s1=df_thaw['obs_area_km2'].values,
    s2=df_thaw['cf_area_km2'].values,
    c1=COLOR_THAW, c2=COLOR_CF,
    label1='Thaw', label2='No-Thaw',
    slope_label1='Thaw slope', slope_label2='No-Thaw slope',
    ylabel='Suitable land area (km²)',
    panel_letter='(b)',
    divergence_year=DIVERGENCE,
)

# ── Shared y-axis ─────────────────────────────────────────────────────────────
all_vals = np.concatenate([
    df_fao['obs_area_km2'].values, df_fao['fao_area_km2'].values,
    df_thaw['obs_area_km2'].values, df_thaw['cf_area_km2'].values,
])
y_max = np.nanmax(all_vals) * 1.05
ax_a.set_ylim(1000, y_max)
ax_b.set_ylim(1000, y_max)

plt.tight_layout()
plt.subplots_adjust(bottom=0.30)
fig.savefig(OUT_PATH, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {OUT_PATH}')