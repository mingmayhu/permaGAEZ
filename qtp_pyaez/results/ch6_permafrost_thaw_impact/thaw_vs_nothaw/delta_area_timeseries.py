"""
Per-crop delta suitable land area (thaw minus no-thaw) time series
===================================================================
- x axis starts at 1999
- overall mean shown as bar chart (green/red by sign)
- per-crop lines overlaid (solid = sig, dashed = non-sig)
- same crop colour scheme as Ch5 figure
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
from scipy import stats

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
CSV_PATH = r'./results/ch6_permafrost_thaw_impact/thaw_vs_nothaw/figure_exports/per_crop_suitability_timeseries.csv'
OUT_PATH = r'./results/ch6_permafrost_thaw_impact/thaw_vs_nothaw/figure_exports/delta_area_timeseries.png'

ALPHA    = 0.05
DPI      = 300
START_YR = 1999

os.chdir(WORK_DIR)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# ── Fonts ─────────────────────────────────────────────────────────────────────
FONT     = 'Helvetica'
REG_PATH = '/System/Library/Fonts/Helvetica.ttc'

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

fp_legend = FontProperties(fname=REG_PATH, size=11)
AXIS_FS   = 13
TICK_FS   = 12

# ── Crop colours ──────────────────────────────────────────────────────────────
CROP_ORDER = [
    'Winter Barley', 'Spring Barley', 'Winter Wheat', 'Spring Wheat',
    'Silage Maize', 'White Potato', 'Oat', 'Dry Pea', 'Winter Rape', 'Spring Rape'
]
CROP_COLORS = {
    'Winter Barley': '#1f77b4',
    'Spring Barley': '#ff9e0d',
    'Winter Wheat':  '#7cd67c',
    'Spring Wheat':  '#e84444',
    'Silage Maize':  '#42d6e7',
    'White Potato':  '#ff68af',
    'Oat':           '#ffcf3e',
    'Dry Pea':       '#7f7f7f',
    'Winter Rape':   '#8c564b',
    'Spring Rape':   '#be87f1',
}
CROP_DISPLAY = {
    'Winter Barley': 'Winter barley',
    'Spring Barley': 'Spring barley',
    'Winter Wheat':  'Winter wheat',
    'Spring Wheat':  'Spring wheat',
    'Silage Maize':  'Silage maize',
    'White Potato':  'White potato',
    'Oat':           'Spring oat',
    'Dry Pea':       'Dry pea',
    'Winter Rape':   'Winter rapeseed',
    'Spring Rape':   'Spring rapeseed',
}

# ── Wilcoxon helper ───────────────────────────────────────────────────────────
def wilcoxon_p(series):
    s = np.array(series, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < 4:
        return np.nan
    try:
        _, p = stats.wilcoxon(s)
        return p
    except Exception:
        return np.nan

# ── Load & compute delta ──────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

years_all  = np.array(sorted(df['year'].unique()))
post_mask  = years_all >= START_YR
years_plot = years_all[post_mask]

delta = {}
for crop in CROP_ORDER:
    sub = df[df['crop'] == crop].set_index('year')
    d   = np.array([sub.loc[y, 'obs_area_km2'] - sub.loc[y, 'cf_area_km2']
                    if y in sub.index else np.nan
                    for y in years_all])
    delta[crop] = d

overall_delta = np.nanmean(np.stack(list(delta.values())), axis=0)
overall_plot  = overall_delta[post_mask]

wilcoxon_ps = {crop: wilcoxon_p(delta[crop][post_mask]) for crop in CROP_ORDER}

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor('white')

# Overall mean as bars — green positive, red negative
ax.bar(years_plot, overall_plot, color='black', alpha=0.25,
       width=0.8, zorder=2)

# y=0 reference
ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.4, zorder=3)

# Per-crop lines
for crop in CROP_ORDER:
    d   = delta[crop][post_mask]
    p   = wilcoxon_ps[crop]
    sig = (p < ALPHA) if not np.isnan(p) else False
    ax.plot(years_plot, d,
            color=CROP_COLORS[crop],
            linewidth=1.6 if sig else 1.2,
            linestyle='-' if sig else '--',
            alpha=0.6, zorder=4)

# ── Axis formatting ───────────────────────────────────────────────────────────
xtick_years = years_plot[::3]
ax.set_xlim(START_YR - 0.5, 2018.5)
ax.set_xticks(xtick_years)
ax.set_xticklabels([str(y) for y in xtick_years],
                   rotation=45, ha='right', fontsize=TICK_FS)
ax.set_ylabel('Δ Suitable land area (km²)\n(thaw − no-thaw)',
              fontsize=AXIS_FS, labelpad=5)
ax.set_xlabel('Year', fontsize=AXIS_FS, labelpad=3)
ax.tick_params(which='major', labelsize=TICK_FS, length=4)
ax.margins(y=0.05)

# ── Legend ────────────────────────────────────────────────────────────────────
crop_handles = []
for crop in CROP_ORDER:
    p   = wilcoxon_ps[crop]
    sig = (p < ALPHA) if not np.isnan(p) else False
    crop_handles.append(
        Line2D([0], [0], color=CROP_COLORS[crop],
               linewidth=1.8 if sig else 1.4,
               linestyle='-' if sig else '--',
               alpha=0.8, label=CROP_DISPLAY[crop])
    )

bar_handle     = mpatches.Patch(color='black', alpha=0.25,
                                label='Overall mean')
sig_handle     = Line2D([0], [0], color='grey', linewidth=1.8,
                        linestyle='-', label='Wilcoxon p < 0.05')
nonsig_handle  = Line2D([0], [0], color='grey', linewidth=1.4,
                        linestyle='--', label='Non-significant')

leg = ax.legend(
    handles=crop_handles + [bar_handle, sig_handle, nonsig_handle],
    loc='upper center',
    bbox_to_anchor=(0.5, -0.28),
    ncol=4, frameon=False,
    handlelength=2.5, handletextpad=0.5, columnspacing=1.0
)
for text in leg.get_texts():
    text.set_fontproperties(fp_legend)

plt.tight_layout()
plt.subplots_adjust(bottom=0.32)
fig.savefig(OUT_PATH, dpi=DPI, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f'Saved → {OUT_PATH}')