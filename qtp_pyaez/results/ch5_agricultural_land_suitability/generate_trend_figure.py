"""
Overall Agricultural Land Suitability Trends — Chapter 5
==========================================================
Reads pre-computed CSVs and produces a two-panel figure:
  (a) Mean suitability score (1–5) across all crops
  (b) Suitable land area (km²) across all crops

Individual crop lines shown in muted colours; overall mean in black.
Sen's slope shown as a text box annotation per panel.
95% bootstrap CI shown as shaded band around Sen's slope line.
Single shared legend for crops + overall below both panels.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
import seaborn as sns
from pymannkendall import original_test as mk_test
from matplotlib.ticker import AutoMinorLocator

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR  = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
CSV_DIR   = r'./results/ch5_agricultural_land_suitability/outputs/csv'
OUT_PATH  = r'./results/ch5_agricultural_land_suitability/outputs/overall_suitability_trends.png'

ALPHA = 0.05
DPI   = 300

os.chdir(WORK_DIR)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# ── Font setup ────────────────────────────────────────────────────────────────
FONT      = 'Helvetica'
BOLD_PATH = '/Users/ming-mayhu/Library/Fonts/Helvetica LT 75 Bold.ttf'
REG_PATH  = '/System/Library/Fonts/Helvetica.ttc'

# ── Seaborn theme ─────────────────────────────────────────────────────────────
sns.set_theme(
    style='ticks',
    rc={
        'font.family':        'sans-serif',
        'font.sans-serif':    [FONT],
        'xtick.direction':    'out',
        'ytick.direction':    'out',
        'xtick.major.size':   4,
        'ytick.major.size':   4,
        'axes.edgecolor':     '#000000',
        'axes.linewidth':     0.8,
    }
)

OVERALL_COLOR = 'black'
TREND_COLOR   = 'black'

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
    print(np.max(series), np.min(series), np.mean(series))
    return {
        'tau': mk.Tau, 'p': mk.p, 'pstr': pstr,
        'slope': mk.slope, 'intercept': mk.intercept,
        'significant': mk.p < ALPHA,
        'sen_line': line,
    }

def bootstrap_sen_ci(series, n_boot=1000, ci=95):
    """Bootstrap 95% CI on Sen's slope; returns (lo, hi)."""
    s = np.array(series, dtype=float)
    valid_idx = np.where(np.isfinite(s))[0]
    if len(valid_idx) < 4:
        return (np.nan, np.nan)
    s_valid = s[valid_idx]
    slopes = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = np.sort(rng.choice(len(s_valid), size=len(s_valid), replace=True))
        slopes.append(mk_test(s_valid[idx]).slope)
    lo = np.percentile(slopes, (100 - ci) / 2)
    hi = np.percentile(slopes, 100 - (100 - ci) / 2)
    return (lo, hi)

# ── Load data ─────────────────────────────────────────────────────────────────
df_mean = pd.read_csv(f'{CSV_DIR}/per_crop_mean_suitability.csv')
df_area = pd.read_csv(f'{CSV_DIR}/per_crop_area_suitable_km2.csv')

years_arr    = df_mean['Year'].values.astype(int)
crop_cols    = [c for c in df_mean.columns if c not in ('Year', 'Overall')]
mean_overall = df_mean['Overall'].values
area_overall = df_area['Overall'].values

mk_mean = run_mk(mean_overall)
mk_area = run_mk(area_overall)

ci_mean = bootstrap_sen_ci(mean_overall)
ci_area = bootstrap_sen_ci(area_overall)

# ── Crop colour palette ───────────────────────────────────────────────────────
crop_colors = [
    '#1f77b4', '#ff9e0d', '#7cd67c', '#e84444', '#42d6e7',
    '#ff68af', '#ffcf3e', '#7f7f7f', '#8c564b', '#be87f1'
]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('white')

fp_bold      = FontProperties(fname=BOLD_PATH, size=22)
fp_normal    = FontProperties(fname=REG_PATH,  size=20)
fp_legend    = FontProperties(fname=REG_PATH,  size=20)
fp_textbox   = FontProperties(fname=REG_PATH,  size=20)

xtick_years = years_arr[::5]

panels = [
    (axes[0], df_mean, mean_overall, mk_mean, ci_mean,
     'Mean suitability score (1–5)',
     'Mean suitability across all crops', 'a', ''),
    (axes[1], df_area, area_overall, mk_area, ci_area,
     'Suitable land area (km²)',
     'Suitable land area across all crops', 'b', 'km²'),
]

for ax, df, overall, mk, ci, ylabel, title, letter, unit in panels:

    # Individual crop lines
    for i, crop in enumerate(crop_cols):
        ax.plot(years_arr, df[crop].values,
                color=crop_colors[i], linewidth=1.5,
                alpha=0.5)

    # Overall mean
    ax.plot(years_arr, overall,
            color=OVERALL_COLOR, linewidth=4,
            zorder=5)

    # Sen's slope line + CI band
    if mk:
        ax.plot(years_arr, mk['sen_line'],
                color=TREND_COLOR, linewidth=1.8,
                linestyle='--', dashes=(6, 3),
                zorder=6)

        # CI band around Sen's slope
        if not np.isnan(ci[0]):
            valid = np.isfinite(overall)
            x_idx = np.arange(valid.sum())
            lo_line = np.full(len(overall), np.nan)
            hi_line = np.full(len(overall), np.nan)
            lo_line[valid] = mk['intercept'] + ci[0] * x_idx
            hi_line[valid] = mk['intercept'] + ci[1] * x_idx
            ax.fill_between(years_arr, lo_line, hi_line,
                            color=TREND_COLOR, alpha=0.12,
                            zorder=5)

        # Sen's slope text box
        slope_txt = (f"Sen's slope: {mk['slope']:+.5f} {unit} yr⁻¹ ({mk['pstr']})")
        ax.text(0.01, 1.12, slope_txt,
                transform=ax.transAxes,
                va='top', ha='left',
                fontproperties=fp_textbox,
                bbox=dict(
                    facecolor='white',
                    edgecolor='#cccccc',
                    linewidth=0.8,
                    alpha=0.9)
                )

    # Axis formatting
    ax.set_xlim(1978.5, 2018.5)
    ax.set_xticks(xtick_years)
    ax.set_xticklabels([str(y) for y in xtick_years],
                       rotation=45, ha='right', fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20, labelpad=6)
    ax.set_xlabel('Year', fontsize=20, labelpad=4)
    ax.tick_params(which='major', labelsize=20, length=4,
                   color='#000000', width=0.8)
    ax.spines['left'].set_color('#000000')
    ax.spines['bottom'].set_color('#000000')

    ax.text(-0.07, 1.15, f'({letter})',
            transform=ax.transAxes,
            va='bottom', ha='left',
            fontproperties=fp_normal)

# ── Single shared legend below both panels ────────────────────────────────────
crop_handles = [
    Line2D([0], [0], color=crop_colors[i], linewidth=2.5, alpha=0.6)
    for i in range(len(crop_cols))
]
overall_handle = Line2D([0], [0], color=OVERALL_COLOR, linewidth=4)
sen_handle     = Line2D([0], [0], color=TREND_COLOR, linewidth=2.5,
                        linestyle='--', dashes=(6, 3))

handles = crop_handles + [overall_handle, sen_handle]
labels  = list(crop_cols) + ["Overall mean", "Sen's slope (95% CI)"]

leg = fig.legend(handles=handles, labels=labels,
                 loc='lower center',
                 bbox_to_anchor=(0.5, -0.05),
                 framealpha=0, edgecolor='none',
                 handlelength=3.5, borderpad=0.1,
                 ncol=6)
for text in leg.get_texts():
    text.set_fontproperties(fp_legend)
for handle in leg.legend_handles:
    try:
        handle.set_linewidth(3)
    except AttributeError:
        pass

plt.tight_layout()
plt.subplots_adjust(wspace=0.25, bottom=0.22)
fig.savefig(OUT_PATH, dpi=DPI, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f'Saved to {OUT_PATH}')