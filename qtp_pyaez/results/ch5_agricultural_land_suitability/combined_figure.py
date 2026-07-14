"""
Combined 4-Panel Figure — Chapter 5
=====================================
Layout:
  Row 0: ax_a (mean suitability)  |  ax_b (suitable land area)
  Row 1: crop legend spanning both columns            [legend strip]
  Row 2: ax_c (class distribution) |  ax_d (expansion vs intensification)
  Row 3: class legend (left)       |  expand/intens legend (right) [legend strip]
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
WORK_DIR    = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
CSV_DIR     = r'./results/ch5_agricultural_land_suitability/outputs/csv'
CSV_DIR_OLD = r'./results/agricultural_land_suitability/outputs/csv'
OUT_PATH    = r'./results/ch5_agricultural_land_suitability/outputs/chapter5_combined_4panel.png'

ALPHA = 0.05
DPI   = 300

os.chdir(WORK_DIR)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# ── Font setup ────────────────────────────────────────────────────────────────
FONT      = 'Helvetica'
BOLD_PATH = '/Users/ming-mayhu/Library/Fonts/Helvetica LT 75 Bold.ttf'
REG_PATH  = '/System/Library/Fonts/Helvetica.ttc'

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
    return dict(tau=mk.Tau, p=mk.p, pstr=pstr, slope=mk.slope,
                intercept=mk.intercept, significant=mk.p < ALPHA,
                sen_line=line)

def bootstrap_sen_ci(series, n_boot=1000, ci=95):
    s         = np.array(series, dtype=float)
    valid_idx = np.where(np.isfinite(s))[0]
    if len(valid_idx) < 4:
        return (np.nan, np.nan)
    s_valid = s[valid_idx]
    rng     = np.random.default_rng(42)
    slopes  = [mk_test(s_valid[np.sort(rng.choice(len(s_valid), len(s_valid), replace=True))]).slope
               for _ in range(n_boot)]
    return (np.percentile(slopes, (100-ci)/2), np.percentile(slopes, 100-(100-ci)/2))

def draw_trend(ax, years_arr, overall, mk, ci, color=TREND_COLOR):
    if not mk:
        return
    ax.plot(years_arr, mk['sen_line'], color=color, linewidth=1.8,
            linestyle='--', dashes=(6, 3), zorder=6)
    if not np.isnan(ci[0]):
        valid = np.isfinite(overall)
        x_idx = np.arange(valid.sum())
        lo = np.full(len(overall), np.nan)
        hi = np.full(len(overall), np.nan)
        lo[valid] = mk['intercept'] + ci[0] * x_idx
        hi[valid] = mk['intercept'] + ci[1] * x_idx
        ax.fill_between(years_arr, lo, hi, color=color, alpha=0.12, zorder=5)

# ── Load data ─────────────────────────────────────────────────────────────────
df_mean  = pd.read_csv(f'{CSV_DIR}/per_crop_mean_suitability.csv')
df_area  = pd.read_csv(f'{CSV_DIR}/per_crop_area_suitable_km2.csv')

def try_csv(name):
    try:
        return pd.read_csv(f'{CSV_DIR}/{name}')
    except FileNotFoundError:
        return pd.read_csv(f'{CSV_DIR_OLD}/{name}')

df_class = try_csv('overall_class_area_km2.csv')
df_exp   = try_csv('expansion_vs_intensification.csv')

years_arr    = df_mean['Year'].values.astype(int)
crop_cols    = [c for c in df_mean.columns if c not in ('Year', 'Overall')]
mean_overall = df_mean['Overall'].values
area_overall = df_area['Overall'].values

mk_mean  = run_mk(mean_overall)
mk_area  = run_mk(area_overall)
ci_mean  = bootstrap_sen_ci(mean_overall)
ci_area  = bootstrap_sen_ci(area_overall)

xtick_years = years_arr[::5]

# ── Colours ───────────────────────────────────────────────────────────────────
crop_colors = [
    '#1f77b4', '#ff9e0d', '#7cd67c', '#e84444', '#42d6e7',
    '#ff68af', '#ffcf3e', '#7f7f7f', '#8c564b', '#be87f1'
]

CLASS_COLORS = {2: '#c2e699', 3: '#78c679', 4: '#31a354', 5: '#006837'}
TREND_COLORS_CLASS = {2: '#7ab84e', 3: '#3d8a3d', 4: '#0f6e2a', 5: '#003d1f'}
CLASS_LABELS = {
    2: 'Class 2 (marginally suitable)',
    3: 'Class 3 (moderately suitable)',
    4: 'Class 4 (suitable)',
    5: 'Class 5 (very suitable)',
}

EXPAND_LIGHT  = "#F2C4C0"   # light pink — expansion, individual crops
INTENS_LIGHT  = "#B0C8D8" # light purple — intensification, individual crops
EXPAND_DARK   = "#C97B75"   # dark pink — expansion, overall
INTENS_DARK   = "#5E8FA8" 

# ── Font properties ───────────────────────────────────────────────────────────
fp_normal  = FontProperties(fname=REG_PATH, size=16)
fp_legend  = FontProperties(fname=REG_PATH, size=16)
fp_textbox = FontProperties(fname=REG_PATH, size=16)
fp_panel   = FontProperties(fname=REG_PATH, size=16)

AXIS_FS = 16
TICK_FS = 16

# ── Figure & GridSpec ─────────────────────────────────────────────────────────
# 4 rows: [top panels] [crop legend] [bottom panels] [bottom legends]
# Heights: panels tall, legend rows short
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('white')

gs = fig.add_gridspec(
    4, 2,
    height_ratios=[3.2, 1.8, 3.2, 1.8],   # bottom panels shorter than top
    hspace=0.10,
    wspace=0.2,
    top=0.97, bottom=0.02, left=0.07, right=0.97
)

ax_a    = fig.add_subplot(gs[0, 0])   # mean suitability
ax_b    = fig.add_subplot(gs[0, 1])   # suitable land area
ax_leg1 = fig.add_subplot(gs[1, :])   # crop legend (spans both columns)
ax_c    = fig.add_subplot(gs[2, 0])   # class distribution
ax_d    = fig.add_subplot(gs[2, 1])   # expansion vs intensification
ax_leg2 = fig.add_subplot(gs[3, 0])   # class legend
ax_leg3 = fig.add_subplot(gs[3, 1])   # expand/intens legend

for ax in [ax_leg1, ax_leg2, ax_leg3]:
    ax.set_axis_off()

# ═══════════════════════════════════════════════════════════════════════════════
# Panel (a) — Mean suitability
# ═══════════════════════════════════════════════════════════════════════════════
for i, crop in enumerate(crop_cols):
    ax_a.plot(years_arr, df_mean[crop].values,
              color=crop_colors[i], linewidth=1.4, alpha=0.5)
ax_a.plot(years_arr, mean_overall, color=OVERALL_COLOR, linewidth=3.5, zorder=5)
draw_trend(ax_a, years_arr, mean_overall, mk_mean, ci_mean)

if mk_mean:
    ax_a.text(0.02, 0.97,
              f"Sen's slope: {mk_mean['slope']:+.5f} yr⁻¹ ({mk_mean['pstr']})",
              transform=ax_a.transAxes, va='top', ha='left',
              fontproperties=fp_textbox,)
            #   bbox=dict(facecolor='white', edgecolor='#cccccc', linewidth=0.8, alpha=0.9))

ax_a.set_xlim(1978.5, 2018.5)
ax_a.set_xticks(xtick_years)
ax_a.set_xticklabels([str(y) for y in xtick_years], rotation=45, ha='right', fontsize=TICK_FS)
ax_a.set_ylabel('Mean suitability score (1–5)', fontsize=AXIS_FS, labelpad=5)
ax_a.set_xlabel('Year', fontsize=AXIS_FS, labelpad=3)
ax_a.tick_params(which='major', labelsize=TICK_FS, length=4)
ax_a.text(-0.10, 1.04, '(a)', transform=ax_a.transAxes,
          va='bottom', ha='left', fontproperties=fp_panel)

# ═══════════════════════════════════════════════════════════════════════════════
# Panel (b) — Suitable land area
# ═══════════════════════════════════════════════════════════════════════════════
for i, crop in enumerate(crop_cols):
    ax_b.plot(years_arr, df_area[crop].values,
              color=crop_colors[i], linewidth=1.4, alpha=0.5)
ax_b.plot(years_arr, area_overall, color=OVERALL_COLOR, linewidth=3.5, zorder=5)
draw_trend(ax_b, years_arr, area_overall, mk_area, ci_area)

if mk_area:
    ax_b.text(0.02, 0.97,
              f"Sen's slope: {mk_area['slope']:+.1f} km² yr⁻¹ ({mk_area['pstr']})",
              transform=ax_b.transAxes, va='top', ha='left',
              fontproperties=fp_textbox)
            #   bbox=dict(facecolor='white', edgecolor='#cccccc', linewidth=0.8, alpha=0.9))

ax_b.set_xlim(1978.5, 2018.5)
ax_b.set_xticks(xtick_years)
ax_b.set_xticklabels([str(y) for y in xtick_years], rotation=45, ha='right', fontsize=TICK_FS)
ax_b.set_ylabel('Suitable land area (km²)', fontsize=AXIS_FS, labelpad=5)
ax_b.set_xlabel('Year', fontsize=AXIS_FS, labelpad=3)
ax_b.tick_params(which='major', labelsize=TICK_FS, length=4)
ax_b.text(-0.10, 1.04, '(b)', transform=ax_b.transAxes,
          va='bottom', ha='left', fontproperties=fp_panel)

# ═══════════════════════════════════════════════════════════════════════════════
# Crop legend — row 1, spanning both columns
# ═══════════════════════════════════════════════════════════════════════════════
crop_handles   = [Line2D([0], [0], color=crop_colors[i], linewidth=2.2, alpha=0.6)
                  for i in range(len(crop_cols))]
overall_handle = Line2D([0], [0], color=OVERALL_COLOR, linewidth=3.5)
sen_handle     = Line2D([0], [0], color=TREND_COLOR, linewidth=2,
                        linestyle='--', dashes=(6, 3))

handles_crop = crop_handles + [overall_handle, sen_handle]
labels_crop  = list(crop_cols) + ['Overall mean', "Sen's slope (95% CI)"]

leg1 = ax_leg1.legend(
    handles=handles_crop, labels=labels_crop,
    loc='center', ncol=6,
    frameon=False,
    handlelength=2.5, handletextpad=0.5, columnspacing=1.2,
    borderpad=0
)
for text in leg1.get_texts():
    text.set_fontproperties(fp_legend)
for h in leg1.legend_handles:
    try: h.set_linewidth(2.5)
    except: pass

# ═══════════════════════════════════════════════════════════════════════════════
# Panel (c) — Class distribution stacked area
# ═══════════════════════════════════════════════════════════════════════════════
bottom = np.zeros(len(years_arr))
for c in [2, 3, 4, 5]:
    vals = np.where(np.isfinite(df_class[f'area_class_{c}'].values),
                    df_class[f'area_class_{c}'].values, 0.0)
    ax_c.fill_between(years_arr, bottom, bottom + vals,
                      color=CLASS_COLORS[c], alpha=0.85)
    bottom += vals

cumulative_top = np.zeros(len(years_arr))
for c in [2, 3, 4, 5]:
    vals = np.where(np.isfinite(df_class[f'area_class_{c}'].values),
                    df_class[f'area_class_{c}'].values, 0.0)
    cumulative_top += vals
    mk_c = run_mk(cumulative_top)
    if mk_c:
        ax_c.plot(years_arr, mk_c['sen_line'],
                  color=TREND_COLORS_CLASS[c], linewidth=1.8,
                  linestyle='--', dashes=(5, 3), zorder=5)

ax_c.set_xlim(1978.5, 2018.5)
ax_c.set_xticks(xtick_years)
ax_c.set_ylim(0, 9000)
ax_c.margins(y=0.02)  # reduce from default 0.05
ax_c.set_xticklabels([str(y) for y in xtick_years], rotation=45, ha='right', fontsize=TICK_FS)
ax_c.set_ylabel('Suitable land area (km²)', fontsize=AXIS_FS, labelpad=5)
ax_c.set_xlabel('Year', fontsize=AXIS_FS, labelpad=3)
ax_c.tick_params(which='major', labelsize=TICK_FS, length=4)
ax_c.text(-0.10, 1.04, '(e)', transform=ax_c.transAxes,
          va='bottom', ha='left', fontproperties=fp_panel)

# ═══════════════════════════════════════════════════════════════════════════════
# Panel (d) — Expansion vs Intensification
# ═══════════════════════════════════════════════════════════════════════════════
overall_row = pd.DataFrame([{
    'crop': 'Overall',
    'pct_from_expansion':       df_exp['pct_from_expansion'].mean(),
    'pct_from_intensification': df_exp['pct_from_intensification'].mean(),
}])
df_plot      = pd.concat([df_exp[['crop','pct_from_expansion','pct_from_intensification']],
                          overall_row], ignore_index=True)
crop_labels_d = df_plot['crop'].tolist()
x_d           = np.arange(len(crop_labels_d))
is_overall_d  = np.array([c == 'Overall' for c in crop_labels_d])

expand_col = [EXPAND_DARK if o else EXPAND_LIGHT for o in is_overall_d]
intens_col = [INTENS_DARK if o else INTENS_LIGHT for o in is_overall_d]

for xi, val, col in zip(x_d, df_plot['pct_from_expansion'], expand_col):
    ax_d.bar(xi, val, color=col, alpha=0.9, edgecolor='white', linewidth=0.5)
for xi, val, bot, col in zip(x_d, df_plot['pct_from_intensification'],
                              df_plot['pct_from_expansion'], intens_col):
    ax_d.bar(xi, val, bottom=bot, color=col, alpha=0.9, edgecolor='white', linewidth=0.5)

ax_d.axhline(50, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
ax_d.set_xticks(x_d)
ax_d.set_xticklabels(crop_labels_d, rotation=35, ha='right', fontsize=TICK_FS)
ax_d.set_ylabel('Percent of total suitability gain (%)', fontsize=AXIS_FS, labelpad=5)
ax_d.set_ylim(0, 100)
ax_d.tick_params(which='major', labelsize=TICK_FS, length=4)
ax_d.text(-0.10, 1.04, '(f)', transform=ax_d.transAxes,
          va='bottom', ha='left', fontproperties=fp_panel)

# ═══════════════════════════════════════════════════════════════════════════════
# Class legend — row 3 left (panel c)
# ═══════════════════════════════════════════════════════════════════════════════
fill_handles_c = [mpatches.Patch(color=CLASS_COLORS[c], alpha=0.85, label=CLASS_LABELS[c])
                  for c in [5, 4, 3, 2]]

trend_handles_c = []
for c in [5, 4, 3, 2]:
    mk_c_solo = run_mk(df_class[f'area_class_{c}'].values)
    lbl = (f"Class {c} Sen's slope: {mk_c_solo['slope']:+.1f} km² yr⁻¹ ({mk_c_solo['pstr']})"
           if mk_c_solo else f'Class {c} trend: n/a')
    trend_handles_c.append(
        Line2D([0], [0], color=TREND_COLORS_CLASS[c], linewidth=2,
               linestyle='--', dashes=(5, 3), label=lbl))

leg2 = ax_leg2.legend(
    handles=fill_handles_c + trend_handles_c,
    loc='center', ncol=2,
    frameon=False,
    handlelength=2.2, handletextpad=0.5, columnspacing=1.0,
    borderpad=0
)
for text in leg2.get_texts():
    text.set_fontproperties(fp_legend)

# ═══════════════════════════════════════════════════════════════════════════════
# Expand/intens legend — row 3 right (panel d)
# ═══════════════════════════════════════════════════════════════════════════════
handles_d = [
    mpatches.Patch(color=EXPAND_DARK, alpha=0.9, label='Expansion'),
    mpatches.Patch(color=INTENS_DARK, alpha=0.9, label='Intensification'),
]
leg3 = ax_leg3.legend(
    handles=handles_d,
    loc='center', ncol=2,
    frameon=False,
    handlelength=2.2, handletextpad=0.5, columnspacing=1.2,
    borderpad=0.
)
for text in leg3.get_texts():
    text.set_fontproperties(fp_legend)

fig.savefig(OUT_PATH, dpi=DPI, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f'Saved → {OUT_PATH}')