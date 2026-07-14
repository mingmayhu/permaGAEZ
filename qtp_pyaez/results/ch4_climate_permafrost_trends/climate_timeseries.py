"""
2-panel regional climate trends figure for Chapter 4
Panels: (a) Mean Temperature  (b) Total Precipitation
Publication-quality using seaborn theming + Helvetica
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
from pymannkendall import original_test as mk_test
from osgeo import gdal
from matplotlib.ticker import AutoMinorLocator

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR  = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH = r'./data_input/qilian_mask_new.tif'
CLIM_DIR  = r'./data_input/climate_yearly'
OUT_PATH  = r'./results/climate_permafrost_trends/outputs/climate_2panel.png'

YEARS_ALL = list(range(1979, 2019))
ALPHA     = 0.05
DPI       = 300

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
        'font.family':       'sans-serif',
        'font.sans-serif':   [FONT],
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'xtick.direction':   'out',
        'ytick.direction':   'out',
        'xtick.major.size':  4,
        'ytick.major.size':  4,
        'axes.edgecolor':    '#000000',
        'axes.linewidth':    0.8,
    }
)

units_map = {'a': ' °C yr⁻¹ ', 'b': ' mm yr⁻¹ '}
OBS_BLUE  = "#5789df"
TREND_RED = "#f390cb"

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_raster(path):
    ds = gdal.Open(path)
    if ds is None:
        return None
    band   = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    arr    = ds.ReadAsArray().astype(float)
    arr[arr < -1e10] = np.nan
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr

def load_mask():
    return load_raster(MASK_PATH).astype(bool)

def regional_mean(arr, mask):
    valid = mask & np.isfinite(arr)
    return float(np.nanmean(arr[valid])) if valid.any() else np.nan

def load_clim_annual(var_file, years, mask, agg='mean'):
    annual = []
    for year in years:
        path = f'{CLIM_DIR}/{year}/{var_file}'
        if not os.path.exists(path):
            annual.append(np.nan)
            continue
        arr = np.load(path).astype(float)
        if arr.ndim == 3:
            arr = np.nansum(arr, axis=2) if agg == 'sum' else np.nanmean(arr, axis=2)
        arr[~mask] = np.nan
        annual.append(regional_mean(arr, mask))
    return np.array(annual)

def run_mk(series):
    s     = np.array(series, dtype=float)
    valid = np.isfinite(s)
    if valid.sum() < 4:
        return None
    mk   = mk_test(s[valid])
    line = np.full(len(s), np.nan)
    line[valid] = mk.intercept + mk.slope * np.arange(valid.sum())
    return {'tau': mk.Tau, 'p': mk.p, 'slope': mk.slope,
            'intercept': mk.intercept, 'significant': mk.p < ALPHA,
            'sen_line': line}

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
    return (np.percentile(slopes, (100-ci)/2),
            np.percentile(slopes, 100-(100-ci)/2))

# ── Load data ─────────────────────────────────────────────────────────────────
mask      = load_mask()
years_arr = np.array(YEARS_ALL)

print('Loading climate data ...')
tmax_ann  = load_clim_annual('TempMax.npy', YEARS_ALL, mask, 'mean')
tmin_ann  = load_clim_annual('TempMin.npy', YEARS_ALL, mask, 'mean')
tmean_ann = (tmax_ann + tmin_ann) / 2
prec_ann  = load_clim_annual('Precip.npy',  YEARS_ALL, mask, 'sum')

print('Running MK tests ...')
mk_tmean = run_mk(tmean_ann)
mk_prec  = run_mk(prec_ann)

print('Bootstrapping Sen slope CIs ...')
ci_tmean = bootstrap_sen_ci(tmean_ann)
ci_prec  = bootstrap_sen_ci(prec_ann)

panels = [
    (tmean_ann, mk_tmean, ci_tmean, 'Mean regional temperature (°C)',  'a'),
    (prec_ann,  mk_prec,  ci_prec,  'Total annual precipitation (mm)', 'b'),
]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.patch.set_facecolor('white')

fp_normal   = FontProperties(fname=REG_PATH, size=14)
xtick_years = years_arr[::5]

for ax, (series, mk, ci, ylabel, letter) in zip(axes, panels):

    # Fill under series
    ax.fill_between(years_arr, np.nanmin(series), series,
                    color=OBS_BLUE, alpha=0.12, zorder=1)

    # Observed line
    ax.plot(years_arr, series,
            color=OBS_BLUE, linewidth=1.6, zorder=3,
            marker='o', markersize=3.2,
            markerfacecolor=OBS_BLUE, markeredgewidth=0)

    # Sen's slope + 95% CI band
    if mk:
        pstring   = 'p < 0.001' if mk['p'] < 0.001 else f'p = {mk["p"]:.3f}'
        slope_lbl = f"Sen's slope: {mk['slope']:+.3f}{units_map[letter]}({pstring})"
        ax.plot(years_arr, mk['sen_line'],
                color=TREND_RED, linewidth=1.8,
                linestyle='--', dashes=(6, 3),
                zorder=4, label=slope_lbl)

        if not np.isnan(ci[0]):
            valid = np.isfinite(series)
            x_idx = np.arange(valid.sum())
            lo_line = np.full(len(series), np.nan)
            hi_line = np.full(len(series), np.nan)
            lo_line[valid] = mk['intercept'] + ci[0] * x_idx
            hi_line[valid] = mk['intercept'] + ci[1] * x_idx
            ax.fill_between(years_arr, lo_line, hi_line,
                            color=TREND_RED, alpha=0.15, zorder=3,
                            label='95% CI')

    # Axis formatting
    pad = (np.nanmax(series) - np.nanmin(series)) * 0.1
    ax.set_ylim(np.nanmin(series) - pad / 3, np.nanmax(series) + pad)
    ax.set_xlim(1978.5, 2018.5)
    ax.set_xticks(xtick_years)
    ax.set_xticklabels([str(y) for y in xtick_years],
                       rotation=45, ha='right', fontsize=14)
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.set_ylabel(ylabel, fontsize=14, labelpad=6)
    ax.set_xlabel('Year', fontsize=14, labelpad=4)
    ax.tick_params(labelsize=14)
    ax.tick_params(which='minor', color='#999999', width=0.6)
    ax.spines['left'].set_color('#000000')
    ax.spines['bottom'].set_color('#000000')

    # ax.text(-0.05, 1.08, f'({letter})',
    #         transform=ax.transAxes,
    #         va='bottom', ha='left',
    #         fontproperties=fp_normal)

    leg = ax.legend(loc='upper left',
                    bbox_to_anchor=(0, 1.08),
                    bbox_transform=ax.transAxes,
                    edgecolor='none',
                    handlelength=2.5, borderpad=0.5)
    # leg = ax.legend(loc='upper left',
    #             bbox_to_anchor=(0.01, 1.1),
    #             bbox_transform=ax.transAxes,
    #             edgecolor='none',
    #             handlelength=2.5, borderpad=0.5)
    # for text in leg.get_texts():
    #     text.set_fontproperties(fp_normal)

plt.tight_layout()
plt.subplots_adjust(hspace=0.85, wspace=0.2)
fig.savefig(OUT_PATH, dpi=DPI, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f'\nSaved to {OUT_PATH}')