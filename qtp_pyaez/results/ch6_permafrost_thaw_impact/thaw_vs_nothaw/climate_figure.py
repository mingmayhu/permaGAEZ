"""
Figure: Climate Correlation Scatter Plots — ΔSuitability vs Climate Variables
==============================================================================
2x2 scatter plot styled to match thesis figure conventions:
  Top left     — Mean Temperature (1999-2018) vs mean ΔSuitability
  Top right    — Mean Precipitation (1999-2018) vs mean ΔSuitability
  Bottom left  — Temperature Change (post minus pre) vs mean ΔSuitability
  Bottom right — Precipitation Change (post minus pre) vs mean ΔSuitability

Each panel:
  - One dot per pixel, coloured by ΔSuitability value (PiYG)
  - OLS trend line
  - Spearman r and p in subtitle
  - Zero reference line

Input:
  ./data_input/climate_yearly/
  ./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/5_spatial/1_mean_delta/
      overall_mean_delta_suit.tif

Output:
  ./results/permafrost_thaw_impact/thaw_vs_nothaw/figures/
      fig_thaw_climate_scatter.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import spearmanr
from osgeo import gdal

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR        = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
CLIM_DIR        = r'./data_input/climate_yearly'
DELTA_PATH      = (r'./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/'
                   r'5_spatial/1_mean_delta/overall_mean_delta_suit.tif')
MASK_PATH       = r'./data_input/qilian mask.tif'
PERMAFROST_PATH = r'./data_input/permafrost_qilian.tif'
OUT_DIR         = r'./results/permafrost_thaw_impact/thaw_vs_nothaw/figures'
OUT_PATH        = f'{OUT_DIR}/fig_thaw_climate_scatter.png'
DPI             = 300

FONT      = 'Helvetica'
BOLD_PATH = '/Users/ming-mayhu/Library/Fonts/Helvetica LT 75 Bold.ttf'
REG_PATH  = '/System/Library/Fonts/Helvetica.ttc'

YEARS_PRE  = list(range(1979, 1999))
YEARS_POST = list(range(1999, 2019))

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
    arr = load_raster(MASK_PATH)
    mask = arr.astype(bool)
    pf_arr = load_raster(PERMAFROST_PATH)
    if pf_arr is not None:
        lake_mask = ((pf_arr == 0) | ~np.isfinite(pf_arr)) & mask
        mask[lake_mask] = False
        print(f'  Excluded {lake_mask.sum()} lake/nodata pixels')
    return mask

def load_clim_mean(var_file, years, mask, agg='mean'):
    stack = []
    for year in years:
        path = f'{CLIM_DIR}/{year}/{var_file}'
        if not os.path.exists(path):
            continue
        arr = np.load(path).astype(float)
        if arr.ndim == 3:
            arr = np.nansum(arr, axis=2) if agg == 'sum' else np.nanmean(arr, axis=2)
        if arr.shape != mask.shape:
            continue
        arr[~mask] = np.nan
        stack.append(arr)
    if not stack:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(np.stack(stack), axis=0)

def ols_line(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return None, None
    z    = np.polyfit(x[valid], y[valid], 1)
    xfit = np.linspace(x[valid].min(), x[valid].max(), 200)
    return xfit, np.polyval(z, xfit)

# ── Load data ─────────────────────────────────────────────────────────────────
print('Loading mask ...')
mask = load_mask()

print('Loading climate data ...')
temp_max_post  = load_clim_mean('TempMax.npy', YEARS_POST, mask, agg='mean')
temp_min_post  = load_clim_mean('TempMin.npy', YEARS_POST, mask, agg='mean')
temp_max_pre   = load_clim_mean('TempMax.npy', YEARS_PRE,  mask, agg='mean')
temp_min_pre   = load_clim_mean('TempMin.npy', YEARS_PRE,  mask, agg='mean')
precip_post    = load_clim_mean('Precip.npy',  YEARS_POST, mask, agg='sum')
precip_pre     = load_clim_mean('Precip.npy',  YEARS_PRE,  mask, agg='sum')

# Derive mean temp from max and min
temp_post = (temp_max_post + temp_min_post) / 2
temp_pre  = (temp_max_pre  + temp_min_pre)  / 2

temp_mean   = temp_post
temp_change = temp_post - temp_pre
precip_mean = precip_post
precip_change = precip_post - precip_pre

print('Loading overall mean delta ...')
delta_arr = load_raster(DELTA_PATH)
if delta_arr is not None:
    delta_arr[~mask] = np.nan

# ── Panels ────────────────────────────────────────────────────────────────────
panels = [
    {
        'x_arr' : temp_mean,
        'xlabel': 'Mean temperature (°C)',
        'title' : '(a)',
    },
    {
        'x_arr' : precip_mean,
        'xlabel': 'Mean annual precipitation (mm)',
        'title' : '(b)',
    },
    {
        'x_arr' : temp_change,
        'xlabel': 'Temperature change (°C)',
        'title' : '(c)',
    },
    {
        'x_arr' : precip_change,
        'xlabel': 'Precipitation change (mm)',
        'title' : '(d)',
    },
]

# Shared colour scale anchored at zero
valid_delta = delta_arr[mask & np.isfinite(delta_arr)]
vlim = float(np.nanpercentile(np.abs(valid_delta), 98))
vlim = max(vlim, 1e-4)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes      = axes.flatten()
fig.patch.set_facecolor('white')

for ax, panel in zip(axes, panels):
    x_arr = panel['x_arr']
    if x_arr is None:
        ax.set_visible(False)
        continue

    valid = mask & np.isfinite(x_arr) & np.isfinite(delta_arr)
    xv    = x_arr[valid]
    yv    = delta_arr[valid]

    r, p    = spearmanr(xv, yv)
    sig_str = '*' if p < 0.05 else ''

    # Scatter coloured by ΔSuitability
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
    sc   = ax.scatter(xv, yv, c=yv, cmap='RdBu', norm=norm,
                      s=14, alpha=0.5, edgecolors='none', zorder=3)
    plt.colorbar(sc, ax=ax, shrink=0.75,
                 label='ΔSuitability')

    # OLS trend line
    xfit, yfit = ols_line(xv, yv)
    if xfit is not None:
        ax.plot(xfit, yfit, color='black', linewidth=1.5,
                linestyle='--', zorder=5, label='Spearman r = {:.3f} (p < 0.001)'.format(r))

    # Zero reference
    ax.axhline(0, color='#aaaaaa', linewidth=0.8, linestyle='-', zorder=1)

    ax.set_xlabel(panel['xlabel'], fontsize=12, fontproperties=fp_reg)
    ax.set_ylabel('Mean Δsuitability',
                  fontsize=12, fontproperties=fp_reg)
    ax.set_title(
        f"{panel['title']}\n",
        fontsize=12, pad=8, loc='left'
    )
    ax.tick_params(labelsize=12)
    ax.legend(
    loc='upper right',
    bbox_to_anchor=(0.63, 1.08),
    bbox_transform=ax.transAxes,
    fontsize=12,
    frameon=False
)


# f"Spearman r = {r:.3f} (p < 0.001)",
plt.tight_layout()
fig.savefig(OUT_PATH, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {OUT_PATH}')