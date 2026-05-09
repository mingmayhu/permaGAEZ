"""
Figure: Permafrost Variable Scatter Plots — ΔSuitability vs ALT & Soil Moisture
=================================================================================
Styled identically to fig_thaw_climate_scatter.py:
  Top left     — Mean ALT (1999-2018) vs mean ΔSuitability
  Top right    — Mean Soil Moisture (1999-2018) vs mean ΔSuitability
  Bottom left  — ALT Change (post minus pre) vs mean ΔSuitability
  Bottom right — Soil Moisture Change (post minus pre) vs mean ΔSuitability

ALT panels: permafrost pixels only (seasonally frozen excluded)
Soil moisture panels: all mask pixels

Each panel:
  - One dot per pixel, coloured by ΔSuitability value (RdBu)
  - OLS trend line with Spearman r in legend
  - Zero reference line
  - Panel label (a)(b)(c)(d), left-aligned title

Input:
  ./data_input/permafrost_yearly/
  ./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/5_spatial/1_mean_delta/
      overall_mean_delta_suit.tif

Output:
  ./results/permafrost_thaw_impact/thaw_vs_nothaw/figures/
      fig_thaw_permafrost_scatter.png
"""

import os
import warnings
import numpy as np
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
PERM_DIR        = r'./data_input/permafrost_yearly'
DELTA_PATH      = (r'./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/'
                   r'5_spatial/1_mean_delta/overall_mean_delta_suit.tif')
MASK_PATH       = r'./data_input/qilian mask.tif'
PERMAFROST_PATH = r'./data_input/permafrost_qilian.tif'
OUT_DIR         = r'./results/permafrost_thaw_impact/thaw_vs_nothaw/figures'
OUT_PATH        = f'{OUT_DIR}/fig_thaw_permafrost_scatter.png'
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
    arr  = load_raster(MASK_PATH)
    mask = arr.astype(bool)
    pf_arr = load_raster(PERMAFROST_PATH)
    if pf_arr is not None:
        lake_mask = ((pf_arr == 0) | ~np.isfinite(pf_arr)) & mask
        mask[lake_mask] = False
        print(f'  Excluded {lake_mask.sum()} lake/nodata pixels')
    return mask, pf_arr

def perm_only_mask(mask, pf_arr):
    """Permafrost pixels only — exclude seasonally frozen (value 2)."""
    if pf_arr is None:
        return mask
    return mask & (np.round(pf_arr).astype(int) == 1)

def load_perm_mean(var_file, years, mask, agg='mean'):
    stack = []
    for year in years:
        path = f'{PERM_DIR}/{year}/{var_file}'
        if not os.path.exists(path):
            continue
        arr = np.load(path).astype(float)
        if arr.ndim == 3:
            arr = np.nanmax(arr, axis=2) if agg == 'max' else np.nanmean(arr, axis=2)
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
mask, pf_arr = load_mask()
pf_mask      = perm_only_mask(mask, pf_arr)
print(f'  All pixels: {mask.sum()},  Permafrost only: {pf_mask.sum()}')

print('Loading permafrost data ...')
alt_post = load_perm_mean('active_layer_depth.npy',  YEARS_POST, mask, agg='max')
alt_pre  = load_perm_mean('active_layer_depth.npy',  YEARS_PRE,  mask, agg='max')
sm_post  = load_perm_mean('avail_soil_moisture.npy', YEARS_POST, mask, agg='mean')
sm_pre   = load_perm_mean('avail_soil_moisture.npy', YEARS_PRE,  mask, agg='mean')

alt_mean   = alt_post
alt_change = alt_post - alt_pre
sm_mean    = sm_post
sm_change  = sm_post  - sm_pre

print('Loading overall mean delta ...')
delta_arr = load_raster(DELTA_PATH)
if delta_arr is not None:
    delta_arr[~mask] = np.nan

# ── Panels ────────────────────────────────────────────────────────────────────
panels = [
    {
        'x_arr'     : alt_mean,
        'pixel_mask': pf_mask,
        'xlabel'    : 'Mean active layer depth (m)',
        'title'     : '(a)',
    },
    {
        'x_arr'     : sm_mean,
        'pixel_mask': mask,
        'xlabel'    : 'Mean available soil moisture',
        'title'     : '(b)',
        'note'      : None,
    },
    {
        'x_arr'     : alt_change,
        'pixel_mask': pf_mask,
        'xlabel'    : 'Active layer depth change (m)',
        'title'     : '(c)',
    },
    {
        'x_arr'     : sm_change,
        'pixel_mask': mask,
        'xlabel'    : 'Soil moisture change',
        'title'     : '(d)',
        'note'      : None,
    },
]

# Shared colour scale anchored at zero — match climate scatter
norm = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes      = axes.flatten()
fig.patch.set_facecolor('white')

for ax, panel in zip(axes, panels):
    x_arr      = panel['x_arr']
    pixel_mask = panel['pixel_mask']

    if x_arr is None:
        ax.set_visible(False)
        continue

    valid = pixel_mask & np.isfinite(x_arr) & np.isfinite(delta_arr)
    xv    = x_arr[valid]
    yv    = delta_arr[valid]

    r, p    = spearmanr(xv, yv)
    sig_str = '*' if p < 0.05 else ''

    # Scatter — identical to climate scatter
    sc = ax.scatter(xv, yv, c=yv, cmap='RdBu', norm=norm,
                    s=14, alpha=0.5, edgecolors='none', zorder=3)
    plt.colorbar(sc, ax=ax, shrink=0.75, label='ΔSuitability')

    # OLS trend line with Spearman r in legend — identical to climate scatter
    xfit, yfit = ols_line(xv, yv)
    if xfit is not None:
        ax.plot(xfit, yfit, color='black', linewidth=1.5,
                linestyle='--', zorder=5,
                label='Spearman r = {:.3f} (p < 0.001)'.format(r)
                      if p < 0.001 else
                      'Spearman r = {:.3f} (p = {:.3f})'.format(r, p))

    # Zero reference
    ax.axhline(0, color='#aaaaaa', linewidth=0.8, linestyle='-', zorder=1)

    ax.set_xlabel(panel['xlabel'], fontsize=12, fontproperties=fp_reg)
    ax.set_ylabel('Mean Δsuitability', fontsize=12, fontproperties=fp_reg)

    # Left-aligned title matching climate scatter exactly
    title_str = f"{panel['title']}\n\n"
    ax.set_title(title_str, fontsize=12, pad=8, loc='left')

    ax.tick_params(labelsize=12)
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(0., 1.1),
        bbox_transform=ax.transAxes,
        fontsize=12,
        frameon=False
    )

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {OUT_PATH}')