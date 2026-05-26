"""
Figure: Climate Space Heatmap — ΔSuitability by Temperature × Precipitation
=============================================================================
Styled to match fig_fao_drivers_heatmap.py:
  - X-axis: mean temperature bins (1999-2018)
  - Y-axis: mean precipitation bins (1999-2018)
  - Cells coloured by mean ΔSuitability (PiYG, green = thaw helps)
  - Purple box outline on cells significantly different from zero
    (Wilcoxon signed-rank test on pixel values within each bin, p < 0.05)
  - Pixel count annotated in each cell
  - make_axes_locatable colorbar, white grid lines, Helvetica font

Input:
  ./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/6_climate_corr/
      climate_correlation_results.csv   (for Spearman summary)
  Permafrost/climate data loaded directly for bin computation

  OR reads pre-computed overall_climate_space.png values from:
  ./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/6_climate_corr/
      climate_space/overall_climate_space_data.csv  (if exported)

  Falls back to recomputing from:
    ./data_input/climate_yearly/
    ./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/5_spatial/1_mean_delta/
        overall_mean_delta_suit.tif

Output:
  ./results/permafrost_thaw_impact/thaw_vs_nothaw/figures/
      fig_thaw_climate_space.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import wilcoxon
from osgeo import gdal

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR   = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
CLIM_DIR   = r'./data_input/climate_yearly'
DELTA_PATH = (r'./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/'
              r'5_spatial/1_mean_delta/overall_mean_delta_suit.tif')
MASK_PATH       = r'./data_input/qilian mask.tif'
PERMAFROST_PATH = r'./data_input/permafrost_qilian.tif'
OUT_DIR    = r'./results/permafrost_thaw_impact/thaw_vs_nothaw/figures'
OUT_PATH   = f'{OUT_DIR}/fig_thaw_climate_space.png'
DPI        = 300

FONT      = 'Helvetica'
BOLD_PATH = '/Users/ming-mayhu/Library/Fonts/Helvetica LT 75 Bold.ttf'
REG_PATH  = '/System/Library/Fonts/Helvetica.ttc'

YEARS_ALL  = list(range(1979, 2019))
YEARS_PRE  = list(range(1979, 1999))
YEARS_POST = list(range(1999, 2019))
N_BINS     = 6    # number of bins per axis
MIN_PIX    = 5    # minimum pixels per bin to show
SIG_ALPHA  = 0.05

vmax = 0.10   # colorbar range — adjust if needed

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
        'font.family':        'sans-serif',
        'font.sans-serif':    [FONT],
        # 'axes.spines.top':    False,
        # 'axes.spines.right':  False,
        # 'axes.spines.bottom': False,
        # 'axes.spines.left':   False,
        'axes.edgecolor':     '#000000',
        'axes.linewidth':     0.8,
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
    return mask

def load_clim_mean(var_file, years, mask, agg='mean'):
    """Load climate variable and return mean across years."""
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

# ── Load data ─────────────────────────────────────────────────────────────────
print('Loading mask ...')
mask = load_mask()

print('Loading climate means (1999-2018) ...')
temp_arr   = load_clim_mean('TempMax.npy', YEARS_POST, mask, agg='mean')
temp_min   = load_clim_mean('TempMin.npy', YEARS_POST, mask, agg='mean')
precip_arr = load_clim_mean('Precip.npy',  YEARS_POST, mask, agg='sum')

# Derive mean temp from max and min
if temp_arr is not None and temp_min is not None:
    temp_arr = (temp_arr + temp_min) / 2
    print('  temp_mean derived from (TempMax + TempMin) / 2')

print('Loading overall mean delta ...')
delta_arr = load_raster(DELTA_PATH)
if delta_arr is not None:
    delta_arr[~mask] = np.nan

# ── Flatten to pixel arrays ────────────────────────────────────────────────────
valid = (mask
         & np.isfinite(temp_arr)
         & np.isfinite(precip_arr)
         & np.isfinite(delta_arr))

temp_flat   = temp_arr[valid]
precip_flat = precip_arr[valid]
delta_flat  = delta_arr[valid]

print(f'  Valid pixels: {valid.sum()}')
print(f'  Temp range:   [{temp_flat.min():.1f}, {temp_flat.max():.1f}] °C')
print(f'  Precip range: [{precip_flat.min():.0f}, {precip_flat.max():.0f}] mm')
print(f'  Delta range:  [{delta_flat.min():.4f}, {delta_flat.max():.4f}]')

# ── Build bins ────────────────────────────────────────────────────────────────
temp_edges   = np.unique(np.nanpercentile(temp_flat,   np.linspace(0, 100, N_BINS + 1)))
precip_edges = np.unique(np.nanpercentile(precip_flat, np.linspace(0, 100, N_BINS + 1)))
n_temp   = len(temp_edges) - 1
n_precip = len(precip_edges) - 1

temp_labels   = [f'{temp_edges[i]:.1f}\nto\n{temp_edges[i+1]:.1f}°C'
                 for i in range(n_temp)]
precip_labels = [f'{precip_edges[i]:.0f}–{precip_edges[i+1]:.0f}'
                 for i in range(n_precip)]

# Assign each pixel to a bin
ti_arr = np.clip(np.digitize(temp_flat,   temp_edges)   - 1, 0, n_temp   - 1)
pi_arr = np.clip(np.digitize(precip_flat, precip_edges) - 1, 0, n_precip - 1)

# Build grid of mean delta and pixel count, collect pixel lists for significance
mean_grid = np.full((n_precip, n_temp), np.nan)
count_grid = np.zeros((n_precip, n_temp), dtype=int)
sig_grid   = np.zeros((n_precip, n_temp), dtype=bool)

pixel_bins = [[[] for _ in range(n_temp)] for _ in range(n_precip)]
for ti, pi, dv in zip(ti_arr, pi_arr, delta_flat):
    pixel_bins[pi][ti].append(dv)

for pi in range(n_precip):
    for ti in range(n_temp):
        vals = np.array(pixel_bins[pi][ti])
        n    = len(vals)
        count_grid[pi, ti] = n
        if n >= MIN_PIX:
            mean_grid[pi, ti] = float(np.mean(vals))
            # Wilcoxon signed-rank test against zero
            nonzero = vals[vals != 0]
            if len(nonzero) >= 4:
                try:
                    _, p = wilcoxon(vals, alternative='two-sided')
                    sig_grid[pi, ti] = p < SIG_ALPHA
                except Exception:
                    pass

# ── Figure ────────────────────────────────────────────────────────────────────
fig_w = n_temp   * 1.2 + 2.5
fig_h = n_precip * 1.0 + 2.0
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.patch.set_facecolor('white')

im = ax.imshow(mean_grid, cmap='RdBu', vmin=-vmax, vmax=vmax,
               aspect='auto', origin='lower')

# Colorbar via make_axes_locatable
divider = make_axes_locatable(ax)
cax     = divider.append_axes('right', size='3%', pad=0.1)
cbar    = plt.colorbar(im, cax=cax)
cbar.set_label('Mean Δsuitability (class units)', fontsize=12,
               fontproperties=fp_reg)
cbar.ax.tick_params(labelsize=12)

# Cell text + significance box
for pi in range(n_precip):
    for ti in range(n_temp):
        n   = count_grid[pi, ti]
        val = mean_grid[pi, ti]
        sig = sig_grid[pi, ti]

        if n < MIN_PIX:
            # Show n in grey for empty/sparse bins
            ax.text(ti, pi, f'n={n}',
                    ha='center', va='center',
                    fontsize=9, color='#cccccc',
                    fontproperties=fp_reg)
            continue

        text_color = 'white' if (np.isfinite(val) and abs(val) > vmax * 0.7) else 'black'
        ax.text(ti, pi, f'{val:.3f}\n(n={n})',
                ha='center', va='center',
                fontsize=10, color=text_color,
                fontproperties=fp_reg)

        if sig:
            rect = mpatches.FancyBboxPatch(
                (ti - 0.47, pi - 0.47), 0.95, 0.95,
                boxstyle='square,pad=0',
                linewidth=2, edgecolor='#8b1889', facecolor='none',
                zorder=5
            )
            ax.add_patch(rect)

# Axes
ax.set_xticks(range(n_temp))
ax.set_xticklabels(temp_labels, fontsize=11)
ax.set_yticks(range(n_precip))
ax.set_yticklabels(precip_labels, fontsize=11)
ax.set_xlabel('Mean temperature (°C)', fontsize=12, fontproperties=fp_reg)
ax.set_ylabel('Mean annual precipitation (mm)', fontsize=12,
              fontproperties=fp_reg)

# White grid lines between cells
for x in np.arange(-0.5, n_temp, 1):
    ax.axvline(x, color='white', linewidth=2)
for y in np.arange(-0.5, n_precip, 1):
    ax.axhline(y, color='white', linewidth=2)

# Significance legend
sig_patch = mpatches.Patch(
    linewidth=2, edgecolor='#8b1889', facecolor='white'
)
ax.legend(
    handles=[sig_patch],
    labels=['p < 0.05 (Wilcoxon)'],
    loc='upper right',
    bbox_to_anchor=(0.66, 1.05),
    bbox_transform=ax.transAxes,
    fontsize=12,
    frameon=False
)

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {OUT_PATH}')