"""
Figure: Permafrost Space Heatmap — ΔSuitability by ΔALT × ΔASM
===============================================================
Same visual style as fig_thaw_permafrost_space.py but bins pixels by:
  - X-axis: change in ALT (max 1999-2018 minus max 1979-1998)
  - Y-axis: change in ASM (mean 1999-2018 minus mean 1979-1998)

  - Cells coloured by mean ΔSuitability (RdBu, anchored at 0)
  - Purple box outline on cells significantly different from zero
    (Wilcoxon signed-rank test on pixel values within each bin, p < 0.05)
  - Pixel count annotated in each cell
  - make_axes_locatable colorbar, white grid lines, Helvetica font

All valid non-lake pixels included (seasonally frozen + permafrost).

Input:
  ./data_input/permafrost_yearly/{year}/active_layer_depth.npy
  ./data_input/permafrost_yearly/{year}/avail_soil_moisture.npy
  ./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/5_spatial/1_mean_delta/
      overall_mean_delta_suit.tif
  ./data_input/permafrost_qilian.tif
  ./data_input/qilian mask.tif

Output:
  ./results/permafrost_thaw_impact/thaw_vs_nothaw/figures/
      fig_thaw_permafrost_change_space.png
"""

import os
import warnings
import numpy as np
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
WORK_DIR        = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
PERM_DIR        = r'./data_input/permafrost_yearly'
DELTA_PATH      = (r'./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/'
                   r'5_spatial/1_mean_delta/overall_mean_delta_suit.tif')
MASK_PATH       = r'./data_input/qilian mask.tif'
PERMAFROST_PATH = r'./data_input/permafrost_qilian.tif'
OUT_DIR         = r'./results/permafrost_thaw_impact/thaw_vs_nothaw/figures'
OUT_PATH        = f'{OUT_DIR}/fig_thaw_permafrost_change_space.png'
DPI             = 300

FONT      = 'Helvetica'
BOLD_PATH = '/Users/ming-mayhu/Library/Fonts/Helvetica LT 75 Bold.ttf'
REG_PATH  = '/System/Library/Fonts/Helvetica.ttc'

YEARS_PRE  = list(range(1979, 1999))
YEARS_POST = list(range(1999, 2019))
N_BINS     = 10
MIN_PIX    = 5
SIG_ALPHA  = 0.05
VMAX       = 0.025

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
        'font.family':      'sans-serif',
        'font.sans-serif':  [FONT],
        'axes.edgecolor':   '#000000',
        'axes.linewidth':   0.8,
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
    """All valid non-lake pixels (seasonally frozen + permafrost)."""
    arr    = load_raster(MASK_PATH)
    mask   = arr.astype(bool)
    pf_arr = load_raster(PERMAFROST_PATH)
    if pf_arr is not None:
        lake_mask = ((pf_arr == 0) | (pf_arr == 1) | ~np.isfinite(pf_arr)) & mask
        mask[lake_mask] = False
        print(f'  Excluded {lake_mask.sum()} lake/nodata pixels')
    return mask

def load_period_mean(var_file, years, mask, agg='mean'):
    """Load and aggregate a permafrost variable over a set of years."""
    stack = []
    for year in years:
        path = f'{PERM_DIR}/{year}/{var_file}'
        if not os.path.exists(path):
            continue
        arr = np.load(path).astype(float)
        if arr.ndim == 3:
            arr = (np.nanmax(arr, axis=2) if agg == 'max'
                   else np.nanmean(arr, axis=2))
        if arr.shape != mask.shape:
            continue
        arr[~mask] = np.nan
        stack.append(arr)
    if not stack:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(np.stack(stack), axis=0)

# ── Load mask ─────────────────────────────────────────────────────────────────
print('Loading mask …')
mask = load_mask()
print(f'  Valid pixels: {mask.sum()}')

# ── Load ALT — max per year, then mean over period ────────────────────────────
print('Loading ALT (pre and post periods) …')
alt_pre  = load_period_mean('active_layer_depth.npy', YEARS_PRE,  mask, agg='max')
alt_post = load_period_mean('active_layer_depth.npy', YEARS_POST, mask, agg='max')

if alt_pre is None or alt_post is None:
    raise RuntimeError('Could not load ALT data for one or both periods.')

delta_alt = alt_post - alt_pre

# ── Load ASM — mean per year, then mean over period ───────────────────────────
print('Loading ASM (pre and post periods) …')
asm_pre  = load_period_mean('avail_soil_moisture.npy', YEARS_PRE,  mask, agg='mean')
asm_post = load_period_mean('avail_soil_moisture.npy', YEARS_POST, mask, agg='mean')

if asm_pre is None or asm_post is None:
    raise RuntimeError('Could not load ASM data for one or both periods.')

delta_asm = asm_post - asm_pre

# ── Load delta suitability ────────────────────────────────────────────────────
print('Loading overall mean delta suitability …')
delta_suit = load_raster(DELTA_PATH)
if delta_suit is not None:
    delta_suit[~mask] = np.nan

# ── Flatten to valid pixels ───────────────────────────────────────────────────
valid = (
    mask
    & np.isfinite(delta_alt)
    & np.isfinite(delta_asm)
    & np.isfinite(delta_suit)
)

alt_flat   = delta_alt[valid]
asm_flat   = delta_asm[valid]
delta_flat = delta_suit[valid]

print(f'  Valid pixels for binning: {valid.sum()}')
print(f'  ΔALT range:  [{alt_flat.min():.3f}, {alt_flat.max():.3f}] m')
print(f'  ΔASM range:  [{asm_flat.min():.2f}, {asm_flat.max():.2f}]')
print(f'  ΔSuit range: [{delta_flat.min():.4f}, {delta_flat.max():.4f}]')

# ── Build bins (equal-frequency / quantile) ───────────────────────────────────
alt_edges = np.unique(
    np.nanpercentile(alt_flat, np.linspace(0, 100, N_BINS + 1))
)
asm_edges = np.unique(
    np.nanpercentile(asm_flat, np.linspace(0, 100, N_BINS + 1))
)

n_alt = len(alt_edges) - 1
n_asm = len(asm_edges) - 1

alt_labels = [f'{alt_edges[i]:.2f}–{alt_edges[i+1]:.2f}'
              for i in range(n_alt)]
asm_labels = [f'{asm_edges[i]:.1f}–{asm_edges[i+1]:.1f}'
              for i in range(n_asm)]

# Assign pixels to bins
ai_arr = np.clip(np.digitize(alt_flat, alt_edges) - 1, 0, n_alt - 1)
si_arr = np.clip(np.digitize(asm_flat, asm_edges) - 1, 0, n_asm - 1)

# Build grid
mean_grid  = np.full((n_asm, n_alt), np.nan)
count_grid = np.zeros((n_asm, n_alt), dtype=int)
sig_grid   = np.zeros((n_asm, n_alt), dtype=bool)

pixel_bins = [[[] for _ in range(n_alt)] for _ in range(n_asm)]
for ai, si, dv in zip(ai_arr, si_arr, delta_flat):
    pixel_bins[si][ai].append(dv)

for si in range(n_asm):
    for ai in range(n_alt):
        vals = np.array(pixel_bins[si][ai])
        n    = len(vals)
        count_grid[si, ai] = n
        if n >= MIN_PIX:
            mean_grid[si, ai] = float(np.mean(vals))
            nonzero = vals[vals != 0]
            if len(nonzero) >= 4:
                try:
                    _, p = wilcoxon(vals, alternative='two-sided')
                    sig_grid[si, ai] = p < SIG_ALPHA
                except Exception:
                    pass

# ── Figure ────────────────────────────────────────────────────────────────────
fig_w = n_alt * 1.2 + 2.5
fig_h = n_asm  * 1.0 + 2.0
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.patch.set_facecolor('white')

im = ax.imshow(mean_grid, cmap='RdBu', vmin=-VMAX, vmax=VMAX,
               aspect='auto', origin='lower')

# Colorbar
divider = make_axes_locatable(ax)
cax     = divider.append_axes('right', size='3%', pad=0.1)
cbar    = plt.colorbar(im, cax=cax)
cbar.set_label('Mean ΔSuitability (class units)', fontsize=12,
               fontproperties=fp_reg)
cbar.ax.tick_params(labelsize=12)

# Cell text + significance box
for si in range(n_asm):
    for ai in range(n_alt):
        n   = count_grid[si, ai]
        val = mean_grid[si, ai]
        sig = sig_grid[si, ai]

        if n < MIN_PIX:
            ax.text(ai, si, f'n={n}',
                    ha='center', va='center',
                    fontsize=9, color='#cccccc',
                    fontproperties=fp_reg)
            continue

        text_color = ('white' if (np.isfinite(val) and
                                   abs(val) > VMAX * 0.7)
                      else 'black')
        ax.text(ai, si, f'{val:.3f}\n(n={n})',
                ha='center', va='center',
                fontsize=10, color=text_color,
                fontproperties=fp_reg)

        if sig:
            rect = mpatches.FancyBboxPatch(
                (ai - 0.47, si - 0.47), 0.95, 0.95,
                boxstyle='square,pad=0',
                linewidth=2, edgecolor='#8b1889', facecolor='none',
                zorder=5
            )
            ax.add_patch(rect)

# Axes
ax.set_xticks(range(n_alt))
ax.set_xticklabels(alt_labels, fontsize=11, rotation=15, ha='right')
ax.set_yticks(range(n_asm))
ax.set_yticklabels(asm_labels, fontsize=11)
ax.set_xlabel('ΔALT: change in active layer depth (m)\n(post 1999–2018 minus pre 1979–1998)',
              fontsize=12, fontproperties=fp_reg)
ax.set_ylabel('ΔASM: change in available soil moisture\n(post 1999–2018 minus pre 1979–1998)',
              fontsize=12, fontproperties=fp_reg)
ax.set_title('OVERALL — Mean ΔSuitability\n'
             'in ΔALT × ΔASM Space\n'
             'Blue = thaw helps, Red = thaw hurts | Numbers = pixel count',
             fontsize=11, fontproperties=fp_reg, pad=10)

# White grid lines
for x in np.arange(-0.5, n_alt, 1):
    ax.axvline(x, color='white', linewidth=2)
for y in np.arange(-0.5, n_asm, 1):
    ax.axhline(y, color='white', linewidth=2)

# Significance legend
sig_patch = mpatches.Patch(
    linewidth=2, edgecolor='#8b1889', facecolor='white'
)
ax.legend(
    handles=[sig_patch],
    labels=['p < 0.05 (Wilcoxon)'],
    loc='upper right',
    bbox_to_anchor=(0.74, 1.05),
    bbox_transform=ax.transAxes,
    fontsize=12,
    frameon=False
)

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {OUT_PATH}')