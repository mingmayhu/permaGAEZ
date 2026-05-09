"""
Figure: Spatial Thaw Effect Maps
=================================
2-panel spatial figure:
  Top    — Overall mean ΔSuitability (Thaw − No-Thaw), mean 1999-2018
           RdBu diverging colormap, lake pixels masked
  Bottom — Overall pixel-wise thaw effect classification
           5 classes: sig negative, consistently negative, mixed,
                      consistently positive, sig positive

Input TIFs:
  ./results/permafrost_thaw_impact/thaw_vs_nothaw/figure_exports/
      overall_mean_delta_1999_2018.tif
      overall_pixel_classification.tif

Output:
  ./results/permafrost_thaw_impact/thaw_vs_nothaw/figures/
      fig_thaw_spatial.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm, BoundaryNorm, ListedColormap
from matplotlib.font_manager import FontProperties
from osgeo import gdal

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR      = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
DELTA_TIF     = (r'./results/permafrost_thaw_impact/thaw_vs_nothaw/figure_exports/'
                 r'overall_mean_delta_1999_2018.tif')
CLASS_TIF     = (r'./results/permafrost_thaw_impact/thaw_vs_nothaw/figure_exports/'
                 r'overall_pixel_classification.tif')
OUT_DIR       = r'./results/permafrost_thaw_impact/thaw_vs_nothaw/figures'
OUT_PATH      = f'{OUT_DIR}/fig_thaw_spatial.png'
DPI           = 300

FONT      = 'Helvetica'
BOLD_PATH = '/Users/ming-mayhu/Library/Fonts/Helvetica LT 75 Bold.ttf'
REG_PATH  = '/System/Library/Fonts/Helvetica.ttc'

# Classification colours — matches spatial_analysis.py convention
CLASS_COLORS = [
    '#D73027',   # 0: significantly negative
    '#FC8D59',   # 1: consistently negative
    '#FFFFBF',   # 2: mixed / no effect
    '#91BFDB',   # 3: consistently positive
    '#4575B4',   # 4: significantly positive
]
CLASS_LABELS = [
    'Significantly negative',
    'Consistently negative',
    'Mixed / no effect',
    'Consistently positive',
    'Significantly positive',
]

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
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.edgecolor':    '#000000',
        'axes.linewidth':    0.8,
    }
)

# ── Load rasters ──────────────────────────────────────────────────────────────
def load_raster(path):
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(f'Cannot open: {path}')
    band   = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    arr    = ds.ReadAsArray().astype(float)
    arr[arr < -1e10] = np.nan
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr

delta_arr = load_raster(DELTA_TIF)
class_arr = load_raster(CLASS_TIF)

# ── Colour scale for delta map ────────────────────────────────────────────────
valid_delta = delta_arr[np.isfinite(delta_arr)]
vlim = float(np.nanpercentile(np.abs(valid_delta), 98))
vlim = max(vlim, 1e-4)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 10))
fig.patch.set_facecolor('white')

# ── Panel 1: Mean delta suitability ───────────────────────────────────────────
ax1 = axes[0]
norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
im1  = ax1.imshow(delta_arr, cmap='RdBu', norm=norm)
plt.colorbar(im1, ax=ax1, shrink=0.6, pad=0.02,
             label='Mean ΔSuitability (class units)')
ax1.set_title('Overall Mean ΔSuitability (Thaw − No-Thaw)\nMean 1999–2018, all 10 crops',
              fontsize=12, fontproperties=fp_bold, pad=8)
ax1.axis('off')

# ── Panel 2: Pixel-wise classification ────────────────────────────────────────
ax2  = axes[1]
cmap = ListedColormap(CLASS_COLORS)
bnorm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], ncolors=5)
im2  = ax2.imshow(class_arr, cmap=cmap, norm=bnorm)
ax2.set_title('Overall Pixel-wise Thaw Effect Classification\nModal class across all 10 crops',
              fontsize=12, fontproperties=fp_bold, pad=8)
ax2.axis('off')

# Legend for classification
patches = [
    mpatches.Patch(facecolor=COLOR, edgecolor='none', label=LABEL)
    for COLOR, LABEL in zip(CLASS_COLORS, CLASS_LABELS)
]
ax2.legend(
    handles=patches,
    loc='lower right',
    fontsize=9,
    frameon=True,
    framealpha=0.9,
    edgecolor='#cccccc'
)

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {OUT_PATH}')