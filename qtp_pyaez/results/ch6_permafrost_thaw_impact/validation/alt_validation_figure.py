"""
figure_alt_validation.py
------------------------
Two-panel figure for Ch. 6.3 ALT validation:
  Left  : scatter plot of SHAW mean peak ALT vs. NIEER mean ALT (786 pixels)
  Right : spatial delta map (SHAW − NIEER) on the Qilian grid

Reuses the same data loading logic as compare_nieer.py.
Output: figures/alt_validation.pdf  (and .png at 300 dpi)
"""

import os
import numpy as np
import matplotlib
import matplotlib.lines
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.font_manager import FontProperties
from scipy import stats
from osgeo import gdal

# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------
FONT_DIR  = "/Users/ming-mayhu/Library/Fonts"
FONT_REG  = os.path.join(FONT_DIR, "Helvetica.ttf")
FONT_BOLD = os.path.join(FONT_DIR, "Helvetica LT 75 Bold.ttf")

if os.path.exists(FONT_REG):
    from matplotlib import font_manager
    font_manager.fontManager.addfont(FONT_REG)
    font_manager.fontManager.addfont(FONT_BOLD)
    matplotlib.rcParams["font.family"] = "Helvetica"
else:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"

fp_bold = FontProperties(fname=FONT_BOLD) if os.path.exists(FONT_BOLD) else None

matplotlib.rcParams.update({
    "font.size"        : 8,
    "axes.labelsize"   : 8,
    "axes.titlesize"   : 8,
    "xtick.labelsize"  : 7,
    "ytick.labelsize"  : 7,
    "legend.fontsize"  : 7,
    "figure.dpi"       : 150,
    "axes.linewidth"   : 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = "/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez"

MASK_PATH       = os.path.join(BASE, "data_input/qilian_mask_new.tif")
PERM_MASK_PATH  = os.path.join(BASE, "data_input/permafrost_qilian.tif")
SHAW_ALT_DIR    = os.path.join(BASE, "data_input/permafrost_yearly")

NIEER_ALT_PATH  = "/Users/ming-mayhu/Desktop/NIEER_permafrost_dataset_released/NIEER_permafrost_dataset_released/NIEER_ALT.tif"

YEARS     = list(range(2000, 2017))
FIG_DIR   = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers (identical to compare_nieer.py)
# ---------------------------------------------------------------------------

def read_tif(path):
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Cannot open: {path}")
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    if nodata is not None:
        arr[arr == nodata] = np.nan
    gt  = ds.GetGeoTransform()
    prj = ds.GetProjection()
    ds  = None
    return arr, gt, prj


def reproject_to_match(src_path, ref_gt, ref_prj, ref_rows, ref_cols,
                       resample_alg=gdal.GRA_Bilinear):
    src_ds  = gdal.Open(src_path, gdal.GA_ReadOnly)
    if src_ds is None:
        raise FileNotFoundError(f"Cannot open: {src_path}")
    src_band   = src_ds.GetRasterBand(1)
    src_arr    = src_band.ReadAsArray().astype(np.float32)
    src_nodata = src_band.GetNoDataValue()
    if src_nodata is not None:
        src_arr[src_arr == src_nodata] = np.nan
    mem_drv  = gdal.GetDriverByName("MEM")
    clean_ds = mem_drv.Create("", src_ds.RasterXSize, src_ds.RasterYSize,
                              1, gdal.GDT_Float32)
    clean_ds.SetGeoTransform(src_ds.GetGeoTransform())
    clean_ds.SetProjection(src_ds.GetProjection())
    clean_ds.GetRasterBand(1).WriteArray(src_arr)
    clean_ds.GetRasterBand(1).SetNoDataValue(float('nan'))
    src_ds = None
    dst_ds = mem_drv.Create("", ref_cols, ref_rows, 1, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(ref_gt)
    dst_ds.SetProjection(ref_prj)
    dst_ds.GetRasterBand(1).Fill(float('nan'))
    dst_ds.GetRasterBand(1).SetNoDataValue(float('nan'))
    gdal.ReprojectImage(clean_ds, dst_ds,
                        clean_ds.GetProjection(), ref_prj, resample_alg)
    arr = dst_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    arr[~np.isfinite(arr)] = np.nan
    clean_ds = dst_ds = None
    return arr


def get_ref_grid(mask_path):
    ds  = gdal.Open(mask_path, gdal.GA_ReadOnly)
    gt  = ds.GetGeoTransform()
    prj = ds.GetProjection()
    nr  = ds.RasterYSize
    nc  = ds.RasterXSize
    ds  = None
    return gt, prj, nr, nc


def build_valid_mask(mask_path, perm_mask_path):
    mask_arr, _, _ = read_tif(mask_path)
    perm_arr, _, _ = read_tif(perm_mask_path)
    return (mask_arr > 0) & (~np.isnan(mask_arr)) & (perm_arr != 0) & (~np.isnan(perm_arr))

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading reference grid and mask …")
ref_gt, ref_prj, ref_rows, ref_cols = get_ref_grid(MASK_PATH)
valid = build_valid_mask(MASK_PATH, PERM_MASK_PATH)

print("Loading SHAW ALT 2000–2016 …")
shaw_alt_stack = []
for yr in YEARS:
    path = os.path.join(SHAW_ALT_DIR, str(yr), "active_layer_depth.npy")
    arr  = np.load(path)
    shaw_alt_stack.append(np.nanmax(arr, axis=2))
shaw_alt_mean = np.nanmean(shaw_alt_stack, axis=0)

print("Reprojecting NIEER ALT …")
nieer_alt = reproject_to_match(NIEER_ALT_PATH, ref_gt, ref_prj, ref_rows, ref_cols) / 100.0  # cm → m

# ---------------------------------------------------------------------------
# Prepare scatter data
# ---------------------------------------------------------------------------
shaw_v  = shaw_alt_mean[valid]
nieer_v = nieer_alt[valid]
both    = ~np.isnan(shaw_v) & ~np.isnan(nieer_v)
sa      = shaw_v[both]
na      = nieer_v[both]

r, p    = stats.spearmanr(sa, na)
bias    = np.nanmean(sa - na)

# ---------------------------------------------------------------------------
# Prepare delta map
# ---------------------------------------------------------------------------
delta      = shaw_alt_mean - nieer_alt   # SHAW − NIEER, full grid
delta_masked = np.where(valid, delta, np.nan)

# Symmetric colour range centred on zero
vmax = np.nanpercentile(np.abs(delta_masked[valid & np.isfinite(delta_masked)]), 95)
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

# Compute lon/lat extent from geotransform for imshow extent
x0   = ref_gt[0]
y0   = ref_gt[3]
dx   = ref_gt[1]
dy   = ref_gt[5]
extent = [x0, x0 + dx * ref_cols, y0 + dy * ref_rows, y0]   # [left, right, bottom, top]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
fig.subplots_adjust(left=0.14, right=0.97, bottom=0.13, top=0.97)

ax.scatter(na, sa, s=4, alpha=0.5, color="#3A7DC9", linewidths=0, rasterized=True)

# 1:1 line
lim_min = min(na.min(), sa.min()) - 0.1
lim_max = max(na.max(), sa.max()) + 0.1
ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=0.8, label="1:1")

# OLS regression line
m, b   = np.polyfit(na, sa, 1)
x_line = np.linspace(lim_min, lim_max, 200)
ax.plot(x_line, m * x_line + b, color="#E05C2A", lw=1.0, label=f"OLS (slope={m:.2f})")

ax.set_xlim(lim_min, lim_max)
ax.set_ylim(lim_min, lim_max)
ax.set_aspect("equal")
ax.set_xlabel("NIEER ALT (m)")
ax.set_ylabel("SHAW ALT (m)")

# Stats annotation
stats_txt = (f"$r_s$ = {r:.3f}\n"
             f"Bias = {bias:+.3f} m\n"
             f"$n$ = {len(sa)}")
ax.text(0.04, 0.97, stats_txt, transform=ax.transAxes,
        fontsize=6.5, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.5))

ax.legend(loc="lower right", frameon=True, framealpha=0.8,
          edgecolor="0.7", fontsize=6.5)

ax.text(-0.15, 1.03, "(a)", transform=ax.transAxes,
        fontsize=8)

# ── Save figure ───────────────────────────────────────────────────────────
out_pdf = os.path.join(FIG_DIR, "alt_validation.pdf")
out_png = os.path.join(FIG_DIR, "alt_validation.png")
fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")

# ── Export delta map as GeoTIFF ────────────────────────────────────────────
out_tif = os.path.join(FIG_DIR, "alt_delta_shaw_minus_nieer.tif")

driver  = gdal.GetDriverByName("GTiff")
ds_out  = driver.Create(out_tif, ref_cols, ref_rows, 1, gdal.GDT_Float32,
                        options=["COMPRESS=LZW", "TILED=YES"])
ds_out.SetGeoTransform(ref_gt)
ds_out.SetProjection(ref_prj)
band = ds_out.GetRasterBand(1)
band.WriteArray(delta_masked)
band.SetNoDataValue(np.nan)
band.SetDescription("SHAW mean peak ALT minus NIEER mean ALT (m), 2000-2016")
ds_out.FlushCache()
ds_out = None
print(f"Saved: {out_tif}")