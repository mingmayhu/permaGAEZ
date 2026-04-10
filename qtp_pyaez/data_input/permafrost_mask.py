"""
Permafrost Mask Creation & Diagnostic
======================================
Creates two masks from permafrost_qilian.tif:
  1. permafrost_only_mask.npy  — study area AND permafrost (value == 1)
  2. seasonal_only_mask.npy    — study area AND seasonally frozen (value == 2)

Also saves permafrost_qilian_binary.tif for use in QGIS overlays.

Reports pixel counts and produces a diagnostic figure.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from osgeo import gdal

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR      = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH     = r'./data_input/qilian mask.tif'
PERM_MAP_PATH = r'./data_input/permafrost_qilian.tif'
OUT_ROOT      = r'./results_analysis/outputs/permafrost_mask_check'

os.chdir(WORK_DIR)
os.makedirs(OUT_ROOT, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    geo_info = (ds.GetGeoTransform(), ds.GetProjection(),
                ds.RasterXSize, ds.RasterYSize)
    return arr, geo_info

def save_raster(path, arr, geo_info):
    geo, proj, nx, ny = geo_info
    driver = gdal.GetDriverByName('GTiff')
    ds_out = driver.Create(path, nx, ny, 1, gdal.GDT_Float32)
    ds_out.SetGeoTransform(geo)
    ds_out.SetProjection(proj)
    band = ds_out.GetRasterBand(1)
    band.WriteArray(arr.astype(np.float32))
    band.SetNoDataValue(-9999.0)
    ds_out.FlushCache()


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    mask_arr, geo_info = load_raster(MASK_PATH)
    perm_arr, _        = load_raster(PERM_MAP_PATH)

    mask     = mask_arr.astype(bool)
    perm_arr = np.where(np.isfinite(perm_arr), perm_arr, 0)

    # ── Create masks ──────────────────────────────────────────────────────────
    perm_mask     = mask & (perm_arr == 1)   # permafrost only
    seasonal_mask = mask & (perm_arr == 2)   # seasonally frozen only

    n_total    = int(mask.sum())
    n_perm     = int(perm_mask.sum())
    n_seasonal = int(seasonal_mask.sum())
    n_other    = n_total - n_perm - n_seasonal

    print('── Permafrost Coverage ──')
    print(f'  Total mask pixels:          {n_total}')
    print(f'  Permafrost (value=1):       {n_perm} '
          f'({n_perm/n_total*100:.1f}%)')
    print(f'  Seasonally frozen (value=2):{n_seasonal} '
          f'({n_seasonal/n_total*100:.1f}%)')
    print(f'  Other/zero:                 {n_other} '
          f'({n_other/n_total*100:.1f}%)')

    # ── Save masks ────────────────────────────────────────────────────────────
    np.save(f'./data_input/permafrost_only_mask.npy',
            perm_mask.astype(bool))
    np.save(f'./data_input/seasonal_only_mask.npy',
            seasonal_mask.astype(bool))

    # Save as GeoTIFF so scripts can use it as a drop-in mask replacement
    # 1 = inside permafrost mask, 0 = outside
    perm_tif = np.where(perm_mask, 1.0, 0.0)
    save_raster('./data_input/permafrost_only_mask.tif',
                perm_tif, geo_info)

    seasonal_tif = np.where(seasonal_mask, 1.0, 0.0)
    save_raster('./data_input/seasonal_only_mask.tif',
                seasonal_tif, geo_info)

    print('\n  ✓ Masks saved:')
    print('     ./data_input/permafrost_only_mask.npy')
    print('     ./data_input/seasonal_only_mask.npy')
    print('     ./data_input/permafrost_only_mask.tif')
    print('     ./data_input/seasonal_only_mask.tif')

    # ── Diagnostic figure ─────────────────────────────────────────────────────
    # Build display array
    display = np.full(mask_arr.shape, np.nan)
    display[mask & (perm_arr == 1)] = 1   # permafrost
    display[mask & (perm_arr == 2)] = 2   # seasonally frozen
    display[mask & (perm_arr == 0)] = 0   # other inside mask

    cmap   = plt.cm.colors.ListedColormap(['#d9d9d9', '#2166AC', '#f4a582'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm   = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(display, cmap=cmap, norm=norm)
    ax.axis('off')
    ax.set_title(
        f'Permafrost Coverage — Qilian Mountain Region\n'
        f'Permafrost: {n_perm} pixels ({n_perm/n_total*100:.1f}%) | '
        f'Seasonally Frozen: {n_seasonal} pixels ({n_seasonal/n_total*100:.1f}%)',
        fontsize=12, fontweight='bold'
    )
    patches = [
        mpatches.Patch(color='#2166AC', label=f'Permafrost (n={n_perm})'),
        mpatches.Patch(color='#f4a582', label=f'Seasonally Frozen (n={n_seasonal})'),
        mpatches.Patch(color='#d9d9d9', label=f'Other (n={n_other})'),
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=10)
    plt.tight_layout()
    fig.savefig(f'{OUT_ROOT}/permafrost_coverage.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n  ✓ Diagnostic figure saved to: {OUT_ROOT}/')


if __name__ == '__main__':
    run()