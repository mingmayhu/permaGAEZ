"""
Southeast Cluster Consistency Check
=====================================
Loads mean ΔSuitability rasters from Step 1 for all crops and checks
how many crops show negative mean ΔSuitability at each pixel.

If the southeast cluster consistently shows negative values across
multiple crops, that confirms it is a real spatial signal rather
than a crop-specific artifact.

Outputs:
  - n_crops_negative.tif   — count of crops with negative ΔSuit per pixel
  - n_crops_positive.tif   — count of crops with positive ΔSuit per pixel
  - cluster_consistency.png — map of crop count negative/positive
  - cluster_stats.csv       — pixel-level summary

Outputs written to: ./results_analysis/outputs/6_spatial_analysis/cluster_check/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from osgeo import gdal

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR  = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH = r'./data_input/qilian mask.tif'
DELTA_DIR = './results_analysis/outputs/6_spatial_analysis/1_mean_delta'
OUT_ROOT  = './results_analysis/outputs/6_spatial_analysis/cluster_check'

CROPS = [
    {'label': 'Winter Barley', 'tag': 'combined_winter_barley'},
    {'label': 'Spring Barley', 'tag': 'combined_spring_barley'},
    {'label': 'Winter Wheat',  'tag': 'combined_winter_wheat'},
    {'label': 'Spring Wheat',  'tag': 'combined_spring_wheat'},
    {'label': 'Silage Maize',  'tag': 'combined_silage_maize'},
    {'label': 'White Potato',  'tag': 'combined_white_potato'},
    {'label': 'Oat',           'tag': 'combined_oat'},
    {'label': 'Dry Pea',       'tag': 'combined_dry_pea'},
    {'label': 'Winter Rape',   'tag': 'combined_winter_rape'},
    {'label': 'Spring Rape',   'tag': 'combined_spring_rape'},
]

os.chdir(WORK_DIR)
os.makedirs(OUT_ROOT, exist_ok=True)

N_CROPS = len(CROPS)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_raster(path):
    ds = gdal.Open(path)
    if ds is None:
        return None, None
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

def load_mask():
    arr, _ = load_raster(MASK_PATH)
    return arr.astype(bool)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(mask):
    rows, cols  = mask.shape
    geo_info_ref = None

    # Stack all mean delta rasters
    delta_stack = []
    loaded_labels = []

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        path = f'{DELTA_DIR}/{tag}_mean_delta_suit.tif'
        arr, geo_info = load_raster(path)
        if arr is None:
            print(f'⚠ Missing: {path}')
            continue
        arr[~mask] = np.nan
        delta_stack.append(arr)
        loaded_labels.append(label)
        if geo_info_ref is None:
            geo_info_ref = geo_info

    n_loaded = len(delta_stack)
    print(f'Loaded {n_loaded} crops.')
    delta_stack = np.array(delta_stack)   # (n_crops, rows, cols)

    # ── Count crops positive/negative at each pixel ───────────────────────────
    # Only count where the pixel has a valid (non-NaN, non-zero) delta
    n_negative = np.zeros((rows, cols), dtype=float)
    n_positive = np.zeros((rows, cols), dtype=float)
    n_valid    = np.zeros((rows, cols), dtype=float)

    for i in range(n_loaded):
        arr = delta_stack[i]
        valid = mask & np.isfinite(arr)
        n_valid   += valid.astype(float)
        n_negative += ((arr < 0) & valid).astype(float)
        n_positive += ((arr > 0) & valid).astype(float)

    # Set outside mask to NaN
    n_negative = np.where(mask & (n_valid > 0), n_negative, np.nan)
    n_positive = np.where(mask & (n_valid > 0), n_positive, np.nan)
    n_valid    = np.where(mask, n_valid, np.nan)

    # Net: positive count minus negative count
    net = np.where(mask & np.isfinite(n_positive) & np.isfinite(n_negative),
                   n_positive - n_negative, np.nan)

    # ── Save rasters ──────────────────────────────────────────────────────────
    for arr, fname in [
        (n_negative, f'{OUT_ROOT}/n_crops_negative.tif'),
        (n_positive, f'{OUT_ROOT}/n_crops_positive.tif'),
        (net,        f'{OUT_ROOT}/net_crops.tif'),
    ]:
        out = np.where(np.isfinite(arr), arr, -9999.0)
        save_raster(fname, out, geo_info_ref)

    # ── Main figure: 3 panels ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # Panel 1: number of crops with negative ΔSuit
    cmap_neg = plt.get_cmap('YlOrRd', n_loaded + 1)
    disp_neg = np.where(mask, n_negative, np.nan)
    im0 = axes[0].imshow(disp_neg, cmap=cmap_neg, vmin=0, vmax=n_loaded)
    axes[0].set_title('No. of Crops with Negative Mean ΔSuitability\n'
                      '(higher = more crops where thaw hurts)',
                      fontsize=11, fontweight='bold')
    axes[0].axis('off')
    cb0 = plt.colorbar(im0, ax=axes[0], shrink=0.75,
                       label='Number of crops')
    cb0.set_ticks(range(n_loaded + 1))

    # Panel 2: number of crops with positive ΔSuit
    cmap_pos = plt.get_cmap('YlGn', n_loaded + 1)
    disp_pos = np.where(mask, n_positive, np.nan)
    im1 = axes[1].imshow(disp_pos, cmap=cmap_pos, vmin=0, vmax=n_loaded)
    axes[1].set_title('No. of Crops with Positive Mean ΔSuitability\n'
                      '(higher = more crops where thaw helps)',
                      fontsize=11, fontweight='bold')
    axes[1].axis('off')
    cb1 = plt.colorbar(im1, ax=axes[1], shrink=0.75,
                       label='Number of crops')
    cb1.set_ticks(range(n_loaded + 1))

    # Panel 3: net (positive - negative crops)
    net_vlim = n_loaded
    disp_net = np.where(mask, net, np.nan)
    im2 = axes[2].imshow(disp_net, cmap='RdBu',
                         vmin=-net_vlim, vmax=net_vlim)
    axes[2].set_title('Net Crops (Positive − Negative)\n'
                      'Blue = thaw helps more crops, Red = thaw hurts more crops',
                      fontsize=11, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], shrink=0.75, label='Net crop count')

    fig.suptitle('Cross-Crop Consistency of Thaw Impact on Suitability\n'
                 'Mean ΔSuitability (Observed − No-Thaw, 1999–2018)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{OUT_ROOT}/cluster_consistency.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Threshold map: pixels negative in ≥ 5, 7, or all crops ──────────────
    thresholds = [5, 7, n_loaded]
    fig, axes  = plt.subplots(1, len(thresholds), figsize=(7 * len(thresholds), 6))

    for ax, thresh in zip(axes, thresholds):
        # Show pixels that are negative in at least `thresh` crops
        neg_cluster = np.where(mask & (n_negative >= thresh), n_negative, np.nan)
        pos_context = np.where(mask & (n_negative < thresh), 0, np.nan)

        # Background: show all mask pixels in light grey
        ax.imshow(np.where(mask, 0, np.nan),
                  cmap='Greys', vmin=-1, vmax=1, alpha=0.3)
        # Overlay: negative cluster pixels
        im = ax.imshow(neg_cluster, cmap='YlOrRd',
                       vmin=thresh, vmax=n_loaded)
        ax.set_title(f'Pixels Negative in ≥{thresh}/{n_loaded} Crops',
                     fontsize=11, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.75, label='No. crops negative')

    fig.suptitle('Spatial Clustering of Negative Thaw Impact\n'
                 'Pixels where thaw reduces suitability across multiple crops',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{OUT_ROOT}/negative_cluster_thresholds.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Stats table ───────────────────────────────────────────────────────────
    records = []
    for thresh in range(1, n_loaded + 1):
        neg_px = int(np.nansum(n_negative >= thresh))
        pos_px = int(np.nansum(n_positive >= thresh))
        records.append({
            'min_crops_agreeing' : thresh,
            'pixels_negative'    : neg_px,
            'pixels_positive'    : pos_px,
            'pct_mask_negative'  : round(neg_px / np.sum(mask) * 100, 3),
            'pct_mask_positive'  : round(pos_px / np.sum(mask) * 100, 3),
        })

    df = pd.DataFrame(records)
    df.to_csv(f'{OUT_ROOT}/cluster_stats.csv', index=False)
    print('\nCluster consistency stats:')
    print(df.to_string(index=False))
    print(f'\n✓ All outputs saved to: {OUT_ROOT}/')


if __name__ == '__main__':
    mask = load_mask()
    run(mask)