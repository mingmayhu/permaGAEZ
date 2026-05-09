"""
Export Mean Suitability TIFs — Chapter 5
=========================================
Computes and exports:
  - Mean suitability 1979-1998 (per crop + overall)
  - Mean suitability 1999-2018 (per crop + overall)
  - Delta suitability (post minus pre, per crop + overall)

Classes 0 and 1 combined into class 1 (not suitable).

Outputs written to:
  ./results/agricultural_land_suitability/outputs/tif/
"""

import os
import numpy as np
from osgeo import gdal, osr

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR  = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH = r'./data_input/qilian_mask_new.tif'
PERM_PATH = r'./data_input/permafrost_qilian.tif'
OUT_DIR   = r'./results/agricultural_land_suitability/outputs/tif'

YEARS_PRE  = list(range(1979, 1999))
YEARS_POST = list(range(1999, 2019))

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
os.makedirs(OUT_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_raster(path):
    ds = gdal.Open(path)
    if ds is None:
        return None, None
    band      = ds.GetRasterBand(1)
    nodata    = band.GetNoDataValue()
    arr       = ds.ReadAsArray().astype(float)
    arr[arr < -1e10] = np.nan
    if nodata is not None:
        arr[arr == nodata] = np.nan
    geo       = ds.GetGeoTransform()
    proj      = ds.GetProjection()
    ds        = None
    return arr, (geo, proj)

def load_mask():
    mask, _    = load_raster(MASK_PATH)
    mask       = mask.astype(bool)
    pf_arr, _  = load_raster(PERM_PATH)
    if pf_arr is not None:
        lake_mask       = ((pf_arr == 0) | ~np.isfinite(pf_arr)) & mask
        mask[lake_mask] = False
        print(f'  Excluded {lake_mask.sum()} lake pixels from mask')
    return mask

def suit_path(tag, year):
    return f'./data_output/final_classification_fixed/{tag}/{year}_suitability_class.tif'

def save_tif(arr, out_path, geo, proj, nodata=-9999.0):
    """Save a float array as a GeoTIFF."""
    rows, cols = arr.shape
    driver     = gdal.GetDriverByName('GTiff')
    ds_out     = driver.Create(out_path, cols, rows, 1, gdal.GDT_Float32,
                               options=['COMPRESS=LZW', 'TILED=YES'])
    ds_out.SetGeoTransform(geo)
    ds_out.SetProjection(proj)
    band = ds_out.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    out = arr.copy()
    out[~np.isfinite(out)] = nodata
    band.WriteArray(out.astype(np.float32))
    band.FlushCache()
    ds_out = None
    print(f'  Saved: {out_path}')

def load_period_mean(tag, years, mask):
    """Average suitability rasters over a list of years."""
    stack = []
    for year in years:
        arr, _ = load_raster(suit_path(tag, year))
        if arr is None:
            continue
        arr[~mask] = np.nan
        arr = np.where(np.isfinite(arr), arr, 0).astype(int)
        arr = np.clip(arr, 0, 5)
        arr[arr == 0] = 1       # combine class 0 into class 1
        arr_flt = arr.astype(float)
        arr_flt[~mask] = np.nan
        stack.append(arr_flt)
    return np.nanmean(stack, axis=0) if stack else None

# ── Main ──────────────────────────────────────────────────────────────────────
def export():
    mask = load_mask()

    # Get georeference info from any one raster
    first_path = suit_path(CROPS[0]['tag'], 1999)
    _, (geo, proj) = load_raster(first_path)

    pre_all  = []
    post_all = []

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        print(f'\nProcessing {label} ...')

        pre  = load_period_mean(tag, YEARS_PRE,  mask)
        post = load_period_mean(tag, YEARS_POST, mask)
        overall = load_period_mean(tag, YEARS_PRE + YEARS_POST, mask)

        if pre is None or post is None:
            print(f'  WARNING: missing data for {label}, skipping')
            continue

        delta = post - pre
        delta[~mask] = np.nan

        # Per-crop TIFs
        # save_tif(pre,   f'{OUT_DIR}/{tag}_mean_pre.tif',   geo, proj)
        # save_tif(post,  f'{OUT_DIR}/{tag}_mean_post.tif',  geo, proj)
        # save_tif(delta, f'{OUT_DIR}/{tag}_delta.tif',      geo, proj)
        save_tif(overall, f'{OUT_DIR}/{tag}_mean_all.tif',     geo, proj)

        # pre_all.append(pre)
        # post_all.append(post)

    # # Overall aggregate TIFs
    # print('\nComputing overall aggregate ...')
    # agg_pre   = np.nanmean(pre_all,  axis=0)
    # agg_post  = np.nanmean(post_all, axis=0)
    # agg_delta = agg_post - agg_pre

    # # Mean suitability across all years (pre + post combined)
    # all_periods = pre_all + post_all  # list of all 20 pre + 20 post mean arrays
    # agg_mean_all = np.nanmean(all_periods, axis=0)

    # # Max suitability across all crops (pixel-wise maximum across crop means)
    # agg_max_all  = np.nanmax(pre_all + post_all, axis=0)

    # # Apply mask
    # for arr in [agg_pre, agg_post, agg_delta,
    #             agg_mean_all, agg_max_all]:
    #     arr[~mask] = np.nan

    # save_tif(agg_pre,       f'{OUT_DIR}/overall_mean_pre_1979_1998.tif',    geo, proj)
    # save_tif(agg_post,      f'{OUT_DIR}/overall_mean_post_1999_2018.tif',   geo, proj)
    # save_tif(agg_delta,     f'{OUT_DIR}/overall_delta_suitability.tif',     geo, proj)
    # save_tif(agg_mean_all,  f'{OUT_DIR}/overall_mean_all_1979_2018.tif',    geo, proj)
    # save_tif(agg_max_all,   f'{OUT_DIR}/overall_max_all_1979_2018.tif',     geo, proj)

    print(f'\nAll TIFs saved to {OUT_DIR}/')
    print(f'  Per-crop: {len(CROPS) * 3} files')
    print(f'  Overall:  5 files')


if __name__ == '__main__':
    export()