"""
Export data for FAO drivers heatmap figure
==========================================
Computes Spearman r between (obs-FAO suitability difference) and permafrost
variables (ALT, soil moisture) for each crop and overall.

Uses full 40-year mean (1979-2018), all agricultural pixels (lake pixels excluded).

Output:
  ./results/permafrost_thaw_impact/permafrost_vs_fao/outputs/
      fao_drivers_heatmap_data.csv

Columns: crop, variable, spearman_r, p_value, significant, n_pixels
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from osgeo import gdal

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR        = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH       = r'./data_input/qilian mask.tif'
PERMAFROST_PATH = r'./data_input/permafrost_qilian.tif'
PERM_DIR        = r'./data_input/permafrost_yearly'
OUT_DIR         = r'./results/permafrost_thaw_impact/permafrost_vs_fao/outputs'

YEARS_ALL  = list(range(1979, 2019))

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

PERM_VARS = [
    {'name': 'ALT',          'label': 'Active Layer Depth',
     'file': 'active_layer_depth.npy',  'agg': 'max'},
    {'name': 'soil_moisture', 'label': 'Soil Moisture',
     'file': 'avail_soil_moisture.npy', 'agg': 'mean'},
]

os.chdir(WORK_DIR)
os.makedirs(OUT_DIR, exist_ok=True)


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

def load_mask():
    arr, _ = load_raster(MASK_PATH)
    mask = arr.astype(bool)
    pf_arr, _ = load_raster(PERMAFROST_PATH)
    if pf_arr is not None:
        lake_mask = ((pf_arr == 0) | ~np.isfinite(pf_arr)) & mask
        mask[lake_mask] = False
        print(f'  Excluded {lake_mask.sum()} lake/nodata pixels from mask')
    return mask

def obs_path(tag, year):
    return f'./data_output/final_classification_fixed/{tag}/{year}_suitability_class.tif'

def fao_path(tag, year):
    return f'./data_output/original/final_classification_fixed/{tag}/{year}_suitability_class.tif'

def apply_remap(arr, mask):
    arr_c = arr.copy()
    arr_c[arr_c < 0] = np.nan
    arr_c[~mask] = np.nan
    int_arr = np.where(np.isfinite(arr_c), arr_c, 0).astype(int)
    int_arr[int_arr == 0] = 1
    return np.where(np.isfinite(arr_c), int_arr.astype(float), np.nan)


# ── Load permafrost variable means (full 40-year) ─────────────────────────────

def load_perm_means(mask):
    target = mask.shape
    perm_data = {}
    for pv in PERM_VARS:
        stack = []
        for year in YEARS_ALL:
            path = f'{PERM_DIR}/{year}/{pv["file"]}'
            if not os.path.exists(path):
                stack.append(np.full(target, np.nan))
                continue
            arr = np.load(path).astype(float)
            if arr.ndim == 3:
                arr = (np.nanmax(arr, axis=2) if pv['agg'] == 'max'
                       else np.nanmean(arr, axis=2))
            if arr.shape != target:
                stack.append(np.full(target, np.nan))
                continue
            arr[~mask] = np.nan
            stack.append(arr)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mean_all = np.nanmean(np.array(stack), axis=0)
        mean_all[~mask] = np.nan
        perm_data[pv['name']] = mean_all
        print(f'  {pv["name"]}: loaded 40-year mean')
    return perm_data


# ── Compute per-crop diff maps (full 40-year mean) ────────────────────────────

def load_crop_diff(tag, mask):
    obs_stack, fao_stack = [], []
    for year in YEARS_ALL:
        obs, _ = load_raster(obs_path(tag, year))
        fao, _ = load_raster(fao_path(tag, year))
        if obs is None or fao is None:
            continue
        obs_stack.append(apply_remap(obs, mask))
        fao_stack.append(apply_remap(fao, mask))
    if not obs_stack:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        obs_mean = np.nanmean(np.stack(obs_stack), axis=0)
        fao_mean = np.nanmean(np.stack(fao_stack), axis=0)
    diff = np.where(np.isfinite(obs_mean) & np.isfinite(fao_mean),
                    obs_mean - fao_mean, np.nan)
    diff[~mask] = np.nan
    return diff


# ── Main: compute correlations ────────────────────────────────────────────────

def main():
    print('Loading mask ...')
    mask = load_mask()

    print('\nLoading permafrost variable means ...')
    perm_data = load_perm_means(mask)

    rows = []
    crop_diffs = []

    for crop in CROPS:
        print(f'\n  {crop["label"]}')
        diff = load_crop_diff(crop['tag'], mask)
        if diff is None:
            print(f'    WARNING: no data, skipping')
            continue
        crop_diffs.append(diff)

        for pv in PERM_VARS:
            x = perm_data[pv['name']]
            valid = mask & np.isfinite(diff) & np.isfinite(x)
            xv = x[valid]
            yv = diff[valid]
            if len(xv) < 5:
                continue
            r, p = spearmanr(xv, yv)
            rows.append({
                'crop'       : crop['label'],
                'variable'   : pv['label'],
                'spearman_r' : round(float(r), 4),
                'p_value'    : round(float(p), 4),
                'significant': bool(p < 0.05),
                'n_pixels'   : int(valid.sum()),
            })
            print(f'    {pv["label"]}: r={r:.3f}, p={p:.4f}')

    # Overall (mean diff across all crops)
    print('\n  OVERALL')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        overall_diff = np.nanmean(np.stack(crop_diffs), axis=0)
    overall_diff[~mask] = np.nan

    for pv in PERM_VARS:
        x = perm_data[pv['name']]
        valid = mask & np.isfinite(overall_diff) & np.isfinite(x)
        xv = x[valid]
        yv = overall_diff[valid]
        if len(xv) < 5:
            continue
        r, p = spearmanr(xv, yv)
        rows.append({
            'crop'       : 'OVERALL',
            'variable'   : pv['label'],
            'spearman_r' : round(float(r), 4),
            'p_value'    : round(float(p), 4),
            'significant': bool(p < 0.05),
            'n_pixels'   : int(valid.sum()),
        })
        print(f'    {pv["label"]}: r={r:.3f}, p={p:.4f}')

    df = pd.DataFrame(rows)
    out_path = f'{OUT_DIR}/fao_drivers_heatmap_data.csv'
    df.to_csv(out_path, index=False)
    print(f'\nSaved: {out_path}  ({len(df)} rows)')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()