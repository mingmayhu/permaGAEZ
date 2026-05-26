"""
Export data for FAO comparison figures
=======================================
Exports:
  1. overall_mean_suitability_timeseries.csv   — annual mean suitability and
     suitable land area (km²) for obs and FAO scenarios, 1979-2018

  2. overall_diff_map_1979_2018.tif            — overall mean suitability
     difference (obs minus FAO), averaged over full 40-year period 1979-2018

  3. per_crop_diff_map_1979_2018.tif           — per-crop 10-band GeoTIFF,
     one band per crop, band description = crop label, full 40-year period

  4. permafrost_drivers_overall_1979_2018.csv  — pixel-level table of mean
     delta suitability vs ALT and soil moisture, full 40-year period,
     permafrost pixels only (seasonally frozen excluded)

All outputs written to:
  ./results/permafrost_thaw_impact/permafrost_vs_fao/outputs/
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from osgeo import gdal

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR        = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH       = r'./data_input/qilian_mask_new.tif'
PERMAFROST_PATH = r'./data_input/permafrost_qilian.tif'
PERM_DIR        = r'./data_input/permafrost_yearly'
OUT_DIR         = r'./results/permafrost_thaw_impact/permafrost_vs_fao/outputs'

YEARS_ALL = list(range(1979, 2019))

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
    band   = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    arr    = ds.ReadAsArray().astype(float)
    arr[arr < -1e10] = np.nan
    if nodata is not None:
        arr[arr == nodata] = np.nan
    geo_info = (ds.GetGeoTransform(), ds.GetProjection(),
                ds.RasterXSize, ds.RasterYSize)
    return arr, geo_info

def save_raster_single(path, arr, geo_info, nodata_val=-9999.0):
    geo, proj, nx, ny = geo_info
    driver = gdal.GetDriverByName('GTiff')
    ds_out = driver.Create(path, nx, ny, 1, gdal.GDT_Float32,
                           options=['COMPRESS=LZW'])
    ds_out.SetGeoTransform(geo)
    ds_out.SetProjection(proj)
    band = ds_out.GetRasterBand(1)
    band.WriteArray(np.where(np.isfinite(arr), arr, nodata_val).astype(np.float32))
    band.SetNoDataValue(nodata_val)
    ds_out.FlushCache()

def save_raster_multiband(path, arrays, band_names, geo_info, nodata_val=-9999.0):
    geo, proj, nx, ny = geo_info
    driver = gdal.GetDriverByName('GTiff')
    ds_out = driver.Create(path, nx, ny, len(arrays), gdal.GDT_Float32,
                           options=['COMPRESS=LZW'])
    ds_out.SetGeoTransform(geo)
    ds_out.SetProjection(proj)
    for i, (arr, name) in enumerate(zip(arrays, band_names), start=1):
        band = ds_out.GetRasterBand(i)
        band.WriteArray(np.where(np.isfinite(arr), arr, nodata_val).astype(np.float32))
        band.SetNoDataValue(nodata_val)
        band.SetDescription(name)
    ds_out.FlushCache()

def load_mask():
    arr, _ = load_raster(MASK_PATH)
    mask = arr.astype(bool)
    pf_arr, _ = load_raster(PERMAFROST_PATH)
    if pf_arr is not None:
        lake_mask = ((pf_arr == 0) | ~np.isfinite(pf_arr)) & mask
        mask[lake_mask] = False
        print(f'  Excluded {lake_mask.sum()} lake/nodata pixels from mask')
    return mask

def build_pixel_area_km2(mask):
    ds  = gdal.Open(MASK_PATH)
    gt  = ds.GetGeoTransform()
    nrows, ncols = mask.shape
    lats = gt[3] + gt[5] * (np.arange(nrows) + 0.5)
    pixel_side_km = abs(gt[5]) * 111.32
    area_2d = np.outer(
        pixel_side_km * pixel_side_km * np.cos(np.deg2rad(lats)),
        np.ones(ncols)
    )
    area_2d[~mask] = 0.0
    return area_2d

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

def regional_mean_suit(arr, mask):
    arr_r = apply_remap(arr, mask)
    valid = mask & np.isfinite(arr_r)
    return float(np.nanmean(arr_r[valid])) if valid.any() else np.nan

def regional_area_ge2_km2(arr, mask, pixel_area_km2):
    """Sum pixel areas where suitability class >= 2."""
    arr_c = arr.copy()
    arr_c[arr_c < 0] = np.nan
    arr_int = np.clip(
        np.where(np.isfinite(arr_c), arr_c, 0).astype(int), 0, 5
    )
    arr_int[arr_int == 0] = 1  # remap class 0 -> 1
    suitable = mask & (arr_int >= 2)
    return float(np.sum(pixel_area_km2[suitable]))


# ── 1. Time series CSV (1979-2018, annual) ────────────────────────────────────

def export_timeseries(mask, pixel_area_km2):
    print('\n[1] Exporting time series CSV (1979-2018) ...')

    obs_mean_all, fao_mean_all = [], []
    obs_area_all, fao_area_all = [], []

    for crop in CROPS:
        tag = crop['tag']
        obs_ms, fao_ms, obs_ar, fao_ar = [], [], [], []
        for year in YEARS_ALL:
            obs, _ = load_raster(obs_path(tag, year))
            fao, _ = load_raster(fao_path(tag, year))
            obs_ms.append(regional_mean_suit(obs, mask)          if obs is not None else np.nan)
            fao_ms.append(regional_mean_suit(fao, mask)          if fao is not None else np.nan)
            obs_ar.append(regional_area_ge2_km2(obs, mask, pixel_area_km2) if obs is not None else np.nan)
            fao_ar.append(regional_area_ge2_km2(fao, mask, pixel_area_km2) if fao is not None else np.nan)
        obs_mean_all.append(obs_ms)
        fao_mean_all.append(fao_ms)
        obs_area_all.append(obs_ar)
        fao_area_all.append(fao_ar)
        print(f'  {crop["label"]} done')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        obs_mean_agg = np.nanmean(obs_mean_all, axis=0)
        fao_mean_agg = np.nanmean(fao_mean_all, axis=0)
        obs_area_agg = np.nanmean(obs_area_all, axis=0)
        fao_area_agg = np.nanmean(fao_area_all, axis=0)

    # Overall annual series
    rows = []
    for i, year in enumerate(YEARS_ALL):
        rows.append({
            'year'          : year,
            'obs_mean_suit' : float(obs_mean_agg[i]),
            'fao_mean_suit' : float(fao_mean_agg[i]),
            'diff_mean_suit': float(obs_mean_agg[i] - fao_mean_agg[i]),
            'obs_area_km2'  : float(obs_area_agg[i]),
            'fao_area_km2'  : float(fao_area_agg[i]),
            'diff_area_km2' : float(obs_area_agg[i] - fao_area_agg[i]),
        })

    # Per-crop annual series
    per_crop_rows = []
    for ci, crop in enumerate(CROPS):
        for i, year in enumerate(YEARS_ALL):
            per_crop_rows.append({
                'crop'         : crop['label'],
                'year'         : year,
                'obs_mean_suit': float(obs_mean_all[ci][i]),
                'fao_mean_suit': float(fao_mean_all[ci][i]),
                'obs_area_km2' : float(obs_area_all[ci][i]),
                'fao_area_km2' : float(fao_area_all[ci][i]),
            })

    pd.DataFrame(rows).to_csv(
        f'{OUT_DIR}/overall_mean_suitability_timeseries.csv', index=False)
    pd.DataFrame(per_crop_rows).to_csv(
        f'{OUT_DIR}/per_crop_mean_suitability_timeseries.csv', index=False)
    print(f'  Saved: overall_mean_suitability_timeseries.csv')
    print(f'  Saved: per_crop_mean_suitability_timeseries.csv')


# ── 2 & 3. Difference maps GeoTIFF (full 40-year mean) ───────────────────────

def export_diff_maps(mask):
    print('\n[2/3] Exporting difference map GeoTIFFs (mean 1979-2018) ...')

    geo_info    = None
    crop_diffs  = []
    crop_labels = []

    for crop in CROPS:
        tag = crop['tag']
        obs_stack, fao_stack = [], []

        for year in YEARS_ALL:
            obs, gi = load_raster(obs_path(tag, year))
            fao, _  = load_raster(fao_path(tag, year))
            if obs is None or fao is None:
                continue
            if geo_info is None:
                geo_info = gi
            obs_stack.append(apply_remap(obs, mask))
            fao_stack.append(apply_remap(fao, mask))

        if not obs_stack:
            print(f'  WARNING: no data for {crop["label"]}')
            continue

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            obs_mean = np.nanmean(np.stack(obs_stack), axis=0)
            fao_mean = np.nanmean(np.stack(fao_stack), axis=0)

        diff = np.where(np.isfinite(obs_mean) & np.isfinite(fao_mean),
                        obs_mean - fao_mean, np.nan)
        diff[~mask] = np.nan
        crop_diffs.append(diff)
        crop_labels.append(crop['label'])

        valid_diff = diff[mask & np.isfinite(diff)]
        print(f'  {crop["label"]}: mean diff = {np.nanmean(valid_diff):.4f}, '
              f'obs > FAO = {(valid_diff > 0).mean()*100:.1f}%')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        overall_diff = np.nanmean(np.stack(crop_diffs), axis=0)
    overall_diff[~mask] = np.nan

    out_overall  = f'{OUT_DIR}/overall_diff_map_1979_2018.tif'
    out_per_crop = f'{OUT_DIR}/per_crop_diff_map_1979_2018.tif'
    save_raster_single(out_overall, overall_diff, geo_info)
    save_raster_multiband(out_per_crop, crop_diffs, crop_labels, geo_info)
    print(f'\n  Saved: overall_diff_map_1979_2018.tif')
    print(f'  Saved: per_crop_diff_map_1979_2018.tif  ({len(crop_diffs)} bands)')

    valid_overall = overall_diff[mask & np.isfinite(overall_diff)]
    print(f'\n  Overall: mean diff = {np.nanmean(valid_overall):.4f}, '
          f'obs > FAO = {(valid_overall > 0).mean()*100:.1f}%')

    return overall_diff, geo_info


# ── 4. Permafrost drivers CSV (full 40-year mean, permafrost pixels only) ─────

def export_perm_drivers(mask, overall_diff):
    print('\n[4] Exporting permafrost drivers CSV (mean 1979-2018) ...')

    PERM_VARS = [
        {'name': 'ALT',          'file': 'active_layer_depth.npy',  'agg': 'max'},
        {'name': 'soil_moisture', 'file': 'avail_soil_moisture.npy', 'agg': 'mean'},
    ]
    target = mask.shape

    pf_arr, _ = load_raster(PERMAFROST_PATH)
    if pf_arr is not None:
        pf_mask = mask & np.isin(np.round(pf_arr).astype(int), [1, 2])
        print(f'  Permafrost pixels (seasonally frozen excluded): {pf_mask.sum()}')
    else:
        pf_mask = mask
        print('  WARNING: could not load permafrost raster, using full mask')

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

        valid_vals = mean_all[pf_mask & np.isfinite(mean_all)]
        print(f'  {pv["name"]}: range [{valid_vals.min():.3f}, {valid_vals.max():.3f}]')

    valid = (pf_mask
             & np.isfinite(overall_diff)
             & np.isfinite(perm_data['ALT'])
             & np.isfinite(perm_data['soil_moisture']))

    rows, cols = np.where(valid)
    df = pd.DataFrame({
        'row'                   : rows,
        'col'                   : cols,
        'mean_delta_suitability': overall_diff[valid],
        'mean_ALT_m'            : perm_data['ALT'][valid],
        'mean_soil_moisture_mm' : perm_data['soil_moisture'][valid],
    })

    out_path = f'{OUT_DIR}/permafrost_drivers_overall_1979_2018.csv'
    df.to_csv(out_path, index=False)
    print(f'  Saved: permafrost_drivers_overall_1979_2018.csv  ({len(df)} pixels)')

    r_alt, p_alt = spearmanr(df['mean_ALT_m'],            df['mean_delta_suitability'])
    r_sm,  p_sm  = spearmanr(df['mean_soil_moisture_mm'], df['mean_delta_suitability'])
    print(f'\n  Spearman r (ALT vs ΔSuit):           {r_alt:.3f}  (p={p_alt:.4f})')
    print(f'  Spearman r (soil moisture vs ΔSuit): {r_sm:.3f}  (p={p_sm:.4f})')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Loading mask ...')
    mask           = load_mask()
    pixel_area_km2 = build_pixel_area_km2(mask)

    export_timeseries(mask, pixel_area_km2)
    overall_diff, geo_info = export_diff_maps(mask)
    export_perm_drivers(mask, overall_diff)

    print(f'\nAll exports saved to: {OUT_DIR}/')