"""
Export data for Thaw vs No-Thaw figures
========================================
Sources from analyze_permafrost_impact.py (suitability_score_analysis.py).

Figures this feeds:
  1. Time series — overall mean suitability and % suitable land,
     thaw vs no-thaw, 1979-2018  →  annual_suitability_timeseries.csv

  2. Permutation test dot plot — Sen's slope difference with CI,
     per crop + overall  →  already in trend_suit_results.csv
     (this script just verifies it exists and adds an overall-flagged column)

  3. Wilcoxon summary — median delta and % years positive per crop
     →  already in wilcoxon_suit_results.csv
     (this script just verifies it exists)

  4. Spatial TIFs:
     a. overall_mean_delta_1999_2018.tif  — mean ΔSuitability (Thaw − No-Thaw)
        averaged across all 10 crops, mean over 1999-2018
     b. overall_pixel_classification.tif — pixel-wise thaw effect classification
        (0=sig neg, 1=cons neg, 2=mixed, 3=cons pos, 4=sig pos)
        mean across all crops (modal class per pixel)

All outputs written to:
  ./results/permafrost_thaw_impact/thaw_vs_nothaw/figure_exports/
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon as wilcoxon_test
from osgeo import gdal

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR  = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH = r'./data_input/qilian mask.tif'
PERMAFROST_PATH = r'./data_input/permafrost_qilian.tif'

OUT_DIR   = r'./results/permafrost_thaw_impact/thaw_vs_nothaw/figure_exports'

# Existing outputs from suitability_score_analysis.py
TREND_CSV    = r'./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/4_trend_40yr/trend_suit_results.csv'
OVERALL_CSV  = r'./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/4_trend_40yr/overall_suit_trend_results.csv'
WILCOXON_CSV = r'./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/2_wilcoxon/wilcoxon_suit_results.csv'

# Per-crop annual delta TIFs (from analysis 1)
DELTA_TIF_ROOT = r'./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/1_delta_maps/tif'

# Pixel-wise classification TIFs (from spatial_analysis.py)
SPATIAL_ROOT = r'./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/5_spatial/2_wilcoxon'

YEARS_ALL  = list(range(1979, 2019))
YEARS_CF   = list(range(1999, 2019))
DIVERGENCE = 1999

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

def save_raster(path, arr, geo_info, nodata_val=-9999.0):
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

def cf_path(tag, year):
    return f'./data_output/final_classification_nothaw_fixed/{tag}/{year}_suitability_class.tif'

def remap_arr(arr_int):
    out = arr_int.copy()
    out[out == 0] = 1
    return out

def apply_remap(arr, mask):
    arr_c = arr.copy()
    arr_c[arr_c < 0] = np.nan
    arr_c[~mask] = np.nan
    int_arr = np.where(np.isfinite(arr_c), arr_c, 0).astype(int)
    int_arr = remap_arr(int_arr)
    return np.where(np.isfinite(arr_c), int_arr.astype(float), np.nan)

def regional_mean_suit(arr, mask):
    arr_r = apply_remap(arr, mask)
    valid = mask & np.isfinite(arr_r)
    return float(np.nanmean(arr_r[valid])) if valid.any() else np.nan

def regional_pct_ge2(arr, mask):
    arr_c = arr.copy()
    arr_c[arr_c < 0] = np.nan
    arr_int = np.clip(
        np.where(np.isfinite(arr_c), arr_c, 0).astype(int), 0, 5)
    return float(np.mean(arr_int[mask] >= 2) * 100) if mask.any() else np.nan


# ── Export 1: Annual time series CSV ─────────────────────────────────────────
# The existing trend_suit_results.csv only has MK stats, not the raw annual
# series. We recompute the annual regional means here.

def export_timeseries(mask):
    print('\n[1] Exporting annual time series CSV ...')

    obs_suit_all, cf_suit_all = [], []
    obs_pct_all,  cf_pct_all  = [], []

    for crop in CROPS:
        tag = crop['tag']
        obs_s, cf_s, obs_p, cf_p = [], [], [], []

        for year in YEARS_ALL:
            obs, _ = load_raster(obs_path(tag, year))
            obs_s.append(regional_mean_suit(obs, mask) if obs is not None else np.nan)
            obs_p.append(regional_pct_ge2(obs, mask)   if obs is not None else np.nan)

            if year < DIVERGENCE:
                # Pre-divergence: no-thaw = observed (scenarios identical)
                cf_s.append(obs_s[-1])
                cf_p.append(obs_p[-1])
            else:
                cf, _ = load_raster(cf_path(tag, year))
                cf_s.append(regional_mean_suit(cf, mask) if cf is not None else np.nan)
                cf_p.append(regional_pct_ge2(cf, mask)   if cf is not None else np.nan)

        obs_suit_all.append(np.array(obs_s))
        cf_suit_all.append(np.array(cf_s))
        obs_pct_all.append(np.array(obs_p))
        cf_pct_all.append(np.array(cf_p))
        print(f'  {crop["label"]} done')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        obs_suit_mean = np.nanmean(obs_suit_all, axis=0)
        cf_suit_mean  = np.nanmean(cf_suit_all,  axis=0)
        obs_pct_mean  = np.nanmean(obs_pct_all,  axis=0)
        cf_pct_mean   = np.nanmean(cf_pct_all,   axis=0)

    # Overall annual series
    overall_rows = []
    for i, year in enumerate(YEARS_ALL):
        overall_rows.append({
            'year'         : year,
            'obs_mean_suit': float(obs_suit_mean[i]),
            'cf_mean_suit' : float(cf_suit_mean[i]),
            'obs_pct_ge2'  : float(obs_pct_mean[i]),
            'cf_pct_ge2'   : float(cf_pct_mean[i]),
        })

    # Per-crop annual series
    per_crop_rows = []
    for ci, crop in enumerate(CROPS):
        for i, year in enumerate(YEARS_ALL):
            per_crop_rows.append({
                'crop'         : crop['label'],
                'year'         : year,
                'obs_mean_suit': float(obs_suit_all[ci][i]),
                'cf_mean_suit' : float(cf_suit_all[ci][i]),
                'obs_pct_ge2'  : float(obs_pct_all[ci][i]),
                'cf_pct_ge2'   : float(cf_pct_all[ci][i]),
            })

    pd.DataFrame(overall_rows).to_csv(
        f'{OUT_DIR}/overall_suitability_timeseries.csv', index=False)
    pd.DataFrame(per_crop_rows).to_csv(
        f'{OUT_DIR}/per_crop_suitability_timeseries.csv', index=False)
    print(f'  Saved: overall_suitability_timeseries.csv')
    print(f'  Saved: per_crop_suitability_timeseries.csv')


# ── Export 2: Permutation test CSV (verify + merge overall) ──────────────────

def export_permutation(mask):
    print('\n[2] Verifying permutation test CSV ...')

    if not os.path.exists(TREND_CSV):
        print(f'  WARNING: {TREND_CSV} not found — run suitability_score_analysis.py first')
        return
    if not os.path.exists(OVERALL_CSV):
        print(f'  WARNING: {OVERALL_CSV} not found — run suitability_score_analysis.py first')
        return

    df_crops   = pd.read_csv(TREND_CSV)
    df_overall = pd.read_csv(OVERALL_CSV)

    # Merge into one file, flagging the overall row
    df_crops['is_overall']   = False
    df_overall['is_overall'] = True
    df_combined = pd.concat([df_crops, df_overall], ignore_index=True)

    out_path = f'{OUT_DIR}/permutation_slope_diff.csv'
    df_combined.to_csv(out_path, index=False)
    print(f'  Saved: permutation_slope_diff.csv  ({len(df_combined)} rows)')

    # Quick summary of significant results
    df_perm = df_combined[
        (df_combined['period'] == '1979-2018') &
        (df_combined['metric'] == 'mean_suit')
    ][['crop', 'slope_difference', 'perm_p', 'perm_sig', 'boot_p', 'boot_sig']]
    print('\n  Permutation results (mean_suit, 1979-2018):')
    print(df_perm.to_string(index=False))


# ── Export 3: Wilcoxon CSV (verify) ──────────────────────────────────────────

def export_wilcoxon():
    print('\n[3] Verifying Wilcoxon CSV ...')

    if not os.path.exists(WILCOXON_CSV):
        print(f'  WARNING: {WILCOXON_CSV} not found — run suitability_score_analysis.py first')
        return

    df = pd.read_csv(WILCOXON_CSV)
    out_path = f'{OUT_DIR}/wilcoxon_results.csv'
    df.to_csv(out_path, index=False)
    print(f'  Saved: wilcoxon_results.csv  ({len(df)} crops)')
    print('\n  Summary:')
    print(df[['crop', 'median_delta', 'pct_years_positive',
              'p_greater_zero', 'sig_positive']].to_string(index=False))


# ── Export 4a: Overall mean delta TIF (1999-2018) ────────────────────────────

def export_mean_delta_tif(mask):
    print('\n[4a] Exporting overall mean delta TIF (1999-2018) ...')

    geo_info    = None
    crop_means  = []

    for crop in CROPS:
        tag = crop['tag']
        year_arrays = []

        for year in YEARS_CF:
            tif_path = f'{DELTA_TIF_ROOT}/{tag}/{year}_delta_suit.tif'
            if not os.path.exists(tif_path):
                print(f'  WARNING: missing {tif_path}')
                continue
            arr, gi = load_raster(tif_path)
            if arr is None:
                continue
            if geo_info is None:
                geo_info = gi
            arr[~mask] = np.nan
            year_arrays.append(arr)

        if not year_arrays:
            print(f'  WARNING: no delta TIFs found for {crop["label"]}')
            continue

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            crop_mean = np.nanmean(np.stack(year_arrays), axis=0)
        crop_mean[~mask] = np.nan
        crop_means.append(crop_mean)
        print(f'  {crop["label"]}: mean delta = {np.nanmean(crop_mean[mask & np.isfinite(crop_mean)]):.4f}')

    if not crop_means:
        print('  ERROR: no crop delta arrays found — run suitability_score_analysis.py first')
        return

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        overall_mean = np.nanmean(np.stack(crop_means), axis=0)
    overall_mean[~mask] = np.nan

    out_path = f'{OUT_DIR}/overall_mean_delta_1999_2018.tif'
    save_raster(out_path, overall_mean, geo_info)
    print(f'  Saved: overall_mean_delta_1999_2018.tif')
    print(f'  Overall mean delta: {np.nanmean(overall_mean[mask & np.isfinite(overall_mean)]):.4f}')
    print(f'  Range: [{np.nanmin(overall_mean[mask & np.isfinite(overall_mean)]):.4f}, '
          f'{np.nanmax(overall_mean[mask & np.isfinite(overall_mean)]):.4f}]')


# ── Export 4b: Overall pixel-wise classification TIF ─────────────────────────
# The spatial_analysis.py script saves per-crop classification TIFs.
# Here we compute the modal (most common) class across all crops per pixel,
# which gives the overall pixel-wise thaw effect classification.
#
# Classification codes (matching spatial_analysis.py):
#   0 = significantly negative
#   1 = consistently negative
#   2 = mixed / no effect
#   3 = consistently positive
#   4 = significantly positive

def export_classification_tif(mask):
    print('\n[4b] Exporting overall pixel-wise classification TIF ...')

    geo_info      = None
    crop_class_arrays = []

    for crop in CROPS:
        tag = crop['tag']
        # spatial_analysis.py saves classification as:
        # {SPATIAL_ROOT}/{tag}_pixel_classification.tif
        class_path = f'{SPATIAL_ROOT}/{tag}_classification.tif'

        if not os.path.exists(class_path):
            print(f'  WARNING: missing {class_path} — run spatial_analysis.py first')
            continue

        arr, gi = load_raster(class_path)
        if arr is None:
            continue
        if geo_info is None:
            geo_info = gi

        arr[~mask] = np.nan
        crop_class_arrays.append(arr)
        print(f'  {crop["label"]}: loaded classification TIF')

    if not crop_class_arrays:
        print('  ERROR: no classification TIFs found — run spatial_analysis.py first')
        return

    # Modal class across crops per pixel
    stack = np.stack(crop_class_arrays, axis=0)  # (n_crops, rows, cols)

    # For each pixel count occurrences of each class (0-4) and take the mode
    n_classes = 5
    rows, cols = mask.shape
    modal_class = np.full((rows, cols), np.nan)

    valid_pixels = mask & np.any(np.isfinite(stack), axis=0)
    ry, cx = np.where(valid_pixels)

    for r, c in zip(ry, cx):
        pixel_vals = stack[:, r, c]
        pixel_vals = pixel_vals[np.isfinite(pixel_vals)].astype(int)
        if len(pixel_vals) == 0:
            continue
        counts = np.bincount(pixel_vals, minlength=n_classes)
        modal_class[r, c] = float(np.argmax(counts))

    modal_class[~mask] = np.nan

    out_path = f'{OUT_DIR}/overall_pixel_classification.tif'
    save_raster(out_path, modal_class, geo_info)
    print(f'  Saved: overall_pixel_classification.tif')

    # Summary of class distribution
    labels = {0: 'Sig negative', 1: 'Cons negative', 2: 'Mixed/none',
              3: 'Cons positive', 4: 'Sig positive'}
    valid_cls = modal_class[mask & np.isfinite(modal_class)].astype(int)
    total = len(valid_cls)
    print(f'\n  Overall pixel classification (n={total}):')
    for code, label in labels.items():
        n = int((valid_cls == code).sum())
        print(f'    {label}: {n} ({n/total*100:.1f}%)')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Loading mask ...')
    mask = load_mask()

    # export_timeseries(mask)
    # export_permutation(mask)
    # export_wilcoxon()
    # export_mean_delta_tif(mask)
    export_classification_tif(mask)

    print(f'\nAll exports saved to: {OUT_DIR}/')