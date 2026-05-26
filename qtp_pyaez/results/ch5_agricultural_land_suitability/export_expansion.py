"""
Compute Expansion vs Intensification — Chapter 5
=================================================
Loads raster files for pre (1979-1998) and post (1999-2018) periods,
computes expansion and intensification metrics per crop, saves to CSV.

Outputs:
  ./results/agricultural_land_suitability/outputs/csv/expansion_vs_intensification.csv
  ./results/agricultural_land_suitability/outputs/csv/expansion_counts.csv
"""

import os
import numpy as np
import pandas as pd
from osgeo import gdal

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR  = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH = r'./data_input/qilian_mask_new.tif'
PERM_PATH = r'./data_input/permafrost_qilian.tif'
OUT_DIR   = r'./results/agricultural_land_suitability/outputs/csv'

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
        return None
    band   = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    arr    = ds.ReadAsArray().astype(float)
    arr[arr < -1e10] = np.nan
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr

def load_mask():
    mask   = load_raster(MASK_PATH).astype(bool)
    pf_arr = load_raster(PERM_PATH)
    if pf_arr is not None:
        lake_mask       = ((pf_arr == 0) | ~np.isfinite(pf_arr)) & mask
        mask[lake_mask] = False
        print(f'  Excluded {lake_mask.sum()} lake pixels from mask')
    return mask

def suit_path(tag, year):
    return f'./data_output/final_classification_fixed/{tag}/{year}_suitability_class.tif'

def load_period_mean(tag, years, mask):
    """Load and average suitability rasters over a period."""
    stack = []
    for year in years:
        arr = load_raster(suit_path(tag, year))
        if arr is None:
            continue
        arr[~mask] = np.nan
        arr = np.where(np.isfinite(arr), arr, 0).astype(int)
        arr = np.clip(arr, 0, 5)
        arr[arr == 0] = 1  # combine class 0 into class 1
        arr_flt = arr.astype(float)
        arr_flt[~mask] = np.nan
        stack.append(arr_flt)
    return np.nanmean(stack, axis=0) if stack else None

# ── Main ──────────────────────────────────────────────────────────────────────
def compute(mask):
    expansion_counts = []
    class_units      = []

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        print(f'  Processing {label} ...')

        pre  = load_period_mean(tag, YEARS_PRE,  mask)
        post = load_period_mean(tag, YEARS_POST, mask)

        if pre is None or post is None:
            print(f'  WARNING: missing data for {label}, skipping')
            continue

        pre_cls  = np.round(pre).astype(int)
        post_cls = np.round(post).astype(int)
        valid    = mask & np.isfinite(pre) & np.isfinite(post)

        newly_suitable   = valid & (pre_cls <= 1) & (post_cls >= 2)
        lost_suitable    = valid & (pre_cls >= 2) & (post_cls <= 1)
        already_suitable = valid & (pre_cls >= 2) & (post_cls >= 2)

        intensification = float(np.nanmean(post[already_suitable]) -
                                np.nanmean(pre[already_suitable])) \
                          if already_suitable.sum() > 0 else np.nan

        expansion_counts.append({
            'crop':                        label,
            'newly_suitable_px':           int(newly_suitable.sum()),
            'lost_suitable_px':            int(lost_suitable.sum()),
            'net_expansion_px':            int(newly_suitable.sum() - lost_suitable.sum()),
            'intensification_delta_class': round(intensification, 4)
                                           if not np.isnan(intensification) else np.nan,
        })

        expansion_units      = float(np.nansum(post[newly_suitable] - pre[newly_suitable])) \
                               if newly_suitable.sum() > 0 else 0.0
        intensification_units = float(np.nansum(post[already_suitable] - pre[already_suitable])) \
                                if already_suitable.sum() > 0 else 0.0
        lost_units           = float(np.nansum(post[lost_suitable] - pre[lost_suitable])) \
                               if lost_suitable.sum() > 0 else 0.0
        total = expansion_units + intensification_units + lost_units

        class_units.append({
            'crop':                     label,
            'expansion_units':          round(expansion_units, 2),
            'intensification_units':    round(intensification_units, 2),
            'lost_units':               round(lost_units, 2),
            'total_units':              round(total, 2),
            'pct_from_expansion':       round(100 * expansion_units / total, 1)
                                        if total > 0 else np.nan,
            'pct_from_intensification': round(100 * intensification_units / total, 1)
                                        if total > 0 else np.nan,
            'pct_from_loss':            round(100 * lost_units / total, 1)
                                        if total > 0 else np.nan,
        })

        print(f'    Newly suitable: {newly_suitable.sum()} px | '
              f'Lost: {lost_suitable.sum()} px | '
              f'Intensification: {intensification:.4f} class units')

    df_counts = pd.DataFrame(expansion_counts)
    df_units  = pd.DataFrame(class_units)

    df_counts.to_csv(f'{OUT_DIR}/expansion_counts.csv', index=False)
    df_units.to_csv(f'{OUT_DIR}/expansion_vs_intensification.csv', index=False)
    print(f'\nSaved to {OUT_DIR}/')
    print(df_units.to_string(index=False))


if __name__ == '__main__':
    mask = load_mask()
    compute(mask)