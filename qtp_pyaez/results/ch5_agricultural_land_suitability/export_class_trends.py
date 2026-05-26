"""
Compute Overall Suitability Class Distribution — Chapter 5
===========================================================
Loads all raster files, computes per-class percentages averaged
across all 10 crops for each year, and saves to CSV.

Output:
  ./results/agricultural_land_suitability/outputs/csv/overall_class_distribution.csv
"""

import os
import numpy as np
import pandas as pd
from osgeo import gdal

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR  = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH = r'./data_input/qilian_mask_new.tif'
PERM_PATH = r'./data_input/permafrost_qilian.tif'
OUT_PATH  = r'./results/agricultural_land_suitability/outputs/csv/overall_class_distribution.csv'

YEARS_ALL = list(range(1979, 2019))

CROPS = [
    'combined_winter_barley',
    'combined_spring_barley',
    'combined_winter_wheat',
    'combined_spring_wheat',
    'combined_silage_maize',
    'combined_white_potato',
    'combined_oat',
    'combined_dry_pea',
    'combined_winter_rape',
    'combined_spring_rape',
]

os.chdir(WORK_DIR)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

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

# ── Main ──────────────────────────────────────────────────────────────────────
def compute():
    mask   = load_mask()
    n_mask = mask.sum()
    print(f'  Mask pixels: {n_mask}')

    rows = []
    for year in YEARS_ALL:
        counts = {c: 0 for c in [1, 2, 3, 4, 5]}
        n_loaded = 0

        for tag in CROPS:
            arr = load_raster(suit_path(tag, year))
            if arr is None:
                print(f'  WARNING: missing {tag} {year}')
                continue

            arr_int = arr.copy()
            arr_int[~mask] = np.nan
            arr_int = np.where(np.isfinite(arr_int), arr_int, 0).astype(int)
            arr_int = np.clip(arr_int, 0, 5)
            arr_int[arr_int == 0] = 1  # combine class 0 into class 1

            for c in [1, 2, 3, 4, 5]:
                counts[c] += int(np.sum((arr_int == c) & mask))
            n_loaded += 1

        if n_loaded > 0:
            total = n_mask * n_loaded
            row = {'Year': year}
            for c in [1, 2, 3, 4, 5]:
                row[f'pct_class_{c}'] = round(100.0 * counts[c] / total, 6)
        else:
            row = {'Year': year}
            for c in [1, 2, 3, 4, 5]:
                row[f'pct_class_{c}'] = np.nan

        rows.append(row)
        print(f'  {year}: classes 2-5 = '
              f'{row["pct_class_2"]:.3f} / {row["pct_class_3"]:.3f} / '
              f'{row["pct_class_4"]:.3f} / {row["pct_class_5"]:.3f} %')

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f'\nSaved to {OUT_PATH}')
    print(df.tail())


if __name__ == '__main__':
    compute()