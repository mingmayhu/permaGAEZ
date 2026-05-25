"""
Compute Overall Suitability Class Area (km²) — Chapter 5
=========================================================
Loads all raster files, computes per-class area in km² averaged
across all 10 crops for each year, and saves to CSV.

Output:
  ./results/agricultural_land_suitability/outputs/csv/overall_class_area_km2.csv
"""

import os
import numpy as np
import pandas as pd
from osgeo import gdal

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR  = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH = r'./data_input/qilian_mask_new.tif'
PERM_PATH = r'./data_input/permafrost_qilian.tif'
OUT_PATH  = r'./results/agricultural_land_suitability/outputs/csv/overall_class_area_km2.csv'

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

def suit_path(tag, year):
    return f'./data_output/final_classification_fixed/{tag}/{year}_suitability_class.tif'

# ── Main ──────────────────────────────────────────────────────────────────────
def compute():
    mask           = load_mask()
    pixel_area_km2 = build_pixel_area_km2(mask)
    print(f'  Mask pixels: {mask.sum()}')
    print(f'  Total mask area: {pixel_area_km2.sum():.1f} km²')

    rows = []
    for year in YEARS_ALL:
        # accumulate area per class across crops, then average
        area_sums = {c: 0.0 for c in [1, 2, 3, 4, 5]}
        n_loaded  = 0

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
                area_sums[c] += float(np.sum(pixel_area_km2[(arr_int == c) & mask]))
            n_loaded += 1

        row = {'Year': year}
        if n_loaded > 0:
            for c in [1, 2, 3, 4, 5]:
                row[f'area_class_{c}_km2'] = round(area_sums[c] / n_loaded, 4)
        else:
            for c in [1, 2, 3, 4, 5]:
                row[f'area_class_{c}_km2'] = np.nan

        rows.append(row)
        print(f'  {year}: classes 2-5 = '
              f'{row["area_class_2_km2"]:.1f} / {row["area_class_3_km2"]:.1f} / '
              f'{row["area_class_4_km2"]:.1f} / {row["area_class_5_km2"]:.1f} km²')

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f'\nSaved to {OUT_PATH}')
    print(df.tail())


if __name__ == '__main__':
    compute()