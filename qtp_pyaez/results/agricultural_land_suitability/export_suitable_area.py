"""
Export per-crop suitable land area (km²) to CSV
================================================
Computes area of pixels with suitability class >= 2 for each crop and year,
using cosine latitude correction on 0.1° resolution grid.

Output: ./results/agricultural_land_suitability/outputs/csv/per_crop_area_suitable_km2.csv
Columns: Year, <crop_label> x10, Overall (mean across crops)
"""

import os
import numpy as np
import pandas as pd
from osgeo import gdal

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR        = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH       = r'./data_input/qilian_mask_new.tif'
PERMAFROST_PATH = r'./data_input/permafrost_qilian.tif'
OUT_PATH        = r'./results/agricultural_land_suitability/outputs/csv/per_crop_area_suitable_km2.csv'

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
    pf_arr = load_raster(PERMAFROST_PATH)
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

def suit_path(tag, year):
    return f'./data_output/final_classification_fixed/{tag}/{year}_suitability_class.tif'

def area_ge2_km2(arr, mask, pixel_area_km2):
    """Sum pixel areas where suitability class >= 2 (class 0 remapped to 1)."""
    arr_int = np.where(mask & np.isfinite(arr), arr, 0).astype(int)
    arr_int = np.clip(arr_int, 0, 5)
    arr_int[arr_int == 0] = 1          # remap class 0 -> 1
    suitable = mask & (arr_int >= 2)
    return float(np.sum(pixel_area_km2[suitable]))

# ── Main ──────────────────────────────────────────────────────────────────────
mask           = load_mask()
pixel_area_km2 = build_pixel_area_km2(mask)

rows = []
for year in YEARS_ALL:
    row = {'Year': year}
    crop_areas = []
    for crop in CROPS:
        arr = load_raster(suit_path(crop['tag'], year))
        if arr is not None:
            val = area_ge2_km2(arr, mask, pixel_area_km2)
        else:
            val = np.nan
        row[crop['label']] = val
        crop_areas.append(val)
    row['Overall'] = float(np.nanmean(crop_areas))
    rows.append(row)
    print(f'  {year} done')

df = pd.DataFrame(rows)
df.to_csv(OUT_PATH, index=False)
print(f'\nSaved to {OUT_PATH}')
print(df.tail())