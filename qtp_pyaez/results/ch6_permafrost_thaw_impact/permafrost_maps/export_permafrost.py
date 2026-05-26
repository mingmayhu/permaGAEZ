"""
export_permafrost_majority_vote.py

Exports majority vote permafrost classification tifs for:
1. FAO - whole period (1979-2018)
2. Thaw scenario - whole period (1979-2018)
3. P-GAEZ - period 1 (1979-1998)
4. P-GAEZ - period 2 (1999-2018)

Output: 0 = no permafrost, 1 = permafrost
"""

import os
import numpy as np
from osgeo import gdal

WORK_DIR = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
os.chdir(WORK_DIR)

REF_PATH = r'./data_input/qilian_mask_new.tif'
OUT_DIR  = r'./results/permafrost_maps'
os.makedirs(OUT_DIR, exist_ok=True)

YEARS_FULL = list(range(1979, 2019))
YEARS_P1   = list(range(1979, 1999))
YEARS_P2   = list(range(1999, 2019))

ref_ds     = gdal.Open(REF_PATH)
geotrans   = ref_ds.GetGeoTransform()
projection = ref_ds.GetProjection()
nrow       = ref_ds.RasterYSize
ncol       = ref_ds.RasterXSize
ref_ds     = None

def save_tif(arr, path):
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(path, ncol, nrow, 1, gdal.GDT_Float32)
    ds.SetGeoTransform(geotrans)
    ds.SetProjection(projection)
    ds.GetRasterBand(1).WriteArray(arr.astype(np.float32))
    ds.GetRasterBand(1).SetNoDataValue(-9999)
    ds.FlushCache()
    ds = None
    print(f"Saved: {path}")

def majority_vote_npy(years, path_fn):
    stack = []
    for y in years:
        p = path_fn(y)
        if not os.path.exists(p):
            continue
        stack.append(np.load(p).astype(float))
    if not stack:
        return None
    return (np.mean(np.array(stack), axis=0) >= 0.5).astype(np.float32)

def majority_vote_tif(years, path_fn):
    stack = []
    for y in years:
        p = path_fn(y)
        ds = gdal.Open(p)
        if ds is None:
            continue
        arr = ds.GetRasterBand(1).ReadAsArray().astype(float)
        nd  = ds.GetRasterBand(1).GetNoDataValue()
        if nd is not None:
            arr[arr == nd] = np.nan
        stack.append((arr <= 2).astype(float))
        ds = None
    if not stack:
        return None
    return (np.nanmean(np.array(stack), axis=0) >= 0.5).astype(np.float32)

print("Processing FAO whole period...")
fao_whole = majority_vote_tif(YEARS_FULL, lambda y: f'./data_output/original/module1/{y}/permafrost.tif')
if fao_whole is not None:
    save_tif(fao_whole, os.path.join(OUT_DIR, 'permafrost_fao_whole_period.tif'))

print("Processing thaw scenario whole period...")
thaw_whole = majority_vote_npy(YEARS_FULL, lambda y: f'./data_output/module1/permafrost_maps/permafrost_{y}.npy')
if thaw_whole is not None:
    save_tif(thaw_whole, os.path.join(OUT_DIR, 'permafrost_thaw_whole_period.tif'))

print("Processing P-GAEZ period 1 (1979-1998)...")
pgaez_p1 = majority_vote_npy(YEARS_P1, lambda y: f'./data_output/module1/permafrost_maps/permafrost_{y}.npy')
if pgaez_p1 is not None:
    save_tif(pgaez_p1, os.path.join(OUT_DIR, 'permafrost_pgaez_period1.tif'))

print("Processing P-GAEZ period 2 (1999-2018)...")
pgaez_p2 = majority_vote_npy(YEARS_P2, lambda y: f'./data_output/module1/permafrost_maps/permafrost_{y}.npy')
if pgaez_p2 is not None:
    save_tif(pgaez_p2, os.path.join(OUT_DIR, 'permafrost_pgaez_period2.tif'))

print("\nDone. All tifs saved to", OUT_DIR)