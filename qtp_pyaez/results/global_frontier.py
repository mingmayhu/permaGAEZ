"""
Binary Agricultural Frontier Map — fixed write version
"""

import os
import numpy as np
from osgeo import gdal

# ── Configuration ─────────────────────────────────────────────────────────────
HANNAH_DIR   = '/Users/ming-mayhu/Downloads/Crop_Suitability_3Method'
CURRENT_FILE = '/Users/ming-mayhu/Downloads/Crop_Suitability_3Method/present/allcrops_current_binary.tif'
OUT_DIR      = '/Users/ming-mayhu/Downloads/Crop_Suitability_3Method/outputs'
os.makedirs(OUT_DIR, exist_ok=True)

GCMS_RCP85 = ['ac', 'bc', 'cc', 'cn', 'gf', 'gs', 'ho', 'he', 'hg',
              'in', 'ip', 'mc', 'mg', 'mi', 'mp', 'mr', 'no']

CROPS = ['Cassava', 'Corn', 'Cotton', 'Millet', 'Oilpalm',
         'Peanut', 'Potato', 'Rice', 'Sorghum', 'Soy', 'Sugar', 'Wheat']

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_raster_info(path):
    ds   = gdal.Open(path)
    gt   = ds.GetGeoTransform()
    proj = ds.GetProjection()
    nx   = ds.RasterXSize
    ny   = ds.RasterYSize
    ds   = None
    return gt, proj, nx, ny

def read_band(path, ny, nx):
    ds     = gdal.Open(path)
    band   = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    data   = band.ReadAsArray().astype(np.float64)
    ds     = None
    if nodata is not None:
        data[np.abs(data - nodata) < (abs(nodata) * 1e-6 + 1)] = np.nan
    data = data.astype(np.float32)
    if data.shape != (ny, nx):
        return None
    return data

def write_raster_int16(path, data_int16, gt, proj):
    """Write a pre-prepared int16 numpy array directly — no casting inside."""
    ny, nx = data_int16.shape
    driver = gdal.GetDriverByName('GTiff')
    ds     = driver.Create(path, nx, ny, 1, gdal.GDT_Int16,
                           options=['COMPRESS=LZW', 'TILED=YES'])
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(-9999)
    band.WriteArray(data_int16)
    band.ComputeStatistics(False)   # force stats so viewers can auto-stretch
    ds.FlushCache()
    ds = None
    # Verify
    ds2  = gdal.Open(path)
    back = ds2.GetRasterBand(1).ReadAsArray()
    ds2  = None
    print(f'  Saved → {path}')
    print(f'  Verify: unique values = {np.unique(back)}, non-zero = {(back > 0).sum():,}')

# ── Step 1: Load reference grid & current baseline ────────────────────────────
print('Loading current baseline...')
gt, proj, nx, ny = get_raster_info(CURRENT_FILE)
current_data     = read_band(CURRENT_FILE, ny, nx)
current_suitable = np.where(np.isfinite(current_data), current_data > 0, False)
print(f'  Currently suitable cells: {current_suitable.sum():,}')

# ── Step 2: Build future ensemble ─────────────────────────────────────────────
print('\nProcessing future suitability (RCP8.5, 2070)...')
future_gcm_count = np.zeros((ny, nx), dtype=np.float32)

for crop in CROPS:
    print(f'  {crop}...', end=' ', flush=True)
    crop_sum = np.zeros((ny, nx), dtype=np.float32)
    n_found  = 0
    for gcm in GCMS_RCP85:
        fname = os.path.join(HANNAH_DIR, 'future', crop,
                             f'{crop}_{gcm}_3method_rcp85_2070_bin.tif')
        if not os.path.exists(fname):
            continue
        data = read_band(fname, ny, nx)
        if data is None:
            continue
        valid    = np.isfinite(data)
        crop_sum = np.where(valid, crop_sum + data, crop_sum)
        n_found += 1
    if n_found == 0:
        print('NO FILES FOUND — skipping')
        continue
    future_gcm_count = np.maximum(future_gcm_count, crop_sum)
    print(f'{n_found} GCMs, {(crop_sum > 0).sum():,} suitable cells')

# ── Step 3: Compute frontier ──────────────────────────────────────────────────
print('\nComputing frontier...')
future_suitable = future_gcm_count > 0
frontier_bool   = future_suitable & ~current_suitable

print(f'  Future suitable:  {future_suitable.sum():,} cells')
print(f'  Frontier cells:   {frontier_bool.sum():,}')

# Convert to int16 explicitly before writing
frontier_int16   = frontier_bool.astype(np.int16)          # True→1, False→0
agreement_int16  = np.where(frontier_bool,
                            future_gcm_count.astype(np.int16),
                            np.int16(0))

print(f'  frontier_int16 unique:   {np.unique(frontier_int16)}')
print(f'  agreement_int16 unique:  {np.unique(agreement_int16)[:10]}')

# ── Step 4: Write ─────────────────────────────────────────────────────────────
print('\nWriting outputs...')
write_raster_int16(os.path.join(OUT_DIR, 'frontier_binary.tif'),
                   frontier_int16, gt, proj)
write_raster_int16(os.path.join(OUT_DIR, 'frontier_gcm_agreement.tif'),
                   agreement_int16, gt, proj)

print('\nDone.')