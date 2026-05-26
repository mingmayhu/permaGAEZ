import numpy as np
from osgeo import gdal

WORK_DIR        = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH       = r'./data_input/qilian_mask_new.tif'
PERMAFROST_PATH = r'./data_input/permafrost_qilian.tif'

import os
os.chdir(WORK_DIR)

def load_raster(path):
    ds = gdal.Open(path)
    band   = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    arr    = ds.ReadAsArray().astype(float)
    arr[arr < -1e10] = np.nan
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr

mask = load_raster(MASK_PATH).astype(bool)
pf   = load_raster(PERMAFROST_PATH)
lake_mask = ((pf == 0) | ~np.isfinite(pf)) & mask
mask[lake_mask] = False

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

total_km2  = area_2d.sum()
n_pixels   = mask.sum()

print(f'Mask pixels:      {n_pixels}')
print(f'Total area:       {total_km2:.1f} km²')
print(f'Mean pixel area:  {total_km2/n_pixels:.4f} km²')