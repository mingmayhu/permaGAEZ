"""
Export GeoTIFFs for Chapter 4 spatial maps
===========================================
Outputs 8 GeoTIFFs:

Climate:
  1. mean_temperature_1979-2018.tif       — mean annual temperature, full period
  2. delta_temperature_post-pre.tif       — ΔTemp (1999–2018 minus 1979–1998)
  3. mean_precipitation_1979-2018.tif     — mean annual total precip, full period
  4. delta_precipitation_post-pre.tif     — ΔPrecip (1999–2018 minus 1979–1998)

Permafrost:
  5. mean_alt_1979-2018.tif               — mean ALT, full period
  6. delta_alt_post-pre.tif               — ΔALT (1999–2018 minus 1979–1998)
  7. mean_soil_moisture_1979-2018.tif     — mean available soil moisture, full period
  8. delta_soil_moisture_post-pre.tif     — ΔSoil Moisture (1999–2018 minus 1979–1998)

All outputs share the same CRS and geotransform as the mask raster.
Nodata = -9999.0
"""

import os
import numpy as np
from osgeo import gdal, osr

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR  = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH = r'./data_input/qilian_mask_new.tif'
PERM_DIR  = r'./data_input/permafrost_yearly'
CLIM_DIR  = r'./data_input/climate_yearly'
OUT_DIR   = r'./results/climate_permafrost_trends/outputs/tifs'

YEARS_ALL  = list(range(1979, 2019))
YEARS_PRE  = list(range(1979, 1999))
YEARS_POST = list(range(1999, 2019))

NODATA = -9999.0

os.chdir(WORK_DIR)
os.makedirs(OUT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_raster(path):
    """Load a raster to a float numpy array, masking nodata as NaN."""
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(f'Cannot open raster: {path}')
    band   = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    arr    = ds.ReadAsArray().astype(float)
    arr[arr < -1e10] = np.nan
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr

def get_raster_georef(path):
    """Return (geotransform, projection_wkt) from a raster."""
    ds = gdal.Open(path)
    return ds.GetGeoTransform(), ds.GetProjection()

def load_mask():
    return load_raster(MASK_PATH).astype(bool)

def load_clim_spatial(var_file, years, mask, agg='mean'):
    """Spatial mean map averaged over the given years."""
    stack = []
    for year in years:
        path = f'{CLIM_DIR}/{year}/{var_file}'
        if not os.path.exists(path):
            continue
        arr = np.load(path).astype(float)
        if arr.ndim == 3:
            arr = np.nansum(arr, axis=2) if agg == 'sum' \
                  else np.nanmean(arr, axis=2)
        arr[~mask] = np.nan
        stack.append(arr)
    return np.nanmean(stack, axis=0) if stack else np.full(mask.shape, np.nan)

def load_perm_spatial(var_file, years, mask, agg='mean'):
    """Spatial mean map averaged over the given years."""
    stack = []
    for year in years:
        path = f'{PERM_DIR}/{year}/{var_file}'
        if not os.path.exists(path):
            continue
        arr = np.load(path).astype(float)
        if arr.ndim == 3:
            arr = np.nanmean(arr, axis=2) if agg == 'mean' \
                  else np.nanmax(arr, axis=2)
        arr[~mask] = np.nan
        stack.append(arr)
    return np.nanmean(stack, axis=0) if stack else np.full(mask.shape, np.nan)

def write_tif(arr, mask, geotransform, projection, out_path, nodata=NODATA):
    """Write a 2-D float array as a single-band GeoTIFF."""
    out = np.where(mask & np.isfinite(arr), arr, nodata)
    rows, cols = out.shape
    driver = gdal.GetDriverByName('GTiff')
    ds_out = driver.Create(
        out_path, cols, rows, 1, gdal.GDT_Float32,
        options=['COMPRESS=LZW', 'TILED=YES']
    )
    ds_out.SetGeoTransform(geotransform)
    ds_out.SetProjection(projection)
    band = ds_out.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(out.astype(np.float32))
    band.FlushCache()
    ds_out = None
    print(f'  ✓ {os.path.basename(out_path)}')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    mask = load_mask()
    geotransform, projection = get_raster_georef(MASK_PATH)

    # ── 1 & 2: Temperature ────────────────────────────────────────────────────
    print('\nTemperature …')
    tmax_all  = load_clim_spatial('TempMax.npy', YEARS_ALL, mask, 'mean')
    tmin_all  = load_clim_spatial('TempMin.npy', YEARS_ALL, mask, 'mean')
    tmean_all = (tmax_all + tmin_all) / 2

    tmax_pre  = load_clim_spatial('TempMax.npy', YEARS_PRE,  mask, 'mean')
    tmin_pre  = load_clim_spatial('TempMin.npy', YEARS_PRE,  mask, 'mean')
    tmean_pre = (tmax_pre + tmin_pre) / 2

    tmax_post  = load_clim_spatial('TempMax.npy', YEARS_POST, mask, 'mean')
    tmin_post  = load_clim_spatial('TempMin.npy', YEARS_POST, mask, 'mean')
    tmean_post = (tmax_post + tmin_post) / 2

    write_tif(tmean_all,             mask, geotransform, projection,
              f'{OUT_DIR}/mean_temperature_1979-2018.tif')
    write_tif(tmean_post - tmean_pre, mask, geotransform, projection,
              f'{OUT_DIR}/delta_temperature_post-pre.tif')

    # ── 3 & 4: Precipitation ──────────────────────────────────────────────────
    print('\nPrecipitation …')
    prec_all  = load_clim_spatial('Precip.npy', YEARS_ALL,  mask, 'sum')
    prec_pre  = load_clim_spatial('Precip.npy', YEARS_PRE,  mask, 'sum')
    prec_post = load_clim_spatial('Precip.npy', YEARS_POST, mask, 'sum')

    write_tif(prec_all,            mask, geotransform, projection,
              f'{OUT_DIR}/mean_precipitation_1979-2018.tif')
    write_tif(prec_post - prec_pre, mask, geotransform, projection,
              f'{OUT_DIR}/delta_precipitation_post-pre.tif')

    # ── 5 & 6: ALT ────────────────────────────────────────────────────────────
    print('\nActive Layer Thickness …')
    alt_all  = load_perm_spatial('active_layer_depth.npy', YEARS_ALL,  mask, 'max')
    alt_pre  = load_perm_spatial('active_layer_depth.npy', YEARS_PRE,  mask, 'max')
    alt_post = load_perm_spatial('active_layer_depth.npy', YEARS_POST, mask, 'max')

    write_tif(alt_all,           mask, geotransform, projection,
              f'{OUT_DIR}/mean_alt_1979-2018.tif')
    write_tif(alt_post - alt_pre, mask, geotransform, projection,
              f'{OUT_DIR}/delta_alt_post-pre.tif')

    # ── 7 & 8: Soil Moisture ──────────────────────────────────────────────────
    print('\nSoil Moisture …')
    sm_all  = load_perm_spatial('avail_soil_moisture.npy', YEARS_ALL,  mask, 'mean')
    sm_pre  = load_perm_spatial('avail_soil_moisture.npy', YEARS_PRE,  mask, 'mean')
    sm_post = load_perm_spatial('avail_soil_moisture.npy', YEARS_POST, mask, 'mean')

    write_tif(sm_all,          mask, geotransform, projection,
              f'{OUT_DIR}/mean_soil_moisture_1979-2018.tif')
    write_tif(sm_post - sm_pre, mask, geotransform, projection,
              f'{OUT_DIR}/delta_soil_moisture_post-pre.tif')

    print(f'\n✓ All 8 GeoTIFFs saved to: {OUT_DIR}/')