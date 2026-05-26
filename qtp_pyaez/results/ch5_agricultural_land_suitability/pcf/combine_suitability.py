"""
Step 1: Combine agricultural land suitability across all 10 crops.

Produces two combined rasters for a given year:
  - combined_suitability_max_{year}.tif  : best possible use (max class across crops)
  - combined_suitability_mean_{year}.tif : average suitability (mean class across crops)

Output is written to:
  data_output/pcf/combined_suitability/
"""

import os
import numpy as np
from osgeo import gdal

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR   = '/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
SUIT_DIR   = os.path.join(BASE_DIR, 'data_output', 'final_classification_fixed')
OUT_DIR    = os.path.join(BASE_DIR, 'data_output', 'pcf', 'combined_suitability')

YEAR       = 2018   # Change this or loop over years as needed

CROPS = [
    'combined_winter_barley',
    'combined_spring_barley',
    'combined_winter_wheat',
    'combined_spring_wheat',
    'combined_winter_rape',
    'combined_spring_rape',
    'combined_dry_pea',
    'combined_oat',
    'combined_white_potato',
    'combined_silage_maize',
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_tif(path):
    """Load a GeoTIFF and return (array, dataset). Nodata set to NaN.
    Class 0 (model exclusion) is treated as Class 1 (not suitable)."""
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(f'Cannot open: {path}')
    band = ds.GetRasterBand(1)
    arr  = band.ReadAsArray().astype(np.float32)
    nodata = band.GetNoDataValue()
    if nodata is not None:
        arr[arr == nodata] = np.nan
    # Treat 0 as Class 1 (not suitable) — 0 indicates model exclusion,
    # which is functionally equivalent to unsuitable for agricultural use
    arr[arr == 0] = 1
    return arr, ds


def save_tif(out_path, array, ref_ds):
    """Save a float32 array as GeoTIFF using georef from ref_ds."""
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = array.shape
    out_ds = driver.Create(out_path, cols, rows, 1, gdal.GDT_Float32,
                           options=['COMPRESS=LZW'])
    out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
    out_ds.SetProjection(ref_ds.GetProjection())
    band = out_ds.GetRasterBand(1)
    band.WriteArray(array)
    band.SetNoDataValue(-9999)
    out_ds.FlushCache()
    out_ds = None
    print(f'  Saved: {out_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def combine_suitability(year):
    os.makedirs(OUT_DIR, exist_ok=True)

    arrays   = []
    ref_ds   = None
    missing  = []

    for crop in CROPS:
        tif_path = os.path.join(SUIT_DIR, crop, f'{year}_suitability_class.tif')
        if not os.path.exists(tif_path):
            print(f'  WARNING: missing {tif_path}')
            missing.append(crop)
            continue
        arr, ds = load_tif(tif_path)
        arrays.append(arr)
        if ref_ds is None:
            ref_ds = ds

    if not arrays:
        raise RuntimeError(f'No suitability rasters found for year {year}')

    if missing:
        print(f'\n  {len(missing)} crops missing for {year}: {missing}')
        print(f'  Combining {len(arrays)}/{len(CROPS)} crops.\n')

    stack = np.stack(arrays, axis=0)   # shape: (n_crops, rows, cols)

    # Max suitability across crops (best possible use)
    combined_max  = np.nanmax(stack, axis=0)
    combined_max[np.all(np.isnan(stack), axis=0)] = -9999

    # Mean suitability across crops (average potential)
    combined_mean = np.nanmean(stack, axis=0)
    combined_mean[np.all(np.isnan(stack), axis=0)] = -9999

    # Summary stats (excluding nodata)
    valid_max  = combined_max[combined_max != -9999]
    valid_mean = combined_mean[combined_mean != -9999]

    print(f'\nYear {year} — Combined suitability summary:')
    print(f'  MAX  — mean={np.nanmean(valid_max):.3f}, '
          f'min={np.nanmin(valid_max):.1f}, max={np.nanmax(valid_max):.1f}')
    print(f'  MEAN — mean={np.nanmean(valid_mean):.3f}, '
          f'min={np.nanmin(valid_mean):.2f}, max={np.nanmax(valid_mean):.2f}')

    # Class distribution for max
    print(f'\n  Max suitability class distribution:')
    for cls in [1, 2, 3, 4, 5]:
        n = np.sum(valid_max == cls)
        pct = 100 * n / len(valid_max)
        print(f'    Class {cls}: {n} px ({pct:.1f}%)')
    suitable = np.sum(valid_max >= 2)
    print(f'    Suitable (>= Class 2): {suitable} px '
          f'({100*suitable/len(valid_max):.1f}%)')

    # Save outputs
    max_path  = os.path.join(OUT_DIR, f'combined_suitability_max_{year}.tif')
    mean_path = os.path.join(OUT_DIR, f'combined_suitability_mean_{year}.tif')
    save_tif(max_path,  combined_max,  ref_ds)
    save_tif(mean_path, combined_mean, ref_ds)

    return combined_max, combined_mean, ref_ds


if __name__ == '__main__':
    combine_suitability(YEAR)
    print('\nDone. Combined suitability rasters written to:')
    print(f'  {OUT_DIR}')