"""
PCF Risk Exposure Table
Computes current agricultural land area (km²) by PCF risk class.

Two metrics:
  1. Cropland in risk zone = cropland_fraction × pixel_area
     (how much cropland falls within each risk zone)
  2. Cropland physically exposed = cropland_fraction × A_adj_km2
     (how much cropland would actually be physically disturbed)

Risk classes (from proposal):
  None:   risk == 0
  Low:    0 < risk < 50
  Medium: 50 <= risk <= 80
  High:   risk > 80
"""

import numpy as np
from osgeo import gdal
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

CROPLAND_PATH = '/Users/ming-mayhu/Desktop/毕业论文/pcf/avg_agricultural_land_1km.tif'
RISK_PATH     = '/Users/ming-mayhu/Desktop/毕业论文/pcf/rts_qilian_risk.tif'
AREA_PATH     = '/Users/ming-mayhu/Desktop/毕业论文/pcf/rts_qilian_area.tif'

# Minimum cropland fraction to count a pixel as agricultural
CROPLAND_THRESHOLD = 0.05

# Pixel area in km² at ~38°N for 0.01° resolution
PIXEL_AREA_KM2 = (0.01 * 111.132) * (0.01 * 111.132 * np.cos(np.radians(38)))

# Output CSV
OUT_CSV = '/Users/ming-mayhu/Desktop/毕业论文/pcf/pcf_risk_table.csv'

# ── Helpers ───────────────────────────────────────────────────────────────────

def load(path):
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(f'Cannot open: {path}')
    band = ds.GetRasterBand(1)
    arr  = band.ReadAsArray().astype(np.float32)
    nd   = band.GetNoDataValue()
    if nd is not None:
        arr[arr == nd] = np.nan
    arr[arr < -1e10] = np.nan
    return arr

# ── Load ──────────────────────────────────────────────────────────────────────

print('Loading rasters...')
cropland = load(CROPLAND_PATH)
risk     = load(RISK_PATH)
area_m2  = load(AREA_PATH)

# Convert area from m² to km²
area_km2 = area_m2 / 1_000_000

if not (cropland.shape == risk.shape == area_km2.shape):
    raise ValueError('Raster shapes do not match — resample to same grid first')

# ── Masks ─────────────────────────────────────────────────────────────────────

valid   = np.isfinite(cropland) & np.isfinite(risk) & np.isfinite(area_km2)
ag_mask = valid & (cropland >= CROPLAND_THRESHOLD)

# Metric 1: cropland area in each pixel (km²)
cropland_area = cropland * PIXEL_AREA_KM2

# Metric 2: cropland physically exposed (km²)
# cropland_fraction × A_adj — A_adj already incorporates susceptibility
physically_exposed = cropland * area_km2

# Totals
total_cropland  = float(cropland_area[ag_mask].sum())
total_exposed   = float(physically_exposed[ag_mask].sum())

print(f'\nPixel area at ~38N:       {PIXEL_AREA_KM2:.4f} km²')
print(f'Total valid pixels:       {valid.sum():,}')
print(f'Agricultural pixels:      {ag_mask.sum():,}')
print(f'Total cropland area:      {total_cropland:.2f} km²')
print(f'Total physically exposed: {total_exposed:.2f} km² (±34%)')

# ── Risk class masks ──────────────────────────────────────────────────────────

risk_classes = {
    'None (0%)':       (risk == 0),
    'Low (0-50%)':     (risk > 0)   & (risk < 50),
    'Medium (50-80%)': (risk >= 50) & (risk <= 80),
    'High (>80%)':     (risk > 80),
}

# ── Compute table ─────────────────────────────────────────────────────────────

rows = []
for class_name, risk_class_mask in risk_classes.items():
    mask = ag_mask & risk_class_mask

    cl_area  = float(cropland_area[mask].sum())
    ex_area  = float(physically_exposed[mask].sum())
    cl_pct   = 100 * cl_area / total_cropland if total_cropland > 0 else 0
    ex_pct   = 100 * ex_area / total_exposed  if total_exposed  > 0 else 0

    rows.append({
        'Risk class':                      class_name,
        'Cropland in zone (km²)':          round(cl_area, 2),
        'Cropland in zone (%)':            round(cl_pct, 1),
        'Cropland physically exposed (km²)': round(ex_area, 2),
        'Physically exposed (%)':          round(ex_pct, 1),
    })

# Total row
rows.append({
    'Risk class':                        'Total',
    'Cropland in zone (km²)':            round(total_cropland, 2),
    'Cropland in zone (%)':              100.0,
    'Cropland physically exposed (km²)': round(total_exposed, 2),
    'Physically exposed (%)':            100.0,
})

df = pd.DataFrame(rows)

print('\n── PCF Risk Exposure Table ──────────────────────────────────────────')
print(df.to_string(index=False))

df.to_csv(OUT_CSV, index=False)
print(f'\nSaved to: {OUT_CSV}')