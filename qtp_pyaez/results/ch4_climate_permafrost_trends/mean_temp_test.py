"""
sens_slope_temp.py
------------------
Computes Sen's slope for regional mean annual temperature across valid
Qilian pixels, 1979–2018.

Data:
  TempMean: (rows, cols, days) per year → mean across days → annual mean
  Regional mean: nanmean across valid pixels per year
  Sen's slope: pymannkendall on the 40-year regional mean time series

Output: printed to console.
"""

import os
import numpy as np
import pymannkendall as mk
from osgeo import gdal

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = "/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez"

MASK_PATH      = os.path.join(BASE, "data_input/qilian_mask_new.tif")
PERM_MASK_PATH = os.path.join(BASE, "data_input/permafrost_qilian.tif")
CLIMATE_DIR    = os.path.join(BASE, "data_input/climate_yearly")

YEARS = list(range(1979, 2019))  # 1979–2018 inclusive

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_tif(path):
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Cannot open: {path}")
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    if nodata is not None:
        arr[arr == nodata] = np.nan
    ds = None
    return arr


def build_valid_mask(mask_path, perm_mask_path):
    mask_arr = read_tif(mask_path)
    perm_arr = read_tif(perm_mask_path)
    return (mask_arr > 0) & (~np.isnan(mask_arr)) & (perm_arr != 0) & (~np.isnan(perm_arr))

# ---------------------------------------------------------------------------
# Load mask
# ---------------------------------------------------------------------------
print("Loading mask …")
valid = build_valid_mask(MASK_PATH, PERM_MASK_PATH)
print(f"  Valid pixels: {valid.sum()}")

# ---------------------------------------------------------------------------
# Compute annual regional mean temperature
# ---------------------------------------------------------------------------
print("\nComputing annual regional mean temperature 1979–2018 …")
regional_mean = []

for yr in YEARS:
    path = os.path.join(CLIMATE_DIR, str(yr), "TempMean.npy")
    arr  = np.load(path)                        # (rows, cols, days)
    annual = np.nanmean(arr, axis=2)            # mean across days → (rows, cols)
    region = np.nanmean(annual[valid])          # mean across valid pixels
    regional_mean.append(region)
    print(f"  {yr}: {region:.3f} °C")

regional_mean = np.array(regional_mean)

# ---------------------------------------------------------------------------
# Sen's slope + Mann-Kendall test
# ---------------------------------------------------------------------------
print("\nComputing Sen's slope …")
result = mk.original_test(regional_mean)

# Bootstrap 95% CI on Sen's slope (1000 resamples, seed 42)
rng = np.random.default_rng(42)
n   = len(regional_mean)
slopes_boot = []
for _ in range(1000):
    idx      = np.sort(rng.choice(n, size=n, replace=True))
    y_boot   = regional_mean[idx]
    x_boot   = np.arange(n)[idx]
    pairs    = []
    for i in range(len(x_boot)):
        for j in range(i + 1, len(x_boot)):
            dx = x_boot[j] - x_boot[i]
            if dx != 0:
                pairs.append((y_boot[j] - y_boot[i]) / dx)
    if pairs:
        slopes_boot.append(np.median(pairs))

slopes_boot = np.array(slopes_boot)
ci_lo = np.percentile(slopes_boot, 2.5)
ci_hi = np.percentile(slopes_boot, 97.5)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("REGIONAL MEAN TEMPERATURE TREND (1979–2018)")
print("="*60)
print(f"  Mean temperature (full period) : {regional_mean.mean():.3f} °C")
print(f"  Min annual mean                : {regional_mean.min():.3f} °C ({YEARS[regional_mean.argmin()]})")
print(f"  Max annual mean                : {regional_mean.max():.3f} °C ({YEARS[regional_mean.argmax()]})")
print(f"\n  Sen's slope : {result.slope:.4f} °C yr⁻¹")
print(f"  95% CI      : {ci_lo:.4f} – {ci_hi:.4f} °C yr⁻¹")
print(f"  Kendall's τ : {result.Tau:.4f}")
print(f"  p-value     : {result.p:.4e}")
print(f"  Trend       : {result.trend}")
print("="*60)
print("\nDone.")