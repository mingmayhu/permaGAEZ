"""
Export Climate/Permafrost Data to GraphPad-Ready CSV
=====================================================
Exports one row per year with regional means for:
  - Mean_Temperature_C     : (TempMax + TempMin) / 2, averaged across days
  - Total_Precipitation_mm : Precip summed across days
  - Max_ALT_m              : active_layer_depth, max across days
  - Mean_SoilMoisture_mm   : avail_soil_moisture, mean across days

All values are unrounded. Output is a single CSV ready to paste into GraphPad.

File: ./results/graphpad/climate_permafrost_timeseries.csv
"""

import os
import numpy as np
import pandas as pd
from osgeo import gdal

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR  = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH = r'./data_input/qilian_mask_new.tif'
PERM_PATH = r'./data_input/permafrost_qilian.tif'
CLIM_DIR  = r'./data_input/climate_yearly'
PERM_DIR  = r'./data_input/permafrost_yearly'
OUT_DIR   = r'./results/graphpad'

YEARS = list(range(1979, 2019))

os.chdir(WORK_DIR)
os.makedirs(OUT_DIR, exist_ok=True)


# ── Load mask ─────────────────────────────────────────────────────────────────

def load_mask():
    ds   = gdal.Open(MASK_PATH)
    mask = ds.GetRasterBand(1).ReadAsArray().astype(bool)
    ds   = None

    # Exclude lake pixels (permafrost_qilian == 0 or nodata)
    ds2 = gdal.Open(PERM_PATH)
    nd  = ds2.GetRasterBand(1).GetNoDataValue()
    pf  = ds2.GetRasterBand(1).ReadAsArray().astype(float)
    ds2 = None
    if nd is not None:
        pf[pf == nd] = np.nan
    lake_mask = (pf == 0) | np.isnan(pf)
    mask = mask & ~lake_mask

    return mask


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_clim(year, filename, agg):
    """Load (rows, cols, days) climate array and collapse days axis."""
    path = os.path.join(CLIM_DIR, str(year), filename)
    arr  = np.load(path).astype(float)
    if agg == 'mean':
        return arr.mean(axis=2)
    elif agg == 'sum':
        return arr.sum(axis=2)
    elif agg == 'max':
        return arr.max(axis=2)
    else:
        raise ValueError(f"Unknown agg: {agg}")


def load_perm(year, filename, agg):
    """Load (rows, cols, days) permafrost array and collapse days axis."""
    path = os.path.join(PERM_DIR, str(year), filename)
    arr  = np.load(path).astype(float)
    if agg == 'mean':
        return arr.mean(axis=2)
    elif agg == 'max':
        return np.nanmax(arr, axis=2)
    else:
        raise ValueError(f"Unknown agg: {agg}")


def regional_mean(spatial_arr, mask):
    return float(np.nanmean(spatial_arr[mask]))


# ── Main export ───────────────────────────────────────────────────────────────

def main():
    import time
    t0 = time.time()

    print("Loading mask...")
    mask = load_mask()
    print(f"  ✓ Valid pixels: {mask.sum()}")

    rows = []
    for i, year in enumerate(YEARS, 1):
        t_year = time.time()
        print(f"\n[{i}/{len(YEARS)}] {year}...")

        # Temperature: (TempMax + TempMin) / 2, mean across days
        print(f"  Loading TempMax...", end=' ', flush=True)
        tmax = load_clim(year, 'TempMax.npy', agg='mean')
        print(f"done ({time.time()-t_year:.1f}s)")

        print(f"  Loading TempMin...", end=' ', flush=True)
        t1 = time.time()
        tmin = load_clim(year, 'TempMin.npy', agg='mean')
        print(f"done ({time.time()-t1:.1f}s)")

        tmean = (tmax + tmin) / 2.0
        mean_temp = regional_mean(tmean, mask)

        # Precipitation: sum across days
        print(f"  Loading Precip...", end=' ', flush=True)
        t1 = time.time()
        precip    = load_clim(year, 'Precip.npy', agg='sum')
        mean_prec = regional_mean(precip, mask)
        print(f"done ({time.time()-t1:.1f}s)")

        # ALT: max across days (deepest active layer reached in year)
        print(f"  Loading ALT...", end=' ', flush=True)
        t1 = time.time()
        alt      = load_perm(year, 'active_layer_depth.npy', agg='max')
        mean_alt = regional_mean(alt, mask)
        print(f"done ({time.time()-t1:.1f}s)")

        # Available soil moisture: mean across days
        print(f"  Loading SoilMoisture...", end=' ', flush=True)
        t1 = time.time()
        sm      = load_perm(year, 'avail_soil_moisture.npy', agg='mean')
        mean_sm = regional_mean(sm, mask)
        print(f"done ({time.time()-t1:.1f}s)")

        rows.append({
            'Year':                    year,
            'Mean_Temperature_C':      mean_temp,
            'Total_Precipitation_mm':  mean_prec,
            'Max_ALT_m':               mean_alt,
            'Mean_SoilMoisture_mm':    mean_sm,
        })
        print(f"  → T={mean_temp:.4f}°C  P={mean_prec:.2f}mm  "
              f"ALT={mean_alt:.4f}m  SM={mean_sm:.4f}mm  "
              f"(year total: {time.time()-t_year:.1f}s)")

    df = pd.DataFrame(rows)

    out_path = os.path.join(OUT_DIR, 'climate_permafrost_timeseries.csv')
    df.to_csv(out_path, index=False)
    print(f"\n✓ Saved to: {out_path}")

    print("\n--- GraphPad paste preview ---")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()