"""
Fixed-Boundary Suitability Reclassification
============================================
Computes per-crop global yield boundaries across ALL years and BOTH scenarios,
then reclassifies all raw yield rasters using those fixed boundaries.

This ensures suitability classes are comparable:
  - Across years (temporal trend analysis)
  - Between observed and counterfactual (impact analysis)

Class definitions (same as PyAEZ, but with fixed boundaries):
  0 = no yield
  1 = not suitable       (0%–20% of global max-min range)
  2 = marginally suitable(20%–40%)
  3 = moderately suitable(40%–60%)
  4 = suitable           (60%–80%)
  5 = very suitable      (>80%)

Outputs written to:
  ./data_output/final_classification_fixed/{tag}/{year}_suitability_class.tif
  ./data_output/final_classification_nothaw_fixed/{tag}/{year}_suitability_class.tif
  ./results_analysis/outputs/0_reclassification/boundaries.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from osgeo import gdal

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR  = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH = r'./data_input/qilian mask.tif'
# OUT_ROOT  = './results_analysis/outputs/0_reclassification'
OUT_ROOT  = './results/permafrost_thaw_impact/permafrost_vs_fao/outputs/0_reclassification'

YEARS_OBS = list(range(1979, 2019))   # observed: full 40 years
YEARS_CF  = list(range(1999, 2019))   # counterfactual: 1999-2018 only

CROPS = [
    # {'label': 'Winter Barley', 'tag': 'combined_winter_barley'},
    # {'label': 'Spring Barley', 'tag': 'combined_spring_barley'},
    # {'label': 'Winter Wheat',  'tag': 'combined_winter_wheat'},
    # {'label': 'Spring Wheat',  'tag': 'combined_spring_wheat'},
    # {'label': 'Silage Maize',  'tag': 'combined_silage_maize'},
    # {'label': 'White Potato',  'tag': 'combined_white_potato'},
    # {'label': 'Oat',           'tag': 'combined_oat'},
    # {'label': 'Dry Pea',       'tag': 'combined_dry_pea'},
    # {'label': 'Winter Rape',   'tag': 'combined_winter_rape'},
    # {'label': 'Spring Rape',   'tag': 'combined_spring_rape'},
    {'label': 'Barley',   'tag': 'combined_barley'},
    {'label': 'Rape',   'tag': 'combined_rape'},
    {'label': 'Wheat',   'tag': 'combined_wheat'},
]

os.chdir(WORK_DIR)
os.makedirs(OUT_ROOT, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_raster(path):
    ds = gdal.Open(path)
    if ds is None:
        return None, None
    band   = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    arr    = ds.ReadAsArray().astype(float)
    arr[arr < -1e10] = np.nan
    if nodata is not None:
        arr[arr == nodata] = np.nan
    geo_info = (ds.GetGeoTransform(), ds.GetProjection(),
                ds.RasterXSize, ds.RasterYSize)
    return arr, geo_info

def save_raster(path, arr, geo_info):
    geo, proj, nx, ny = geo_info
    driver = gdal.GetDriverByName('GTiff')
    ds_out = driver.Create(path, nx, ny, 1, gdal.GDT_Float32)
    ds_out.SetGeoTransform(geo)
    ds_out.SetProjection(proj)
    band = ds_out.GetRasterBand(1)
    band.WriteArray(arr.astype(np.float32))
    band.SetNoDataValue(-9999.0)
    ds_out.FlushCache()

def load_mask():
    arr, _ = load_raster(MASK_PATH)
    return arr.astype(bool)

def obs_path(tag, year):
    return f'./data_output/final_classification/{tag}/{year}_raw_yield.tif'

def cf_path(tag, year):
    return f'./data_output/final_classification_nothaw/{tag}/{year}_raw_yield.tif'

def classify_fixed(arr, y_min, y_max, mask):
    """
    Classify yield array using fixed global boundaries.
    Pixels with yield <= 0 or outside mask = class 0.
    """
    out = np.zeros(arr.shape, dtype=float)
    out[~mask] = np.nan

    if y_max <= y_min:
        return out   # degenerate case — all zeros

    rng = y_max - y_min
    b20 = y_min + 0.20 * rng
    b40 = y_min + 0.40 * rng
    b60 = y_min + 0.60 * rng
    b80 = y_min + 0.80 * rng

    valid = mask & np.isfinite(arr) & (arr > 0)
    out[valid & (arr <= b20)] = 1
    out[valid & (arr > b20) & (arr <= b40)] = 2
    out[valid & (arr > b40) & (arr <= b60)] = 3
    out[valid & (arr > b60) & (arr <= b80)] = 4
    out[valid & (arr > b80)] = 5

    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def run(mask):
    boundary_records = []

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        print(f'\n── {label} ──')

        # ── Step 1: collect all valid yield values across both scenarios ───────
        all_vals = []

        for year in YEARS_OBS:
            arr, _ = load_raster(obs_path(tag, year))
            if arr is not None:
                valid = arr[mask & np.isfinite(arr) & (arr > 0)]
                all_vals.extend(valid.tolist())

        for year in YEARS_CF:
            arr, _ = load_raster(cf_path(tag, year))
            if arr is not None:
                valid = arr[mask & np.isfinite(arr) & (arr > 0)]
                all_vals.extend(valid.tolist())

        if not all_vals:
            print(f'  ⚠ No valid yield data for {label} — skipping.')
            continue

        all_vals = np.array(all_vals)
        y_min = float(np.min(all_vals))
        y_max = float(np.max(all_vals))

        boundary_records.append({
            'crop'   : label,
            'tag'    : tag,
            'y_min'  : round(y_min, 4),
            'y_max'  : round(y_max, 4),
            'b20'    : round(y_min + 0.20 * (y_max - y_min), 4),
            'b40'    : round(y_min + 0.40 * (y_max - y_min), 4),
            'b60'    : round(y_min + 0.60 * (y_max - y_min), 4),
            'b80'    : round(y_min + 0.80 * (y_max - y_min), 4),
            'n_vals' : len(all_vals),
        })
        print(f'  Boundaries: min={y_min:.2f}, max={y_max:.2f} kg/ha '
              f'(n={len(all_vals):,} pixel-years)')

        # ── Step 2: reclassify observed rasters ───────────────────────────────
        out_obs_dir = f'./data_output/final_classification_fixed/{tag}'
        os.makedirs(out_obs_dir, exist_ok=True)

        for year in YEARS_OBS:
            arr, geo_info = load_raster(obs_path(tag, year))
            if arr is None:
                print(f'  ⚠ Missing observed {year}')
                continue
            cls = classify_fixed(arr, y_min, y_max, mask)
            out_arr = np.where(np.isfinite(cls), cls, -9999.0)
            save_raster(f'{out_obs_dir}/{year}_suitability_class.tif', out_arr, geo_info)

        print(f'  ✓ Observed rasters reclassified ({len(YEARS_OBS)} years)')

        # # ── Step 3: reclassify counterfactual rasters ─────────────────────────
        out_cf_dir = f'./data_output/final_classification_nothaw_fixed/{tag}'
        os.makedirs(out_cf_dir, exist_ok=True)

        for year in YEARS_CF:
            arr, geo_info = load_raster(cf_path(tag, year))
            if arr is None:
                print(f'  ⚠ Missing counterfactual {year}')
                continue
            cls = classify_fixed(arr, y_min, y_max, mask)
            out_arr = np.where(np.isfinite(cls), cls, -9999.0)
            save_raster(f'{out_cf_dir}/{year}_suitability_class.tif', out_arr, geo_info)

        print(f'  ✓ Counterfactual rasters reclassified ({len(YEARS_CF)} years)')

        # ── Step 4: quick visual check — mean class map for both scenarios ─────
        obs_stack = []
        for year in YEARS_OBS:
            path = f'{out_obs_dir}/{year}_suitability_class.tif'
            arr, _ = load_raster(path)
            if arr is not None:
                arr[arr < 0] = np.nan
                obs_stack.append(arr)

        cf_stack = []
        for year in YEARS_CF:
            path = f'{out_cf_dir}/{year}_suitability_class.tif'
            arr, _ = load_raster(path)
            if arr is not None:
                arr[arr < 0] = np.nan
                cf_stack.append(arr)

        if obs_stack and cf_stack:
            mean_obs = np.nanmean(obs_stack, axis=0)
            mean_cf  = np.nanmean(cf_stack,  axis=0)
            # Use only 1999-2018 observed for fair comparison with CF
            obs_stack_post = obs_stack[20:]   # years 1999-2018
            mean_obs_post  = np.nanmean(obs_stack_post, axis=0)
            delta          = mean_obs_post - mean_cf

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            cmap_suit = plt.get_cmap('RdYlGn', 6)

            im0 = axes[0].imshow(mean_obs_post, cmap=cmap_suit, vmin=0, vmax=5)
            axes[0].set_title('Mean Suitability — Observed\n(1999–2018)', fontsize=11)
            axes[0].axis('off')
            plt.colorbar(im0, ax=axes[0], shrink=0.75,
                         ticks=[0,1,2,3,4,5], label='Class')

            im1 = axes[1].imshow(mean_cf, cmap=cmap_suit, vmin=0, vmax=5)
            axes[1].set_title('Mean Suitability — No-Thaw CF\n(1999–2018)', fontsize=11)
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], shrink=0.75,
                         ticks=[0,1,2,3,4,5], label='Class')

            vlim = max(abs(np.nanpercentile(delta[mask], 2)),
                       abs(np.nanpercentile(delta[mask], 98)))
            im2 = axes[2].imshow(delta, cmap='RdBu', vmin=-vlim, vmax=vlim)
            axes[2].set_title('ΔSuitability (Obs − CF)\n(Mean 1999–2018)', fontsize=11)
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], shrink=0.75, label='Δ Class')

            fig.suptitle(f'{label} — Fixed-Boundary Suitability Classes',
                         fontsize=13, fontweight='bold')
            plt.tight_layout()
            fig.savefig(f'{OUT_ROOT}/{tag}_suitability_check.png',
                        dpi=150, bbox_inches='tight')
            plt.close()

    # ── Save boundary table ────────────────────────────────────────────────────
    df = pd.DataFrame(boundary_records)
    df.to_csv(f'{OUT_ROOT}/boundaries.csv', index=False)
    print(f'\n✓ Boundary table saved to: {OUT_ROOT}/boundaries.csv')
    print(df[['crop', 'y_min', 'y_max', 'b20', 'b40', 'b60', 'b80']].to_string(index=False))
    print(f'\n✓ All reclassified rasters saved.')
    print(f'   Observed:        ./data_output/final_classification_fixed/{{tag}}/')
    print(f'   Counterfactual:  ./data_output/final_classification_nothaw_fixed/{{tag}}/')


if __name__ == '__main__':
    mask = load_mask()
    run(mask)