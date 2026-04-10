"""
Step 1 — Mean ΔSuitability Map (1999–2018)
==========================================
For each pixel, computes the mean ΔSuitability (observed − counterfactual)
across all 20 years (1999–2018).

This is the spatial version of the Wilcoxon result — instead of a single
regional mean, you see the full geographic pattern of where thaw is
consistently helping vs hurting suitability.

Outputs per crop:
  - mean_delta_suit.tif     — mean ΔSuitability raster
  - mean_delta_suit.png     — map figure
  - obs_mean_suit.tif       — mean observed suitability
  - cf_mean_suit.tif        — mean counterfactual suitability

Outputs overall:
  - ALL_CROPS_mean_delta.png — summary panel across all crops
  - ALL_CROPS_mean_delta.tif — aggregate mean ΔSuitability across crops

Outputs written to: ./results_analysis/outputs/6_spatial_analysis/1_mean_delta/
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
OUT_ROOT  = './results_analysis/outputs/6_spatial_analysis/1_mean_delta'

YEARS_CF = list(range(1999, 2019))

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

DIVERGING_CMAP = 'RdBu'

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

def obs_suit_path(tag, year):
    return f'./data_output/final_classification_fixed/{tag}/{year}_suitability_class.tif'

def cf_suit_path(tag, year):
    return f'./data_output/final_classification_nothaw_fixed/{tag}/{year}_suitability_class.tif'

def clean(arr, mask):
    """Set nodata and outside-mask pixels to NaN."""
    out = arr.copy()
    out[~mask] = np.nan
    out[out < 0] = np.nan
    return out

def plot_delta_map(ax, arr, mask, title, vlim, cmap='RdBu'):
    """Plot a ΔSuitability map on a given axis."""
    display = np.where(mask, arr, np.nan)
    im = ax.imshow(display, cmap=cmap, vmin=-vlim, vmax=vlim)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')
    return im

def plot_suit_map(ax, arr, mask, title, cmap='RdYlGn'):
    """Plot a suitability class map on a given axis."""
    display = np.where(mask, arr, np.nan)
    im = ax.imshow(display, cmap=plt.get_cmap(cmap, 6), vmin=0, vmax=5)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')
    return im


# ── Main ──────────────────────────────────────────────────────────────────────

def run(mask):
    summary_deltas = {}   # crop label -> mean delta array (for summary panel)
    geo_info_ref   = None # reference geo info for saving aggregate raster

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        print(f'\n── {label} ──')

        obs_stack = []
        cf_stack  = []

        for year in YEARS_CF:
            obs, geo_info = load_raster(obs_suit_path(tag, year))
            cf,  _        = load_raster(cf_suit_path(tag, year))

            if obs is None or cf is None:
                print(f'  ⚠ missing {year}')
                continue

            obs = clean(obs, mask)
            cf  = clean(cf,  mask)

            obs_stack.append(obs)
            cf_stack.append(cf)

            if geo_info_ref is None:
                geo_info_ref = geo_info

        if not obs_stack or not cf_stack:
            print(f'  ⚠ No data for {label}')
            continue

        obs_stack = np.array(obs_stack)   # (n_years, rows, cols)
        cf_stack  = np.array(cf_stack)

        # ── Compute mean suitability and mean ΔSuitability ────────────────────
        mean_obs   = np.nanmean(obs_stack, axis=0)
        mean_cf    = np.nanmean(cf_stack,  axis=0)

        # ΔSuitability — only where both have valid data
        both_valid = np.isfinite(mean_obs) & np.isfinite(mean_cf)
        mean_delta = np.where(both_valid, mean_obs - mean_cf, np.nan)
        mean_delta[~mask] = np.nan

        summary_deltas[label] = mean_delta

        # ── Quick stats ───────────────────────────────────────────────────────
        valid_delta = mean_delta[mask & np.isfinite(mean_delta)]
        if len(valid_delta) > 0:
            print(f'  Mean ΔSuit: {valid_delta.mean():.4f}')
            print(f'  % pixels positive: {(valid_delta > 0).mean()*100:.1f}%')
            print(f'  % pixels negative: {(valid_delta < 0).mean()*100:.1f}%')
            print(f'  % pixels zero:     {(valid_delta == 0).mean()*100:.1f}%')

        # ── Save rasters ──────────────────────────────────────────────────────
        for arr, fname in [
            (mean_obs,   f'{OUT_ROOT}/{tag}_obs_mean_suit.tif'),
            (mean_cf,    f'{OUT_ROOT}/{tag}_cf_mean_suit.tif'),
            (mean_delta, f'{OUT_ROOT}/{tag}_mean_delta_suit.tif'),
        ]:
            out = np.where(np.isfinite(arr), arr, -9999.0)
            save_raster(fname, out, geo_info_ref)

        # ── Per-crop figure: obs | cf | delta ─────────────────────────────────
        vlim = np.nanpercentile(np.abs(valid_delta), 98) if len(valid_delta) > 0 else 0.5
        if vlim == 0 or np.isnan(vlim):
            vlim = 0.5

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        im0 = plot_suit_map(axes[0], mean_obs, mask,
                            'Mean Suitability — Observed\n(1999–2018)')
        plt.colorbar(im0, ax=axes[0], shrink=0.75,
                     ticks=[0,1,2,3,4,5], label='Class')

        im1 = plot_suit_map(axes[1], mean_cf, mask,
                            'Mean Suitability — No-Thaw CF\n(1999–2018)')
        plt.colorbar(im1, ax=axes[1], shrink=0.75,
                     ticks=[0,1,2,3,4,5], label='Class')

        im2 = plot_delta_map(axes[2], mean_delta, mask,
                             f'Mean ΔSuitability (Obs − CF)\n(1999–2018)',
                             vlim)
        plt.colorbar(im2, ax=axes[2], shrink=0.75, label='Δ Class')

        fig.suptitle(f'{label} — Mean Suitability & Thaw Impact (1999–2018)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        fig.savefig(f'{OUT_ROOT}/{tag}_mean_delta_suit.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  ✓ saved')

    # ── Summary panel: all crops ──────────────────────────────────────────────
    n     = len(summary_deltas)
    ncols = 5
    nrows = -(-n // ncols)

    # Global colour scale across all crops
    all_vals = np.concatenate([
        d[mask & np.isfinite(d)] for d in summary_deltas.values()
    ])
    global_vlim = np.nanpercentile(np.abs(all_vals), 98)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4, nrows * 3.5))
    axes = axes.flatten()

    for i, (label, delta) in enumerate(summary_deltas.items()):
        im = axes[i].imshow(
            np.where(mask, delta, np.nan),
            cmap=DIVERGING_CMAP,
            vmin=-global_vlim, vmax=global_vlim
        )
        axes[i].set_title(label, fontsize=10, fontweight='bold')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], shrink=0.8, label='Δ Class')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(
        'Mean ΔSuitability (Observed − No-Thaw), 1999–2018\nAll Crop Types',
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    fig.savefig(f'{OUT_ROOT}/ALL_CROPS_mean_delta_suit.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('\n✓ All-crop summary panel saved.')

    # ── Aggregate mean ΔSuitability across all crops ───────────────────────────
    # Average ΔSuitability across all crops per pixel
    # Suitability is on same scale so direct average is valid
    delta_stack = np.array([
        d for d in summary_deltas.values()
    ])
    agg_delta = np.nanmean(delta_stack, axis=0)
    agg_delta[~mask] = np.nan

    # Save aggregate raster
    out = np.where(np.isfinite(agg_delta), agg_delta, -9999.0)
    save_raster(f'{OUT_ROOT}/ALL_CROPS_mean_delta_suit.tif', out, geo_info_ref)

    # Plot aggregate
    agg_vlim = np.nanpercentile(
        np.abs(agg_delta[mask & np.isfinite(agg_delta)]), 98
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(np.where(mask, agg_delta, np.nan),
                   cmap=DIVERGING_CMAP,
                   vmin=-agg_vlim, vmax=agg_vlim)
    ax.axis('off')
    ax.set_title(
        'Aggregate Mean ΔSuitability — All Crops\n(Observed − No-Thaw, 1999–2018)',
        fontsize=13, fontweight='bold'
    )
    plt.colorbar(im, ax=ax, shrink=0.75, label='Mean Δ Class (avg across crops)')
    plt.tight_layout()
    fig.savefig(f'{OUT_ROOT}/ALL_CROPS_agg_mean_delta_suit.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Summary stats table ───────────────────────────────────────────────────
    records = []
    for label, delta in summary_deltas.items():
        vals = delta[mask & np.isfinite(delta)]
        records.append({
            'crop'            : label,
            'mean_delta'      : round(float(vals.mean()), 5),
            'median_delta'    : round(float(np.median(vals)), 5),
            'pct_positive'    : round(float((vals > 0).mean() * 100), 2),
            'pct_negative'    : round(float((vals < 0).mean() * 100), 2),
            'pct_zero'        : round(float((vals == 0).mean() * 100), 2),
            'n_pixels'        : len(vals),
        })

    df = pd.DataFrame(records)
    df.to_csv(f'{OUT_ROOT}/mean_delta_suit_stats.csv', index=False)
    print('\nSummary stats:')
    print(df.to_string(index=False))
    print(f'\n✓ All outputs saved to: {OUT_ROOT}/')


if __name__ == '__main__':
    mask = load_mask()
    run(mask)