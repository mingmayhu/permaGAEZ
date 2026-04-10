"""
Step 5 — Permafrost Variable Correlation with ΔSuitability
===========================================================
Correlates mean ΔSuitability (per pixel, 1999–2018) with:
  - Active layer depth (ALT)
  - Available soil moisture

For each variable computes:
  A) Mean value across 1999–2018
  B) Change = mean(1999–2018) − mean(1979–1998)

Then runs:
  1. Pixel-wise spatial correlation (Spearman) between
     permafrost variable and mean ΔSuitability across pixels
  2. Temporal correlation (Spearman) between annual regional
     mean permafrost variable and annual regional mean ΔSuitability

Outputs per crop and overall:
  - scatter plots of pixel-wise correlation
  - temporal correlation line plots
  - correlation_results.csv

Outputs written to:
  ./results_analysis/outputs/6_spatial_analysis/5_permafrost_corr/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from osgeo import gdal

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR   = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH  = r'./data_input/qilian mask.tif'
PERM_DIR   = r'./data_input/permafrost_yearly'
DELTA_DIR  = r'./results_analysis/outputs/6_spatial_analysis/1_mean_delta'
OUT_ROOT   = r'./results_analysis/outputs/6_spatial_analysis/5_permafrost_corr'

YEARS_ALL  = list(range(1979, 2019))
YEARS_OBS  = list(range(1979, 1999))   # pre-divergence
YEARS_CF   = list(range(1999, 2019))   # post-divergence

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

PERM_VARS = [
    {'name': 'ALT',           'file': 'active_layer_depth.npy',  'label': 'Active Layer Depth (m)',    'agg': 'max'},
    {'name': 'soil_moisture', 'file': 'avail_soil_moisture.npy', 'label': 'Available Soil Moisture',   'agg': 'mean'},
]

os.chdir(WORK_DIR)
os.makedirs(OUT_ROOT, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_raster(path):
    ds = gdal.Open(path)
    if ds is None:
        return None
    band   = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    arr    = ds.ReadAsArray().astype(float)
    arr[arr < -1e10] = np.nan
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr

def load_mask():
    arr = load_raster(MASK_PATH)
    return arr.astype(bool)

def load_perm_stack(var_file, years, mask, agg='mean'):
    """
    Load a permafrost variable for a list of years.
    Arrays are (rows, cols, days) — aggregate across days first.
    agg='max'  for ALT (deepest active layer reached in year)
    agg='mean' for soil moisture (mean across year)
    Returns (n_years, rows, cols).
    """
    stack = []
    target_shape = mask.shape

    for year in years:
        path = f'{PERM_DIR}/{year}/{var_file}'
        if not os.path.exists(path):
            print(f'  ⚠ Missing: {path}')
            stack.append(np.full(target_shape, np.nan))
            continue

        arr = np.load(path).astype(float)   # (rows, cols, days)

        # Aggregate across day dimension
        if arr.ndim == 3:
            if agg == 'max':
                arr = np.nanmax(arr, axis=2)
            else:
                arr = np.nanmean(arr, axis=2)

        if arr.shape != target_shape:
            print(f'  ⚠ Shape mismatch year {year}: {arr.shape} vs {target_shape}')
            stack.append(np.full(target_shape, np.nan))
            continue

        arr[~mask] = np.nan
        stack.append(arr)

    return np.array(stack)

def regional_mean(arr, mask):
    """Mean over valid mask pixels."""
    valid = mask & np.isfinite(arr)
    return float(np.nanmean(arr[valid])) if valid.any() else np.nan

def load_delta_raster(tag, mask):
    """Load mean ΔSuitability raster for a crop."""
    path = f'{DELTA_DIR}/{tag}_mean_delta_suit.tif'
    arr  = load_raster(path)
    if arr is None:
        return None
    arr[~mask] = np.nan
    return arr

def suit_path(tag, year):
    return f'./data_output/final_classification_fixed/{tag}/{year}_suitability_class.tif'

def cf_suit_path(tag, year):
    return f'./data_output/final_classification_nothaw_fixed/{tag}/{year}_suitability_class.tif'


# ── Main ──────────────────────────────────────────────────────────────────────

def run(mask):
    rows, cols   = mask.shape
    all_results  = []

    # ── Load permafrost stacks ─────────────────────────────────────────────────
    print('Loading permafrost data …')
    perm_data = {}
    for pv in PERM_VARS:
        stack_all  = load_perm_stack(pv['file'], YEARS_ALL, mask, agg=pv['agg'])
        stack_pre  = stack_all[:len(YEARS_OBS)]    # 1979–1998
        stack_post = stack_all[len(YEARS_OBS):]    # 1999–2018

        mean_all  = np.nanmean(stack_all,  axis=0)
        mean_pre  = np.nanmean(stack_pre,  axis=0)
        mean_post = np.nanmean(stack_post, axis=0)
        change    = mean_post - mean_pre             # Δ = post minus pre

        mean_all[~mask]  = np.nan
        mean_post[~mask] = np.nan
        change[~mask]    = np.nan

        perm_data[pv['name']] = {
            'stack_post' : stack_post,   # (20, rows, cols) for temporal analysis
            'mean_post'  : mean_post,    # mean 1999–2018 for spatial correlation
            'change'     : change,       # change from pre to post for spatial corr
            'label'      : pv['label'],
        }

        print(f'  ✓ {pv["name"]}: mean_post range '
              f'[{np.nanmin(mean_post):.3f}, {np.nanmax(mean_post):.3f}], '
              f'change range [{np.nanmin(change):.3f}, {np.nanmax(change):.3f}]')

    # ── Load aggregate mean ΔSuitability (across all crops) ───────────────────
    print('\nLoading ΔSuitability rasters …')
    delta_arrays = {}
    for crop in CROPS:
        delta = load_delta_raster(crop['tag'], mask)
        if delta is not None:
            delta_arrays[crop['label']] = delta

    # Aggregate across all crops
    agg_delta = np.nanmean(np.array(list(delta_arrays.values())), axis=0)
    agg_delta[~mask] = np.nan
    delta_arrays['OVERALL'] = agg_delta

    # ── Analysis A: Pixel-wise spatial correlation ─────────────────────────────
    print('\n[A] Pixel-wise spatial correlation …')
    out_dir_a = f'{OUT_ROOT}/A_spatial'
    os.makedirs(out_dir_a, exist_ok=True)

    for pv in PERM_VARS:
        pname = pv['name']
        plabel = perm_data[pname]['label']

        for metric, perm_arr, metric_label in [
            ('mean',   perm_data[pname]['mean_post'], f'Mean {plabel} (1999–2018)'),
            ('change', perm_data[pname]['change'],    f'Δ{plabel} (1999–2018 minus 1979–1998)'),
        ]:
            fig, axes = plt.subplots(3, 4, figsize=(20, 15))
            axes = axes.flatten()

            for i_crop, (crop_label, delta) in enumerate(delta_arrays.items()):
                ax = axes[i_crop]

                # Get valid pixels where both are finite
                valid = mask & np.isfinite(delta) & np.isfinite(perm_arr)
                x = perm_arr[valid]
                y = delta[valid]

                if len(x) < 5:
                    ax.set_title(f'{crop_label}\n(insufficient data)', fontsize=9)
                    ax.axis('off')
                    continue

                r, p = spearmanr(x, y)
                sig  = '★' if p < 0.05 else ''

                ax.scatter(x, y, alpha=0.4, s=8, color='#2166AC')
                ax.axhline(0, color='black', linewidth=0.6, linestyle='--')
                ax.axvline(np.nanmean(x), color='grey',
                           linewidth=0.6, linestyle=':')
                ax.set_xlabel(metric_label, fontsize=8)
                ax.set_ylabel('Mean ΔSuitability', fontsize=8)
                ax.set_title(f'{crop_label}\nρ={r:.3f}, p={p:.3f}{sig}',
                             fontsize=9, fontweight='bold')

                all_results.append({
                    'analysis'  : 'spatial',
                    'crop'      : crop_label,
                    'perm_var'  : pname,
                    'metric'    : metric,
                    'spearman_r': round(r, 4),
                    'p_value'   : round(p, 4),
                    'significant': p < 0.05,
                    'n_pixels'  : int(valid.sum()),
                })

            for j in range(i_crop + 1, len(axes)):
                axes[j].axis('off')

            fig.suptitle(
                f'Pixel-wise Spatial Correlation: {metric_label} vs Mean ΔSuitability\n'
                f'★ = significant at p < 0.05',
                fontsize=13, fontweight='bold'
            )
            plt.tight_layout()
            fig.savefig(f'{out_dir_a}/{pname}_{metric}_spatial_corr.png',
                        dpi=150, bbox_inches='tight')
            plt.close()
            print(f'  ✓ {pname} {metric} spatial correlation saved')

    # ── Analysis B: Temporal correlation ──────────────────────────────────────
    print('\n[B] Temporal correlation …')
    out_dir_b = f'{OUT_ROOT}/B_temporal'
    os.makedirs(out_dir_b, exist_ok=True)

    # Build annual regional mean ΔSuitability series per crop
    print('  Building annual ΔSuitability series …')
    annual_delta = {}   # crop_label -> array of 20 values (1999–2018)

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        series = []
        for year in YEARS_CF:
            obs = load_raster(suit_path(tag, year))
            cf  = load_raster(cf_suit_path(tag, year))
            if obs is None or cf is None:
                series.append(np.nan)
                continue
            obs[~mask] = np.nan
            cf[~mask]  = np.nan
            obs[obs < 0] = np.nan
            cf[cf < 0]   = np.nan
            delta = np.where(np.isfinite(obs) & np.isfinite(cf), obs - cf, np.nan)
            valid = mask & np.isfinite(delta)
            series.append(float(np.nanmean(delta[valid])) if valid.any() else np.nan)
        annual_delta[label] = np.array(series)

    # Aggregate across crops
    annual_delta['OVERALL'] = np.nanmean(
        np.array(list(annual_delta.values())[:-1]), axis=0
    )

    # Build annual regional mean permafrost series (1999–2018)
    for pv in PERM_VARS:
        pname  = pv['name']
        plabel = perm_data[pname]['label']
        stack  = perm_data[pname]['stack_post']   # (20, rows, cols)

        annual_perm = np.array([
            regional_mean(stack[i], mask) for i in range(stack.shape[0])
        ])

        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        years_arr = np.array(YEARS_CF)

        for i_crop, (crop_label, delta_series) in enumerate(annual_delta.items()):
            ax = axes[i_crop]

            valid = np.isfinite(annual_perm) & np.isfinite(delta_series)
            if valid.sum() < 4:
                ax.set_title(f'{crop_label}\n(insufficient data)', fontsize=9)
                ax.axis('off')
                continue

            x = annual_perm[valid]
            y = delta_series[valid]
            r, p = spearmanr(x, y)
            sig  = '★' if p < 0.05 else ''

            # Scatter with year labels
            sc = ax.scatter(x, y, c=years_arr[valid], cmap='viridis',
                            s=40, zorder=3)
            plt.colorbar(sc, ax=ax, shrink=0.7, label='Year')
            ax.axhline(0, color='black', linewidth=0.6, linestyle='--')
            ax.set_xlabel(f'Regional Mean {plabel}', fontsize=8)
            ax.set_ylabel('Regional Mean ΔSuitability', fontsize=8)
            ax.set_title(f'{crop_label}\nρ={r:.3f}, p={p:.3f}{sig}',
                         fontsize=9, fontweight='bold')

            all_results.append({
                'analysis'   : 'temporal',
                'crop'       : crop_label,
                'perm_var'   : pname,
                'metric'     : 'annual_mean',
                'spearman_r' : round(r, 4),
                'p_value'    : round(p, 4),
                'significant': p < 0.05,
                'n_years'    : int(valid.sum()),
            })

        for j in range(i_crop + 1, len(axes)):
            axes[j].axis('off')

        fig.suptitle(
            f'Temporal Correlation: Annual Regional Mean {plabel} vs ΔSuitability\n'
            f'(1999–2018) ★ = significant at p < 0.05',
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout()
        fig.savefig(f'{out_dir_b}/{pname}_temporal_corr.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  ✓ {pname} temporal correlation saved')

    # ── Save results CSV ──────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    df.to_csv(f'{OUT_ROOT}/correlation_results.csv', index=False)

    print(f'\n✓ All outputs saved to: {OUT_ROOT}/')
    print('\nSpatial correlation summary:')
    df_s = df[df['analysis'] == 'spatial']
    print(df_s[['crop', 'perm_var', 'metric', 'spearman_r',
                'p_value', 'significant']].to_string(index=False))
    print('\nTemporal correlation summary:')
    df_t = df[df['analysis'] == 'temporal']
    print(df_t[['crop', 'perm_var', 'spearman_r',
                'p_value', 'significant']].to_string(index=False))


if __name__ == '__main__':
    mask = load_mask()
    run(mask)