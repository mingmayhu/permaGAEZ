"""
Active Layer Thickness vs ΔYield — Direct Thaw-Yield Correlation
=================================================================
Tests the most direct hypothesis:
  "Where permafrost thaw was most intense (deeper active layer),
   was the yield impact of thaw larger?"

Three ALT metrics computed per pixel:
  1. Mean ALT 1999-2018        — absolute thaw depth during comparison period
  2. ΔALT (1999-2018 baseline) — did the active layer deepen vs 1979-1998?
  3. ALT trend (m/year)        — is it continuously getting deeper?

All three are correlated against ΔYield per crop.
All 10 crops summarised in one headline heatmap.

Outputs: ./thaw_analysis_output/10_alt_correlation/
"""

# =============================================================================
# CONFIGURATION
# =============================================================================

WORK_DIR  = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH = r'./data_input/qilian mask.tif'
ELEV_PATH = r'./data_input/terrain/elevation.npy'

# One .npy file per year, shape (rows, cols). Update pattern to match your filenames.
ALT_PATH_PATTERN = r'./data_input/permafrost_yearly/{year}/active_layer_depth.npy'

YEARS_ALL        = list(range(1979, 2019))
YEARS_COMPARISON = list(range(1999, 2019))

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

ELEV_BINS = [2500, 3000, 3500, 4000, 4500]
OUT_DIR   = './thaw_analysis_output/10_alt_correlation'
MIN_PIX   = 8   # minimum pixels per bin to compute correlation

# =============================================================================
# IMPORTS
# =============================================================================

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from scipy import stats
from scipy.ndimage import zoom

try:
    from osgeo import gdal
except ImportError:
    import gdal

os.chdir(WORK_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# HELPERS
# =============================================================================

def load_raster(path):
    ds = gdal.Open(path)
    if ds is None:
        return None
    band    = ds.GetRasterBand(1)
    nodata  = band.GetNoDataValue()
    arr     = ds.ReadAsArray().astype(float)
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[arr < -1e10] = np.nan
    return arr

def load_mask():
    return load_raster(MASK_PATH).astype(bool)

def match_grid(arr, target_shape):
    """Resample a 2D array to target_shape. Ignores extra leading dimensions."""
    if arr.ndim == 3:
        time_axis = int(np.argmax(arr.shape))
        arr = np.moveaxis(arr, time_axis, 0)
        arr = np.nanmean(arr, axis=0)
    if arr.shape == target_shape:
        return arr
    zy = target_shape[0] / arr.shape[0]
    zx = target_shape[1] / arr.shape[1]
    return zoom(arr, (zy, zx), order=1)

def load_delta_yield(tag, mask):
    obs_stack, cf_stack = [], []
    for year in YEARS_COMPARISON:
        obs = load_raster(f'./data_output/final_classification/{tag}/{year}_raw_yield.tif')
        cf  = load_raster(f'./data_output/final_classification_nothaw/{tag}/{year}_raw_yield.tif')
        if obs is not None:
            obs[~mask] = np.nan; obs_stack.append(obs)
        if cf is not None:
            cf[~mask]  = np.nan; cf_stack.append(cf)
    if not obs_stack or not cf_stack:
        return None
    mean_obs = np.nanmean(obs_stack, axis=0)
    mean_cf  = np.nanmean(cf_stack,  axis=0)
    delta    = np.where(np.isfinite(mean_obs) & np.isfinite(mean_cf),
                        mean_obs - mean_cf, np.nan)
    delta[~mask] = np.nan
    return delta

# =============================================================================
# LOAD ALT & COMPUTE METRICS
# =============================================================================

def load_alt_metrics(mask):
    print('Loading ALT files...')
    baseline_stack, comp_stack = [], []

    for year in YEARS_ALL:
        path = ALT_PATH_PATTERN.format(year=year)
        try:
            arr = np.load(path).astype(float)
        except FileNotFoundError:
            print(f'  Warning: missing {path}')
            continue
        # Handle 3D arrays: (rows, cols, days) or (days, rows, cols)
        if arr.ndim == 3:
            time_axis = int(np.argmax(arr.shape))
            arr = np.moveaxis(arr, time_axis, 0)
            arr = np.nanmean(arr, axis=0)   # collapse to annual mean ALT
        if arr.ndim != 2:
            print(f'  Skipping {path} — unexpected shape {arr.shape}')
            continue
        arr = match_grid(arr, mask.shape)
        arr[~mask]    = np.nan
        arr[arr <= 0] = np.nan
        if year < 1999:
            baseline_stack.append(arr)
        else:
            comp_stack.append(arr)

    if not baseline_stack or not comp_stack:
        raise RuntimeError('ALT files not found. Check ALT_PATH_PATTERN.')

    alt_base = np.nanmean(baseline_stack, axis=0)
    alt_comp = np.nanmean(comp_stack,     axis=0)
    delta_alt = alt_comp - alt_base

    # Pixel-wise OLS trend over comparison period
    print(f'  Computing pixel-wise trend ({len(comp_stack)} years)...')
    stack_arr = np.array(comp_stack)
    x         = np.arange(stack_arr.shape[0], dtype=float)
    alt_trend = np.full(mask.shape, np.nan)

    for r, c in zip(*np.where(mask)):
        series = stack_arr[:, r, c]
        fin    = np.isfinite(series)
        if fin.sum() >= 4:
            slope, *_ = stats.linregress(x[fin], series[fin])
            alt_trend[r, c] = slope

    print(f'  Done. Mean ALT (comp)={np.nanmean(alt_comp):.2f} m  '
          f'Mean ΔALT={np.nanmean(delta_alt):.3f} m  '
          f'Mean trend={np.nanmean(alt_trend):.4f} m/yr')

    return {'alt_mean_comp': alt_comp,
            'delta_alt'    : delta_alt,
            'alt_trend'    : alt_trend}

# =============================================================================
# ANALYSIS 1 — Spatial ALT maps
# =============================================================================

def analysis_spatial(alt, mask):
    print('\n[1] Spatial ALT maps...')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Active Layer Thickness — Spatial Distribution',
                 fontsize=13, fontweight='bold')
    panels = [
        ('alt_mean_comp', 'Mean ALT 1999-2018 (m)',            'YlOrRd', False),
        ('delta_alt',     'ΔALT: 1999-2018 minus 1979-1998 (m)','RdBu_r', True),
        ('alt_trend',     'ALT trend 1999-2018 (m/yr)',         'RdBu_r', True),
    ]
    for ax, (key, title, cmap, div) in zip(axes, panels):
        arr   = np.where(mask, alt[key], np.nan)
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            ax.set_title(title + '\n(no data)'); continue
        if div:
            vlim = np.nanpercentile(np.abs(valid), 97)
            im   = ax.imshow(arr, cmap=cmap, vmin=-vlim, vmax=vlim,
                             interpolation='nearest')
        else:
            im = ax.imshow(arr, cmap=cmap, interpolation='nearest')
        plt.colorbar(im, ax=ax, shrink=0.75)
        ax.set_title(title, fontsize=10); ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/ALT_spatial_maps.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: ALT_spatial_maps.png')

# =============================================================================
# ANALYSIS 2 — Scatter: ALT vs ΔYield per crop (coloured by elevation)
# =============================================================================

def analysis_scatter(alt, mask, elevation):
    print('\n[2] Scatter plots — ALT vs ΔYield...')
    metric_labels = {
        'alt_mean_comp': 'Mean ALT 1999-2018 (m)',
        'delta_alt'    : 'ΔALT: deepening vs baseline (m)',
        'alt_trend'    : 'ALT trend (m/yr)',
    }
    all_results = []

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        dy = load_delta_yield(tag, mask)
        if dy is None:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        fig.suptitle(f'{label} — Active Layer Thickness vs ΔYield',
                     fontsize=12, fontweight='bold')

        for ax, (mkey, xlabel) in zip(axes, metric_labels.items()):
            arr   = alt[mkey]
            valid = (mask & np.isfinite(dy) & np.isfinite(arr) & np.isfinite(elevation))
            x, y, e = arr[valid], dy[valid], elevation[valid]

            if len(x) < 10:
                ax.set_title('Insufficient data'); continue

            sc = ax.scatter(x, y, c=e, cmap='terrain_r',
                            alpha=0.4, s=6, rasterized=True)
            plt.colorbar(sc, ax=ax, label='Elevation (m)', shrink=0.8)

            slope, intercept, *_ = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 200)
            ax.plot(x_line, slope * x_line + intercept, 'k-', lw=1.5)

            r_sp, p_sp = stats.spearmanr(x, y)
            ax.axhline(0, color='grey', lw=0.7, linestyle='--')
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel('ΔYield (kg/ha)', fontsize=10)
            ax.set_title(f'Spearman r={r_sp:.3f}  p={p_sp:.2e}\n'
                         f'slope={slope:.4f}  n={len(x):,}', fontsize=9)

            all_results.append({
                'crop'       : label,
                'alt_metric' : mkey,
                'spearman_r' : round(r_sp, 4),
                'spearman_p' : round(p_sp, 6),
                'ols_slope'  : round(slope, 5),
                'n_pixels'   : int(len(x)),
                'significant': p_sp < 0.05,
            })

        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/{tag.replace("combined_","")}_scatter.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  {label}')

    df = pd.DataFrame(all_results)
    df.to_csv(f'{OUT_DIR}/alt_correlation_summary.csv', index=False)
    return df

# =============================================================================
# ANALYSIS 3 — Elevation-binned correlation heatmap
# =============================================================================

def analysis_elevation_binned(alt, mask, elevation):
    print('\n[3] Elevation-binned correlation...')
    bins        = ELEV_BINS
    band_labels = [f'{lo}-{hi}m' for lo, hi in zip(bins[:-1], bins[1:])]
    records     = []

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        dy = load_delta_yield(tag, mask)
        if dy is None:
            continue
        for mkey in ['alt_mean_comp', 'delta_alt', 'alt_trend']:
            arr = alt[mkey]
            for lo, hi, blabel in zip(bins[:-1], bins[1:], band_labels):
                sel = (mask & np.isfinite(dy) & np.isfinite(arr)
                       & (elevation >= lo) & (elevation < hi))
                if sel.sum() < MIN_PIX:
                    continue
                r, p = stats.spearmanr(arr[sel], dy[sel])
                records.append({'crop': label, 'elev_band': blabel,
                                 'alt_metric': mkey, 'spearman_r': round(r, 4),
                                 'spearman_p': round(p, 5), 'n': int(sel.sum())})

    df = pd.DataFrame(records)
    df.to_csv(f'{OUT_DIR}/alt_elev_binned.csv', index=False)

    for mkey in ['alt_mean_comp', 'delta_alt', 'alt_trend']:
        sub   = df[df['alt_metric'] == mkey]
        if sub.empty:
            continue
        pivot = sub.pivot_table(index='crop', columns='elev_band',
                                values='spearman_r', aggfunc='first')
        pivot_p = sub.pivot_table(index='crop', columns='elev_band',
                                  values='spearman_p', aggfunc='first')
        fig, ax = plt.subplots(figsize=(10, 7))
        norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        im   = ax.imshow(pivot.values, cmap='RdBu', norm=norm, aspect='auto')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=10)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=10)
        plt.colorbar(im, ax=ax, label='Spearman r')
        ax.set_title(f'r(ΔYield, {mkey}) by elevation  [outline = p<0.05]',
                     fontsize=11, fontweight='bold')
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                p = pivot_p.values[i, j] if pivot_p.values.shape == pivot.values.shape else np.nan
                if np.isfinite(v):
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                            fontsize=9, color='white' if abs(v) > 0.6 else 'black')
                if np.isfinite(p) and p < 0.05:
                    ax.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,
                                            edgecolor='black',lw=1.5))
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/elev_binned_{mkey}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved elev heatmap: {mkey}')
    return df

# =============================================================================
# ANALYSIS 4 — Headline summary heatmap: all crops x ALT metrics
# =============================================================================

def analysis_summary_heatmap(corr_df):
    print('\n[4] Summary heatmap...')
    metric_labels = {
        'alt_mean_comp': 'Mean ALT\n1999-2018',
        'delta_alt'    : 'ΔALT\n(deepening)',
        'alt_trend'    : 'ALT trend\n(m/yr)',
    }
    metrics     = list(metric_labels.keys())
    crop_labels = [c['label'] for c in CROPS]

    r_mat = np.full((len(crop_labels), len(metrics)), np.nan)
    p_mat = np.full((len(crop_labels), len(metrics)), np.nan)

    for i, cl in enumerate(crop_labels):
        for j, mk in enumerate(metrics):
            row = corr_df[(corr_df['crop'] == cl) & (corr_df['alt_metric'] == mk)]
            if not row.empty:
                r_mat[i, j] = row['spearman_r'].values[0]
                p_mat[i, j] = row['spearman_p'].values[0]

    fig, ax = plt.subplots(figsize=(8, 9))
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im   = ax.imshow(r_mat, cmap='RdBu', norm=norm, aspect='auto')
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([metric_labels[m] for m in metrics], fontsize=11)
    ax.set_yticks(range(len(crop_labels)))
    ax.set_yticklabels(crop_labels, fontsize=11)
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cb.set_label('Spearman r', fontsize=11)
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])
    ax.set_title('Permafrost Thaw Intensity vs Yield Impact\n'
                 'Spearman r — all crops x ALT metrics\n'
                 'Black outline = p < 0.05',
                 fontsize=12, fontweight='bold', pad=14)

    for i in range(len(crop_labels)):
        for j in range(len(metrics)):
            v, p = r_mat[i, j], p_mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                        fontsize=10,
                        color='white' if abs(v) > 0.55 else 'black',
                        fontweight='500' if (np.isfinite(p) and p < 0.05) else 'normal')
            if np.isfinite(p) and p < 0.05:
                ax.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,
                                        edgecolor='black',lw=2))

    fig.text(0.02, 0.01,
             'Blue = deeper/more thaw → higher ΔYield (thaw helps)   '
             'Red = deeper/more thaw → lower ΔYield (thaw hurts)',
             fontsize=8, color='#555')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/SUMMARY_alt_vs_deltayield.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: SUMMARY_alt_vs_deltayield.png')
    print('\nFull correlation table:')
    print(corr_df.to_string(index=False))

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print('=' * 60)
    print('Active Layer Thickness vs ΔYield')
    print('=' * 60)

    mask      = load_mask().astype(bool)
    elevation = match_grid(np.load(ELEV_PATH), mask.shape)
    alt       = load_alt_metrics(mask)

    analysis_spatial(alt, mask)
    corr_df = analysis_scatter(alt, mask, elevation)
    analysis_elevation_binned(alt, mask, elevation)
    analysis_summary_heatmap(corr_df)

    print(f'\nAll outputs written to: {OUT_DIR}/')