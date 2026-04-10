"""
Negative ΔYield Cluster Diagnostic
====================================
Identifies pixels where mean(observed - no_thaw) < 0 across crops,
then compares their terrain and soil moisture characteristics against
the rest of the study area to find a physical explanation.

Outputs: ./thaw_analysis_output/12_negative_cluster_diagnostic/
"""

# =============================================================================
# CONFIGURATION
# =============================================================================

WORK_DIR   = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH  = r'./data_input/qilian mask.tif'
ELEV_PATH  = r'./data_input/terrain/elevation.npy'
SLOPE_PATH = r'./data_input/terrain/slope.tif'   # update to your actual path

# Available soil moisture: one .npy per year
ASM_PATH_PATTERN = r'./data_input/permafrost_yearly/{year}/avail_soil_moisture.npy'

YEARS_BASELINE   = list(range(1979, 1999))   # no-thaw period

YEARS_COMPARISON = list(range(1999, 2019))

# Minimum number of crops that must agree on negative ΔYield
# for a pixel to be included in the negative cluster
# 1 = any crop negative, 5 = majority of crops negative
MIN_CROPS_NEGATIVE = 3

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

OUT_DIR = './thaw_analysis_output/12_negative_cluster_diagnostic'

# =============================================================================
# IMPORTS
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    band   = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    arr    = ds.ReadAsArray().astype(float)
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[arr < -1e10] = np.nan
    return arr

def load_mask():
    return load_raster(MASK_PATH).astype(bool)

def match_grid(arr, target_shape):
    if arr.shape == target_shape:
        return arr
    zy = target_shape[0] / arr.shape[0]
    zx = target_shape[1] / arr.shape[1]
    return zoom(arr, (zy, zx), order=1)

def load_mean_diff(tag, mask):
    """Mean annual ΔYield (observed - no-thaw) per pixel over comparison period."""
    obs_stack, cf_stack = [], []
    for year in YEARS_COMPARISON:
        obs = load_raster(f'./data_output/final_classification/{tag}/{year}_raw_yield.tif')
        cf  = load_raster(f'./data_output/final_classification_nothaw/{tag}/{year}_raw_yield.tif')
        if obs is not None:
            obs[~mask] = np.nan
            obs_stack.append(obs)
        if cf is not None:
            cf[~mask] = np.nan
            cf_stack.append(cf)
    if not obs_stack or not cf_stack:
        return None
    mean_diff = np.nanmean(obs_stack, axis=0) - np.nanmean(cf_stack, axis=0)
    mean_diff[~mask] = np.nan
    return mean_diff

def load_asm_means(mask):
    """
    Load mean available soil moisture for both periods:
      - baseline  (1979-1998): represents no-thaw condition
      - comparison (1999-2018): represents observed thaw condition

    Returns delta_asm = mean_comparison - mean_baseline per pixel.
    Positive delta = observed scenario is wetter (waterlogging hypothesis)
    Negative delta = observed scenario is drier (drainage/drought hypothesis)
    """
    print('  Loading available soil moisture...')

    def _load_period(years, label):
        stack = []
        for year in years:
            path = ASM_PATH_PATTERN.format(year=year)
            try:
                arr = np.load(path).astype(float)
            except FileNotFoundError:
                print(f'    Warning: missing {path}')
                continue
            # shape is (rows, cols, days) — collapse days to annual mean
            if arr.ndim == 3:
                arr = np.nanmean(arr, axis=2)
            arr = match_grid(arr, mask.shape)
            arr[~mask] = np.nan
            arr[arr < 0] = np.nan
            stack.append(arr)
        if not stack:
            print(f'    Warning: no ASM files found for {label}')
            return None
        mean = np.nanmean(stack, axis=0)
        print(f'    {label}: {len(stack)} years loaded, '
              f'mean ASM = {np.nanmean(mean):.4f}')
        return mean

    asm_baseline   = _load_period(YEARS_BASELINE,   '1979-1998 baseline')
    asm_comparison = _load_period(YEARS_COMPARISON, '1999-2018 comparison')

    if asm_baseline is None or asm_comparison is None:
        return None, None, None

    delta_asm = np.where(
        np.isfinite(asm_baseline) & np.isfinite(asm_comparison),
        asm_comparison - asm_baseline, np.nan
    )
    return asm_baseline, asm_comparison, delta_asm

# =============================================================================
# STEP 1 — Build negative cluster mask
# =============================================================================

def build_negative_cluster(mask):
    """
    A pixel enters the negative cluster if mean ΔYield < 0
    for at least MIN_CROPS_NEGATIVE crops.
    Also returns per-crop diff maps for plotting.
    """
    print('\n[1] Building negative cluster mask...')
    negative_count = np.zeros(mask.shape, dtype=int)
    crop_diffs = {}

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        diff = load_mean_diff(tag, mask)
        if diff is None:
            continue
        crop_diffs[label] = diff
        # Count pixels where this crop shows negative thaw impact
        negative_count += np.where(np.isfinite(diff) & (diff < 0), 1, 0)
        n_neg = int(np.sum(np.isfinite(diff) & (diff < 0)))
        print(f'  {label}: {n_neg} negative pixels')

    cluster_mask = mask & (negative_count >= MIN_CROPS_NEGATIVE)
    other_mask   = mask & (negative_count <  MIN_CROPS_NEGATIVE) & \
                   np.any([np.isfinite(d) for d in crop_diffs.values()], axis=0)

    print(f'\n  Negative cluster: {cluster_mask.sum()} pixels '
          f'(≥{MIN_CROPS_NEGATIVE} crops negative)')
    print(f'  Rest of study area: {other_mask.sum()} pixels')

    return cluster_mask, other_mask, negative_count, crop_diffs

# =============================================================================
# STEP 2 — Map the cluster
# =============================================================================

def plot_cluster_map(mask, cluster_mask, negative_count, crop_diffs):
    print('\n[2] Mapping negative cluster...')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Negative ΔYield Cluster — Spatial Distribution',
                 fontsize=13, fontweight='bold')

    # Left: how many crops agree on negative
    ax = axes[0]
    display = np.where(mask, negative_count.astype(float), np.nan)
    im = ax.imshow(display, cmap='RdYlGn_r', vmin=0, vmax=len(CROPS),
                   interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Number of crops with negative ΔYield',
                 shrink=0.75)
    ax.set_title(f'Crop agreement on negative ΔYield\n'
                 f'Cluster threshold: ≥{MIN_CROPS_NEGATIVE} crops', fontsize=10)
    ax.axis('off')

    # Right: cluster vs rest
    ax = axes[1]
    cluster_display = np.where(mask, 0.0, np.nan)         # rest = 0
    cluster_display = np.where(cluster_mask, 1.0, cluster_display)  # cluster = 1
    cmap2 = plt.cm.colors if hasattr(plt.cm, 'colors') else None
    from matplotlib.colors import ListedColormap
    cmap2 = ListedColormap(['#b2df8a', '#e31a1c'])
    im2 = ax.imshow(cluster_display, cmap=cmap2, vmin=0, vmax=1,
                    interpolation='nearest')
    cb2 = plt.colorbar(im2, ax=ax, shrink=0.75, ticks=[0.25, 0.75])
    cb2.set_ticklabels(['Rest of area', 'Negative cluster'])
    ax.set_title(f'Negative cluster mask\n'
                 f'({int(cluster_mask.sum())} pixels)', fontsize=10)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/cluster_map.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: cluster_map.png')

# =============================================================================
# STEP 3 — Compare terrain + moisture: cluster vs rest
# =============================================================================

def compare_characteristics(cluster_mask, other_mask, elevation, slope,
                            asm_baseline, asm_comparison, delta_asm):
    print('\n[3] Comparing cluster vs rest characteristics...')

    variables = {
        'Elevation (m)'             : elevation,
        'Slope (°)'                 : slope,
        'ASM baseline 1979-1998'    : asm_baseline,
        'ASM comparison 1999-2018'  : asm_comparison,
        'ΔASM (obs − baseline)'     : delta_asm,
    }
    # drop any None entries
    variables = {k: v for k, v in variables.items() if v is not None}

    records = []
    fig, axes = plt.subplots(1, len(variables),
                             figsize=(4.5 * len(variables), 5))
    if len(variables) == 1:
        axes = [axes]
    fig.suptitle('Negative Cluster vs Rest of Area — Terrain & Moisture',
                 fontsize=12, fontweight='bold')

    for ax, (vname, arr) in zip(axes, variables.items()):
        arr_matched = match_grid(arr, cluster_mask.shape)
        clust_vals  = arr_matched[cluster_mask & np.isfinite(arr_matched)]
        other_vals  = arr_matched[other_mask   & np.isfinite(arr_matched)]

        if len(clust_vals) < 3 or len(other_vals) < 3:
            print(f'  Skipping {vname} — insufficient data')
            continue

        stat, p = stats.mannwhitneyu(clust_vals, other_vals,
                                     alternative='two-sided')
        records.append({
            'variable'      : vname,
            'cluster_mean'  : round(float(np.nanmean(clust_vals)),  4),
            'cluster_median': round(float(np.nanmedian(clust_vals)), 4),
            'other_mean'    : round(float(np.nanmean(other_vals)),   4),
            'other_median'  : round(float(np.nanmedian(other_vals)), 4),
            'mannwhitney_p' : round(p, 5),
            'significant'   : p < 0.05,
        })

        ax.boxplot([clust_vals, other_vals],
                   labels=['Negative\ncluster', 'Rest of\narea'],
                   patch_artist=True,
                   boxprops=dict(facecolor='#e31a1c', alpha=0.6),
                   medianprops=dict(color='black', lw=2))
        sig_str = f'p={p:.3f}{"*" if p < 0.05 else ""}'

        # Add directional annotation for ΔASM
        if 'ΔASM' in vname:
            direction = 'wetter in obs' \
                if np.nanmean(clust_vals) > 0 else 'drier in obs'
            ax.set_title(f'{vname}\n{sig_str}\n({direction})', fontsize=9)
        else:
            ax.set_title(f'{vname}\n{sig_str}', fontsize=10)
        ax.set_ylabel(vname, fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/cluster_vs_rest_boxplots.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: cluster_vs_rest_boxplots.png')

    # Also map ΔASM spatially so you can see if it aligns with the cluster
    if delta_asm is not None:
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
        fig2.suptitle('ΔASM (1999-2018 minus 1979-1998) vs Negative Cluster',
                      fontsize=12, fontweight='bold')
        vlim = np.nanpercentile(np.abs(delta_asm[np.isfinite(delta_asm)]), 97)

        im = axes2[0].imshow(np.where(cluster_mask.shape == delta_asm.shape,
                             delta_asm, np.nan),
                             cmap='RdBu', vmin=-vlim, vmax=vlim,
                             interpolation='nearest')
        plt.colorbar(im, ax=axes2[0], label='ΔASM (m³/m³)', shrink=0.75)
        axes2[0].set_title('ΔASM spatial map', fontsize=10)
        axes2[0].axis('off')

        from matplotlib.colors import ListedColormap
        cmap2 = ListedColormap(['#b2df8a', '#e31a1c'])
        cluster_display = np.where(cluster_mask, 1.0,
                          np.where(other_mask,   0.0, np.nan))
        im2 = axes2[1].imshow(cluster_display, cmap=cmap2, vmin=0, vmax=1,
                              interpolation='nearest')
        cb2 = plt.colorbar(im2, ax=axes2[1], shrink=0.75, ticks=[0.25, 0.75])
        cb2.set_ticklabels(['Rest of area', 'Negative cluster'])
        axes2[1].set_title('Negative cluster mask', fontsize=10)
        axes2[1].axis('off')

        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/delta_asm_vs_cluster.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print('  Saved: delta_asm_vs_cluster.png')

    df = pd.DataFrame(records)
    df.to_csv(f'{OUT_DIR}/cluster_characteristics.csv', index=False)
    print('\nCharacteristics comparison:')
    print(df.to_string(index=False))
    return df

# =============================================================================
# STEP 4 — Per-crop ΔYield in cluster vs rest
# =============================================================================

def compare_per_crop(cluster_mask, other_mask, crop_diffs):
    print('\n[4] Per-crop ΔYield: cluster vs rest...')
    records = []

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('ΔYield Distribution: Negative Cluster vs Rest',
                 fontsize=13, fontweight='bold')
    axes = axes.flatten()

    for ax, (label, diff) in zip(axes, crop_diffs.items()):
        clust_vals = diff[cluster_mask & np.isfinite(diff)]
        other_vals = diff[other_mask  & np.isfinite(diff)]

        if len(clust_vals) < 3 or len(other_vals) < 3:
            ax.set_title(f'{label}\n(no data)')
            continue

        stat, p = stats.mannwhitneyu(clust_vals, other_vals, alternative='two-sided')
        records.append({
            'crop'           : label,
            'cluster_mean_dy': round(float(np.nanmean(clust_vals)), 4),
            'other_mean_dy'  : round(float(np.nanmean(other_vals)), 4),
            'mannwhitney_p'  : round(p, 5),
            'significant'    : p < 0.05,
        })

        ax.boxplot([clust_vals, other_vals],
                   labels=['Cluster', 'Rest'],
                   patch_artist=True,
                   boxprops=dict(facecolor='#e31a1c', alpha=0.5),
                   medianprops=dict(color='black', lw=2))
        ax.axhline(0, color='grey', lw=0.8, linestyle='--')
        ax.set_title(f'{label}\np={p:.3f}{"*" if p < 0.05 else ""}',
                     fontsize=9)
        ax.set_ylabel('ΔYield (kg/ha)', fontsize=8)

    for ax in axes[len(crop_diffs):]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/per_crop_deltayield_cluster.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: per_crop_deltayield_cluster.png')

    df = pd.DataFrame(records)
    df.to_csv(f'{OUT_DIR}/per_crop_cluster_comparison.csv', index=False)
    print('\nPer-crop comparison:')
    print(df.to_string(index=False))
    return df

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import numpy as np
    arr = np.load('./data_input/permafrost_yearly/1979/avail_soil_moisture.npy')
    print(arr.shape)
    arr = np.nanmean(arr, axis=2)
    print(arr.shape)
    print('=' * 60)
    print('Negative ΔYield Cluster Diagnostic')
    print('=' * 60)

    mask      = load_mask().astype(bool)
    elevation = match_grid(np.load(ELEV_PATH), mask.shape)
    slope     = match_grid(load_raster(SLOPE_PATH), mask.shape)
    asm_base, asm_comp, delta_asm = load_asm_means(mask)

    cluster_mask, other_mask, neg_count, crop_diffs = build_negative_cluster(mask)

    if cluster_mask.sum() == 0:
        print('\nNo negative cluster found. Try lowering MIN_CROPS_NEGATIVE.')
    else:
        plot_cluster_map(mask, cluster_mask, neg_count, crop_diffs)
        compare_characteristics(cluster_mask, other_mask, elevation, slope,
                                asm_base, asm_comp, delta_asm)
        compare_per_crop(cluster_mask, other_mask, crop_diffs)

    print(f'\nAll outputs written to: {OUT_DIR}/')