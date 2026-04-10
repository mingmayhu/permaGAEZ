"""
Permafrost Thaw Impact Analysis — All 10 Crop Types
=====================================================
Compares observed (1999–2018) vs. no-thaw counterfactual (_nothaw) runs.

Analyses:
  1. Spatial ΔYield maps (observed minus counterfactual), averaged 1999–2018
  2. Regional mean yield time series (1979–2018 observed + 1999–2018 counterfactual)
  3. Suitability class transition matrices (observed vs. counterfactual, 1999–2018)
  4. Elevation-stratified ΔYield profiles
  5. Mann-Kendall trend test on regional mean yield (both scenarios)
  6. Cross-crop sensitivity summary (ΔYield and ΔSuitability class by crop type)

Outputs written to:  ./analysis_output/
"""

# =============================================================================
# CONFIGURATION — adjust paths to match your setup
# =============================================================================

WORK_DIR   = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH  = r'./data_input/qilian mask.tif'
BASEPATH   = r'./data_input/qilian mask.tif'
ELEV_PATH  = r'./data_input/terrain/elevation.npy'

YEARS_ALL          = list(range(1979, 2019))   # full observed series
YEARS_COMPARISON   = list(range(1999, 2019))   # years with both observed & counterfactual

# Elevation bins (metres) for stratified analysis
ELEV_BINS = list(range(2000, 6000, 500))       # 2000, 2500, 3000, …, 5500

# Crop types: each entry defines the combined-map folder name and its constituent varieties
CROPS = [
    {
        'label'    : 'Winter Barley',
        'tag'      : 'combined_winter_barley',
        'varieties': ['winter_barley_59', 'winter_barley_60',
                      'winter_barley_61', 'winter_barley_62'],
    },
    {
        'label'    : 'Spring Barley',
        'tag'      : 'combined_spring_barley',
        'varieties': ['spring_barley_63', 'spring_barley_64',
                      'spring_barley_65', 'spring_barley_66'],
    },
    {
        'label'    : 'Winter Wheat',
        'tag'      : 'combined_winter_wheat',
        'varieties': ['winter_wheat_1', 'winter_wheat_2',
                      'winter_wheat_3', 'winter_wheat_4'],
    },
    {
        'label'    : 'Spring Wheat',
        'tag'      : 'combined_spring_wheat',
        'varieties': ['spring_wheat_5', 'spring_wheat_6', 'spring_wheat_7',
                      'spring_wheat_8', 'spring_wheat_9'],
    },
    {
        'label'    : 'Silage Maize',
        'tag'      : 'combined_silage_maize',
        'varieties': ['silage_maize_53', 'silage_maize_54', 'silage_maize_55',
                      'silage_maize_56', 'silage_maize_57', 'silage_maize_58'],
    },
    {
        'label'    : 'White Potato',
        'tag'      : 'combined_white_potato',
        'varieties': ['white_potato_135', 'white_potato_136', 'white_potato_137',
                      'white_potato_138', 'white_potato_139', 'white_potato_140',
                      'white_potato_141'],
    },
    {
        'label'    : 'Oat',
        'tag'      : 'combined_oat',
        'varieties': ['spring_oat_128', 'spring_oat_129'
                      , 'spring_oat_130'
                    ],
    },
    {
        'label'    : 'Dry Pea',
        'tag'      : 'combined_dry_pea',
        'varieties': ['dry_pea_189', 'dry_pea_190', 'dry_pea_191'],
    },
    {
        'label'    : 'Winter Rape',
        'tag'      : 'combined_winter_rape',
        'varieties': ['winter_rape_216', 'winter_rape_217',
                      'winter_rape_218', 'winter_rape_219'],
    },
    {
        'label'    : 'Spring Rape',
        'tag'      : 'combined_spring_rape',
        'varieties': ['spring_rape_220', 'spring_rape_221',
                      'spring_rape_222', 'spring_rape_223'],
    },
]

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec

try:
    from osgeo import gdal
except ImportError:
    import gdal

try:
    from pymannkendall import original_test as mk_test
    HAS_MK = True
except ImportError:
    HAS_MK = False
    warnings.warn("pymannkendall not installed — trend tests will use scipy linear regression instead.")
    from scipy.stats import linregress

os.chdir(WORK_DIR)

# =============================================================================
# HELPERS
# =============================================================================

OUT_ROOT = './thaw_analysis_output'

def make_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

make_dirs(OUT_ROOT)

# Add this helper function to the analysis script
def get_mask_display(mask):
    """Returns an RGBA array showing mask as light grey, non-mask as white."""
    display = np.ones((*mask.shape, 4), dtype=float)  # white background
    display[mask, :] = [0.85, 0.85, 0.85, 1.0]        # grey inside mask
    display[~mask, :] = [1.0, 1.0, 1.0, 0.0]          # transparent outside
    return display

def load_raster(path, mask=None):
    ds = gdal.Open(path)
    if ds is None:
        return None
    band = ds.GetRasterBand(1)
    nodata_val = band.GetNoDataValue()
    arr = ds.ReadAsArray().astype(float)
    
    # Replace the actual nodata sentinel first
    if nodata_val is not None:
        arr[arr == nodata_val] = np.nan
    
    # Catch any remaining large sentinels as fallback
    arr[arr < -1e10] = np.nan
    
    return arr

def load_mask():
    arr = load_raster(MASK_PATH)
    return arr.astype(bool) if arr is not None else None

def spatial_mean(arr, mask):
    """Mean over valid pixels that are inside mask AND nonzero."""
    valid = mask & np.isfinite(arr) & (arr > 0)
    return float(np.nanmean(arr[valid])) if valid.any() else np.nan

def load_combined_raw(tag, year, nothaw=False):
    """
    Load the best-variety raw yield raster for a crop type and year.
    nothaw=True loads the _nothaw version.
    """
    suffix = '_nothaw' if nothaw else ''
    path = f'./data_output/final_classification{suffix}/{tag}/{year}_raw_yield.tif'
    return load_raster(path)

def load_combined_class(tag, year, nothaw=False):
    """Load the suitability class raster (0–5)."""
    suffix = '_nothaw' if nothaw else ''
    path = f'./data_output/final_classification{suffix}/{tag}/{year}_final_yield_class.tif'
    return load_raster(path)

def trend_test(series):
    """
    Returns (slope_per_year, p_value).
    Uses Mann-Kendall if available, else OLS.
    """
    s = np.array(series, dtype=float)
    valid = np.isfinite(s)
    if valid.sum() < 4:
        return np.nan, np.nan
    if HAS_MK:
        res = mk_test(s[valid])
        return res.slope, res.p
    else:
        x = np.where(valid)[0]
        slope, _, _, p, _ = linregress(x, s[valid])
        return slope, p

SUITABILITY_CMAP = plt.get_cmap('RdYlGn', 6)   # 0–5
DIVERGING_CMAP   = 'RdBu'

# =============================================================================
# ANALYSIS 1 — Spatial ΔYield maps
# =============================================================================

def analysis_spatial_delta(mask, elevation):
    """
    For each crop type: mean(observed raw yield 1999–2018) minus
    mean(counterfactual raw yield 1999–2018), saved as GeoTIFF + figure.

    Also produces a 2×5 summary figure across all crops.
    """
    print("\n[Analysis 1] Spatial ΔYield maps …")
    out_dir = f'{OUT_ROOT}/1_spatial_delta'
    make_dirs(out_dir)

    summary_deltas = {}   # crop_label -> mean delta array

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        obs_stack  = []
        cf_stack   = []

        for year in YEARS_COMPARISON:
            obs = load_combined_raw(tag, year, nothaw=False)
            cf  = load_combined_raw(tag, year, nothaw=True)
            if obs is None:
                print(f"    ⚠ Missing observed: {tag} {year}")
            if cf is None:
                print(f"    ⚠ Missing counterfactual: {tag} {year}")
            if obs is not None:
                obs[~mask] = np.nan
                obs_stack.append(obs)
            if cf is not None:
                cf[~mask] = np.nan
                cf_stack.append(cf)
        if not obs_stack or not cf_stack:
            print(f"  ⚠ Skipping {label} — missing data.")
            continue

        # Also warn if stacks have different lengths
        if len(obs_stack) != len(cf_stack):
            print(f"  ⚠ {label}: obs has {len(obs_stack)} years, cf has {len(cf_stack)} years — check missing files.")

        if not obs_stack or not cf_stack:
            print(f"  ⚠ Skipping {label} — missing data.")
            continue
        mean_obs = np.nanmean(obs_stack, axis=0)
        mean_cf  = np.nanmean(cf_stack,  axis=0)

        # Only compute delta where BOTH arrays are valid
        both_valid = np.isfinite(mean_obs) & np.isfinite(mean_cf)
        delta = np.where(both_valid, mean_obs - mean_cf, np.nan)
        delta[~mask] = np.nan

        # mean_obs = np.nanmean(obs_stack, axis=0)
        # mean_cf  = np.nanmean(cf_stack,  axis=0)
        # delta    = mean_obs - mean_cf
        # delta[~mask] = np.nan
        summary_deltas[label] = delta

        # Only report stats where at least one scenario has nonzero yield
        either_valid = (mean_obs > 0) | (mean_cf > 0)
        inside = delta[mask & either_valid & np.isfinite(delta)]

        if len(inside) > 0:
            print("mean delta:   ", inside.mean())
            print("median delta: ", np.median(inside))
            print("% pixels negative:", (inside < 0).sum() / len(inside) * 100)
            print("% pixels > 50 kg/ha:", (inside > 50).sum() / len(inside) * 100)
            print("% pixels < -50 kg/ha:", (inside < -50).sum() / len(inside) * 100)

        # Per-crop figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        vmax_yield = np.nanpercentile(np.concatenate([mean_obs[mask], mean_cf[mask]]), 98)
        vlim_delta = np.nanpercentile(np.abs(delta[mask]), 98)

        for ax, arr, title, cmap, vmin, vmax in [
            (axes[0], mean_obs, 'Observed Mean Yield\n(1999–2018)',  'YlGn', 0, vmax_yield),
            (axes[1], mean_cf,  'Counterfactual Mean Yield\n(no thaw)', 'YlGn', 0, vmax_yield),
            (axes[2], delta,    'ΔYield (Obs − No-Thaw)\n[kg/ha]',   DIVERGING_CMAP, -vlim_delta, vlim_delta),
        ]:
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=11)
            ax.axis('off')
            plt.colorbar(im, ax=ax, shrink=0.75, label='kg/ha')

        fig.suptitle(f'{label} — Permafrost Thaw Impact on Yield', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{out_dir}/{tag}_delta_yield.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {label}")

    # Summary panel — all crops
    n = len(summary_deltas)
    if n == 0:
        return
    ncols = 5
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes = axes.flatten()
    global_vlim = np.nanpercentile(
        np.abs(np.concatenate([d[mask & np.isfinite(d)] for d in summary_deltas.values()])), 98)

    for i, (label, delta) in enumerate(summary_deltas.items()):
        im = axes[i].imshow(delta, cmap=DIVERGING_CMAP,
                            vmin=-global_vlim, vmax=global_vlim)
        axes[i].set_title(label, fontsize=10, fontweight='bold')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], shrink=0.8, label='kg/ha')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle('ΔYield (Observed − No-Thaw Counterfactual), Mean 1999–2018\nAll Crop Types',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/ALL_CROPS_delta_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ All-crop summary panel saved.")

# =============================================================================
# ANALYSIS 2 — Regional mean yield time series
# =============================================================================

def analysis_time_series(mask):
    """
    For each crop: plot regional mean yield 1979–2018 (observed) overlaid
    with 1999–2018 (counterfactual). Save CSV of values.
    """
    print("\n[Analysis 2] Regional mean yield time series …")
    out_dir = f'{OUT_ROOT}/2_time_series'
    make_dirs(out_dir)

    all_records = []

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        records = []

        for year in YEARS_ALL:
            obs = load_combined_raw(tag, year, nothaw=False)
            val_obs = spatial_mean(obs, mask) if obs is not None else np.nan
            records.append({'year': year, 'scenario': 'observed', 'mean_yield': val_obs})

        for year in YEARS_COMPARISON:
            cf = load_combined_raw(tag, year, nothaw=True)
            val_cf = spatial_mean(cf, mask) if cf is not None else np.nan
            records.append({'year': year, 'scenario': 'no_thaw', 'mean_yield': val_cf})

        df = pd.DataFrame(records)
        df['crop'] = label
        all_records.append(df)

        obs_df = df[df['scenario'] == 'observed'].sort_values('year')
        cf_df  = df[df['scenario'] == 'no_thaw'].sort_values('year')

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(obs_df['year'], obs_df['mean_yield'],
                color='#2166AC', linewidth=2, marker='o', markersize=4, label='Observed')
        ax.plot(cf_df['year'], cf_df['mean_yield'],
                color='#D6604D', linewidth=2, marker='s', markersize=4,
                linestyle='--', label='No-Thaw Counterfactual')
        ax.axvline(1999, color='grey', linestyle=':', linewidth=1.5, label='Divergence point (1999)')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Regional Mean Yield (kg/ha)', fontsize=12)
        ax.set_title(f'{label} — Regional Mean Yield: Observed vs. No-Thaw', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(f'{out_dir}/{tag}_timeseries.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {label}")

    combined_df = pd.concat(all_records, ignore_index=True)
    combined_df.to_csv(f'{out_dir}/all_crops_yield_timeseries.csv', index=False)
    print("  ✓ CSV saved: all_crops_yield_timeseries.csv")

# =============================================================================
# ANALYSIS 3 — Suitability class transition matrices
# =============================================================================

def analysis_transitions(mask):
    """
    For each crop: aggregate pixel-level class transitions
    (no-thaw → observed) across 1999–2018.
    Output: heatmap transition matrix + area (km²) table.
    Assumes 0.1° resolution ≈ 78 km² per pixel at ~37.5°N — adjust PIXEL_AREA_KM2 if needed.
    """
    print("\n[Analysis 3] Suitability class transition matrices …")
    out_dir = f'{OUT_ROOT}/3_transitions'
    make_dirs(out_dir)

    # Approximate pixel area at centre of study area (lat ~37.8°N, 0.1° grid)
    # Area = (111.32 * cos(lat)) * 111.32 * dx * dy  km²
    # For ERA5-Land ~0.1°: ≈ 78 km². Adjust to your actual resolution.
    PIXEL_AREA_KM2 = 78.0   # <-- adjust if using different resolution

    classes = list(range(6))   # 0–5

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        matrix = np.zeros((6, 6), dtype=float)   # [cf_class, obs_class]

        for year in YEARS_COMPARISON:
            obs_cls = load_combined_class(tag, year, nothaw=False)
            cf_cls  = load_combined_class(tag, year, nothaw=True)
            if obs_cls is None or cf_cls is None:
                continue
            obs_cls = np.where(np.isfinite(obs_cls), np.round(obs_cls), -1).astype(int)
            cf_cls  = np.where(np.isfinite(cf_cls),  np.round(cf_cls),  -1).astype(int)
            valid   = mask & np.isfinite(obs_cls) & np.isfinite(cf_cls)
            obs_cls[~valid] = -1
            cf_cls[~valid]  = -1
            for r in range(6):
                for c in range(6):
                    matrix[r, c] += np.sum((cf_cls == r) & (obs_cls == c) & valid)

        # Convert pixel-years to km² (divide by number of years to get mean annual area)
        matrix_km2 = matrix * PIXEL_AREA_KM2 / len(YEARS_COMPARISON)

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(matrix_km2, cmap='Blues', aspect='auto')
        ax.set_xticks(classes); ax.set_xticklabels([f'Obs {c}' for c in classes])
        ax.set_yticks(classes); ax.set_yticklabels([f'CF {c}' for c in classes])
        ax.set_xlabel('Observed Class (with thaw)', fontsize=11)
        ax.set_ylabel('Counterfactual Class (no thaw)', fontsize=11)
        ax.set_title(f'{label}\nSuitability Class Transitions (mean annual area, km²)', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Mean Annual Area (km²)')
        for r in range(6):
            for c in range(6):
                val = matrix_km2[r, c]
                ax.text(c, r, f'{val:.0f}', ha='center', va='center',
                        fontsize=8, color='white' if val > matrix_km2.max() * 0.5 else 'black')
        plt.tight_layout()
        plt.savefig(f'{out_dir}/{tag}_transition_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Save CSV
        df_mat = pd.DataFrame(matrix_km2, index=[f'CF_{c}' for c in classes],
                              columns=[f'Obs_{c}' for c in classes])
        df_mat.to_csv(f'{out_dir}/{tag}_transition_matrix.csv')
        print(f"  ✓ {label}")

# =============================================================================
# ANALYSIS 4 — Elevation-stratified ΔYield
# =============================================================================

def analysis_elevation(mask, elevation):
    """
    For each crop: compute mean ΔYield within elevation bands across 1999–2018.
    Single figure with all crops overlaid for comparison.
    """
    print("\n[Analysis 4] Elevation-stratified ΔYield …")
    out_dir = f'{OUT_ROOT}/4_elevation'
    make_dirs(out_dir)

    bins  = np.array(ELEV_BINS)
    mids  = bins[:-1] + np.diff(bins) / 2
    records = []

    cmap_colors = plt.get_cmap('tab10', len(CROPS))

    fig_all, ax_all = plt.subplots(figsize=(10, 6))

    for i_crop, crop in enumerate(CROPS):
        tag, label = crop['tag'], crop['label']
        delta_stack = []

        for year in YEARS_COMPARISON:
            obs = load_combined_raw(tag, year, nothaw=False)
            cf  = load_combined_raw(tag, year, nothaw=True)
            if obs is None or cf is None:
                continue
            delta = obs - cf
            delta[~mask] = np.nan
            delta_stack.append(delta)

        if not delta_stack:
            continue

        mean_delta = np.nanmean(delta_stack, axis=0)
        with np.errstate(all='ignore'):
             mean_delta = np.nanmean(delta_stack, axis=0)

        bin_means  = []

        for b_lo, b_hi in zip(bins[:-1], bins[1:]):
            in_bin = mask & (elevation >= b_lo) & (elevation < b_hi) & np.isfinite(mean_delta)
            val    = float(np.nanmean(mean_delta[in_bin])) if in_bin.any() else np.nan
            bin_means.append(val)
            records.append({'crop': label, 'elev_mid': (b_lo + b_hi) / 2,
                            'mean_delta_yield': val})

        # Per-crop figure
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(bin_means, mids, color='#2166AC', linewidth=2, marker='o')
        ax.axvline(0, color='grey', linestyle='--', linewidth=1)
        ax.set_xlabel('ΔYield (Obs − No-Thaw) [kg/ha]', fontsize=12)
        ax.set_ylabel('Elevation [m]', fontsize=12)
        ax.set_title(f'{label} — ΔYield by Elevation Band\n(Mean 1999–2018)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{out_dir}/{tag}_elevation_delta.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Add to all-crops overlay
        ax_all.plot(bin_means, mids, linewidth=1.8, marker='o', markersize=4,
                    label=label, color=cmap_colors(i_crop))

    ax_all.axvline(0, color='black', linestyle='--', linewidth=1)
    ax_all.set_xlabel('ΔYield (Obs − No-Thaw) [kg/ha]', fontsize=12)
    ax_all.set_ylabel('Elevation [m]', fontsize=12)
    ax_all.set_title('Elevation-Stratified ΔYield — All Crops\n(Mean 1999–2018)', fontsize=13, fontweight='bold')
    ax_all.legend(fontsize=9, loc='upper right')
    plt.tight_layout()
    fig_all.savefig(f'{out_dir}/ALL_CROPS_elevation_delta.png', dpi=150, bbox_inches='tight')
    plt.close()

    pd.DataFrame(records).to_csv(f'{out_dir}/elevation_delta_by_crop.csv', index=False)
    print("  ✓ All elevation profiles saved.")

# =============================================================================
# ANALYSIS 5 — Mann-Kendall trend tests
# =============================================================================

def analysis_trends(mask):
    """
    For each crop and scenario (observed 1999–2018, counterfactual 1999–2018):
    run trend test on regional mean yield.
    Output: summary table CSV + bar chart of slopes.
    """
    print("\n[Analysis 5] Trend tests …")
    out_dir = f'{OUT_ROOT}/5_trends'
    make_dirs(out_dir)

    results = []

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']

        for scenario, nothaw in [('Observed', False), ('No-Thaw CF', True)]:
            series = []
            for year in YEARS_COMPARISON:
                arr = load_combined_raw(tag, year, nothaw=nothaw)
                series.append(spatial_mean(arr, mask) if arr is not None else np.nan)
            slope, pval = trend_test(series)
            results.append({
                'crop': label, 'scenario': scenario,
                'slope_kg_ha_yr': round(slope, 2) if not np.isnan(slope) else np.nan,
                'p_value': round(pval, 4) if not np.isnan(pval) else np.nan,
                'significant': (pval < 0.05) if not np.isnan(pval) else False,
            })

    df = pd.DataFrame(results)
    df.to_csv(f'{out_dir}/trend_test_results.csv', index=False)

    # Bar chart: slope by crop, side-by-side for the two scenarios
    pivot = df.pivot(index='crop', columns='scenario', values='slope_kg_ha_yr')
    x = np.arange(len(pivot))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    bars_obs = ax.bar(x - width/2, pivot.get('Observed', 0),    width, label='Observed',       color='#2166AC', alpha=0.85)
    bars_cf  = ax.bar(x + width/2, pivot.get('No-Thaw CF', 0),  width, label='No-Thaw CF',     color='#D6604D', alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(pivot.index, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('Yield Trend (kg/ha/year)', fontsize=12)
    ax.set_title('Yield Trend 1999–2018: Observed vs. No-Thaw Counterfactual\nAll Crop Types',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)

    # Mark significant bars with *
    sig_lookup = df.set_index(['crop', 'scenario'])['significant'].to_dict()
    for bar, crop_name, scenario in [
        *[(b, pivot.index[j], 'Observed')    for j, b in enumerate(bars_obs)],
        *[(b, pivot.index[j], 'No-Thaw CF') for j, b in enumerate(bars_cf)],
    ]:
        if sig_lookup.get((crop_name, scenario), False):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5 * np.sign(bar.get_height()),
                    '*', ha='center', va='bottom', fontsize=12, color='black')

    plt.tight_layout()
    plt.savefig(f'{out_dir}/trend_slopes_barchart.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Trend results saved.")
    print(df.to_string(index=False))

def analysis_trends_pixelwise(mask_arr):
    """
    Run trend test pixel-by-pixel rather than on regional mean.
    Reports % of pixels with significant positive/negative trend.
    """
    print("\n[Analysis 5b] Pixel-wise trend tests …")
    out_dir = f'{OUT_ROOT}/5_trends'
    make_dirs(out_dir)

    results = []

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']

        for scenario, nothaw in [('Observed', False), ('No-Thaw CF', True)]:
            # Stack all years into (n_years, rows, cols)
            stack = []
            for year in YEARS_COMPARISON:
                arr = load_raster(
                    f'./data_output/final_classification{"_nothaw" if nothaw else ""}/{tag}/{year}_raw_yield.tif',
                    mask=mask_arr
                )
                if arr is not None:
                    stack.append(arr)

            if not stack:
                continue

            stack = np.array(stack)   # (n_years, rows, cols)
            rows, cols = mask_arr.shape
            sig_pos = 0
            sig_neg = 0
            total_valid = 0

            for r in range(rows):
                for c in range(cols):
                    if not mask_arr[r, c]:
                        continue
                    series = stack[:, r, c]
                    # Only test pixels with yield in at least half the years
                    if np.sum(series > 0) < len(YEARS_COMPARISON) // 2:
                        continue
                    total_valid += 1
                    slope, pval = trend_test(series)
                    if not np.isnan(pval) and pval < 0.05:
                        if slope > 0:
                            sig_pos += 1
                        else:
                            sig_neg += 1

            results.append({
                'crop': label,
                'scenario': scenario,
                'valid_pixels': total_valid,
                'sig_positive_trend_%': round(100 * sig_pos / total_valid, 1) if total_valid > 0 else 0,
                'sig_negative_trend_%': round(100 * sig_neg / total_valid, 1) if total_valid > 0 else 0,
            })
            print(f"  ✓ {label} {scenario}")

    df = pd.DataFrame(results)
    df.to_csv(f'{out_dir}/pixelwise_trend_results.csv', index=False)
    print(df.to_string(index=False))

# =============================================================================
# ANALYSIS 6 — Cross-crop sensitivity summary
# =============================================================================

def analysis_cross_crop_summary(mask):
    """
    Single summary figure: for each crop type, mean ΔYield and
    mean ΔSuitability class (observed minus counterfactual), 1999–2018.
    """
    print("\n[Analysis 6] Cross-crop sensitivity summary …")
    out_dir = f'{OUT_ROOT}/6_cross_crop'
    make_dirs(out_dir)

    records = []

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        delta_yield   = []
        delta_class   = []

        for year in YEARS_COMPARISON:
            obs_r = load_combined_raw(tag, year, nothaw=False)
            cf_r  = load_combined_raw(tag, year, nothaw=True)
            obs_c = load_combined_class(tag, year, nothaw=False)
            cf_c  = load_combined_class(tag, year, nothaw=True)

            if obs_r is not None and cf_r is not None:
                diff = obs_r - cf_r
                delta_yield.append(spatial_mean(diff, mask))

            if obs_c is not None and cf_c is not None:
                diff_c = obs_c - cf_c
                delta_class.append(spatial_mean(diff_c, mask))

        records.append({
            'crop'              : label,
            'mean_delta_yield'  : np.nanmean(delta_yield)  if delta_yield  else np.nan,
            'mean_delta_class'  : np.nanmean(delta_class)  if delta_class  else np.nan,
        })

    df = pd.DataFrame(records).sort_values('mean_delta_yield', ascending=False)
    df.to_csv(f'{out_dir}/cross_crop_sensitivity.csv', index=False)

    # Figure: two horizontal bar charts side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors_y = ['#2166AC' if v >= 0 else '#D6604D' for v in df['mean_delta_yield']]
    colors_c = ['#2166AC' if v >= 0 else '#D6604D' for v in df['mean_delta_class']]

    axes[0].barh(df['crop'], df['mean_delta_yield'], color=colors_y, edgecolor='white')
    axes[0].axvline(0, color='black', linewidth=0.8)
    axes[0].set_xlabel('Mean ΔYield (kg/ha)', fontsize=12)
    axes[0].set_title('Permafrost Thaw Impact\non Yield (Obs − No-Thaw)', fontsize=12, fontweight='bold')

    axes[1].barh(df['crop'], df['mean_delta_class'], color=colors_c, edgecolor='white')
    axes[1].axvline(0, color='black', linewidth=0.8)
    axes[1].set_xlabel('Mean ΔSuitability Class', fontsize=12)
    axes[1].set_title('Permafrost Thaw Impact\non Suitability Class', fontsize=12, fontweight='bold')

    fig.suptitle('Cross-Crop Sensitivity to Permafrost Thaw (Mean 1999–2018)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/cross_crop_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Cross-crop summary saved.")
    print(df.to_string(index=False))

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("Loading static inputs …")
    mask      = load_mask().astype(bool)
    elevation = np.load(ELEV_PATH)

    analysis_spatial_delta(mask, elevation)
    analysis_time_series(mask)
    analysis_transitions(mask)
    analysis_elevation(mask, elevation)
    analysis_trends(mask)
    analysis_trends_pixelwise(mask)
    analysis_cross_crop_summary(mask)

    print(f"\n\nAll analyses complete. Outputs in: {OUT_ROOT}/")