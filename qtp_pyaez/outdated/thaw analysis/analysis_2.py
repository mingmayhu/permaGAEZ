"""
Climate Correlation Analysis — Extension to Permafrost Thaw Impact Analysis
=============================================================================
Correlates permafrost thaw ΔYield (from existing analysis outputs) with
temperature and precipitation across two 20-year periods:
  - Period 1 (baseline):    1979–1998
  - Period 2 (comparison):  1999–2018

Climate inputs expected as 2D numpy arrays (.npy) matching the spatial
extent/resolution of your yield rasters.

Analyses:
  1. Spatial overlay maps   — ΔYield overlaid with climate anomaly (ΔTemp, ΔPrecip)
  2. Pixel-level correlation — ΔYield vs ΔTemp and ΔYield vs ΔPrecip per crop
  3. Hotspot classification — quadrant analysis (warm/cool × wet/dry) vs ΔYield
  4. Cross-crop climate sensitivity summary

Outputs written to: ./thaw_analysis_output/7_climate_correlation/
"""

# =============================================================================
# CONFIGURATION — mirror your existing script's settings
# =============================================================================

WORK_DIR  = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH = r'./data_input/qilian mask.tif'

# ── Climate numpy arrays ──────────────────────────────────────────────────────
# Each should be a 2D array (rows × cols) matching your raster grid.
# Provide either period means OR pre-computed anomaly arrays — see LOAD section.
CLIMATE = {
    'temperature': {
        'period1': r'./data_input/climate_20yr_averages/1979-1998/TempMean.npy',  # °C
        'period2': r'./data_input/climate_20yr_averages/1999-2018/TempMean.npy',  # °C
        'label'  : 'Temperature (°C)',
        'color'  : '#D6604D',
    },
    'precipitation': {
        'period1': r'./data_input/climate_20yr_averages/1979-1998/Precip.npy',  # mm
        'period2': r'./data_input/climate_20yr_averages/1999-2018/Precip.npy',  # mm
        'label'  : 'Precipitation (mm)',
        'color'  : '#4393C3',
    },
}

# ── Existing analysis output root (from your original script) ─────────────────
ANALYSIS_ROOT = './thaw_analysis_output'
OUT_DIR       = f'{ANALYSIS_ROOT}/7_climate_correlation'

# Crop definitions — must match your original CROPS list
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

# =============================================================================
# IMPORTS
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy import stats
from pathlib import Path

try:
    from osgeo import gdal
except ImportError:
    import gdal

try:
    import libpysal.weights as lps_weights
    from esda.moran import Moran_Local
    HAS_ESDA = True
except ImportError:
    HAS_ESDA = False
    warnings.warn("esda/libpysal not installed — hotspot map uses z-score fallback.")

os.chdir(WORK_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# HELPERS (mirrors your original script)
# =============================================================================

def load_raster(path):
    ds = gdal.Open(path)
    if ds is None:
        return None
    band = ds.GetRasterBand(1)
    nodata_val = band.GetNoDataValue()
    arr = ds.ReadAsArray().astype(float)
    if nodata_val is not None:
        arr[arr == nodata_val] = np.nan
    arr[arr < -1e10] = np.nan
    return arr


def load_mask():
    arr = load_raster(MASK_PATH)
    return arr.astype(bool) if arr is not None else None


def load_delta_yield(tag, mask):
    """
    Re-derives ΔYield (mean obs − mean no-thaw, 1999–2018) directly from
    rasters so this script is self-contained. Mirrors Analysis 1 logic.
    """
    years = list(range(1999, 2019))
    obs_stack, cf_stack = [], []

    for year in years:
        obs_path = f'./data_output/final_classification/{tag}/{year}_raw_yield.tif'
        cf_path  = f'./data_output/final_classification_nothaw/{tag}/{year}_raw_yield.tif'
        obs = load_raster(obs_path)
        cf  = load_raster(cf_path)
        if obs is not None:
            obs[~mask] = np.nan
            obs_stack.append(obs)
        if cf is not None:
            cf[~mask] = np.nan
            cf_stack.append(cf)

    if not obs_stack or not cf_stack:
        return None

    mean_obs = np.nanmean(obs_stack, axis=0)
    mean_cf  = np.nanmean(cf_stack,  axis=0)
    both_valid = np.isfinite(mean_obs) & np.isfinite(mean_cf)
    delta = np.where(both_valid, mean_obs - mean_cf, np.nan)
    delta[~mask] = np.nan
    return delta


# =============================================================================
# LOAD CLIMATE DATA
# =============================================================================

def reduce_to_2d(arr, var_name):
    """
    Collapses a 3D climate array to 2D (rows × cols).

    Handles two common array layouts:
      (rows, cols, time)  — time is the last axis   → axis=2
      (time, rows, cols)  — time is the first axis  → axis=0

    Aggregation:
      - temperature  → mean across time (mean warming signal)
      - precipitation → sum across time (total precip over period)

    Also computes a growing-season (Apr–Sep, days 90–272) mean/sum
    alongside the annual figure, printing both so you can choose.
    """
    if arr.ndim == 2:
        return arr   # already 2D, nothing to do

    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")

    # Detect which axis is time (the one that is NOT 39 or 93, i.e. rows/cols).
    # Generalised: the spatial dims are the two smallest axes.
    shape = arr.shape
    time_axis = int(np.argmax(shape))   # largest dimension = time

    n_time = shape[time_axis]
    print(f"  {var_name}: shape={shape}, treating axis {time_axis} as time ({n_time} steps)")

    # Move time to axis 0 for consistent slicing
    arr_t = np.moveaxis(arr, time_axis, 0)   # (time, rows, cols)

    # ── Annual aggregation ────────────────────────────────────────────
    is_precip = 'precip' in var_name.lower() or 'rain' in var_name.lower()
    if is_precip:
        annual = np.nansum(arr_t, axis=0)
    else:
        annual = np.nanmean(arr_t, axis=0)

    # ── Growing-season slice (Apr–Sep = days 90–272, 0-indexed) ──────
    # Only meaningful for daily data (365/366 steps); skip for monthly (12).
    if n_time >= 365:
        gs = arr_t[90:273]   # Apr 1 – Sep 30 (approx)
        if is_precip:
            gs_agg = np.nansum(gs, axis=0)
        else:
            gs_agg = np.nanmean(gs, axis=0)
        print(f"    Annual  mean/sum pixel-mean: {np.nanmean(annual):.3f}")
        print(f"    GrowSea mean/sum pixel-mean: {np.nanmean(gs_agg):.3f}")
        print(f"    → Using ANNUAL aggregate for correlation (change to gs_agg for growing-season)")
    else:
        print(f"    Monthly data detected ({n_time} steps) — using full-period mean/sum")

    return annual


def load_climate_arrays():
    """
    Loads period1 and period2 numpy arrays for each climate variable.
    Automatically reduces 3D arrays (rows, cols, days) or (days, rows, cols)
    to 2D before computing the anomaly (period2 − period1).
    Returns dict: { 'temperature': {'p1', 'p2', 'delta'}, 'precipitation': {...} }
    """
    climate_data = {}
    for var, cfg in CLIMATE.items():
        print(f"\nLoading {var} …")
        p1_raw = np.load(cfg['period1']).astype(float)
        p2_raw = np.load(cfg['period2']).astype(float)

        p1 = reduce_to_2d(p1_raw, var)
        p2 = reduce_to_2d(p2_raw, var)

        print(f"  Reduced to: p1={p1.shape}, p2={p2.shape}")

        if p1.shape != p2.shape:
            raise ValueError(f"{var}: period1 shape {p1.shape} != period2 shape {p2.shape}")

        # Warn if climate grid doesn't match mask/yield raster
        mask_shape = None  # resolved at runtime via mask arg; checked in overlay

        climate_data[var] = {
            'p1'   : p1,
            'p2'   : p2,
            'delta': p2 - p1,   # positive = warmer/wetter in 1999–2018 vs baseline
            'label': cfg['label'],
            'color': cfg['color'],
        }
    return climate_data


# =============================================================================
# ANALYSIS 7a — Spatial overlay: ΔYield + climate anomaly per crop
# =============================================================================

def analysis_spatial_overlay(mask, climate_data):
    """
    For each crop: 3-panel figure showing
      left  — ΔYield map
      centre — ΔTemperature map (period2 − period1)
      right  — ΔPrecipitation map
    so the reader can visually compare spatial patterns.
    """
    print("\n[7a] Spatial overlay maps …")
    sub_dir = f'{OUT_DIR}/7a_spatial_overlay'
    os.makedirs(sub_dir, exist_ok=True)

    # Guard: if climate grid doesn't match mask, resize with nearest-neighbour
    def match_grid(arr, target_shape, name):
        if arr.shape == target_shape:
            return arr
        print(f"  ⚠ {name} shape {arr.shape} != mask {target_shape} — resizing with zoom")
        from scipy.ndimage import zoom
        zy = target_shape[0] / arr.shape[0]
        zx = target_shape[1] / arr.shape[1]
        return zoom(arr, (zy, zx), order=1)

    d_temp   = np.where(mask, match_grid(climate_data['temperature']['delta'],   mask.shape, 'temperature'),   np.nan)
    d_precip = np.where(mask, match_grid(climate_data['precipitation']['delta'], mask.shape, 'precipitation'), np.nan)

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        delta = load_delta_yield(tag, mask)
        if delta is None:
            print(f"  ⚠ Skipping {label} — no yield data.")
            continue

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        panels = [
            (delta,    'ΔYield (Obs − No-Thaw)\n[kg/ha]',     'RdBu',     None, None),
            (d_temp,   'ΔTemperature (1999–2018 vs 1979–1998)\n[°C]', 'RdBu_r', None, None),
            (d_precip, 'ΔPrecipitation\n[mm]',                 'BrBG',     None, None),
        ]

        for ax, (arr, title, cmap, vmin, vmax) in zip(axes, panels):
            vlim = np.nanpercentile(np.abs(arr[mask]), 98) if vmin is None else abs(vmin)
            im = ax.imshow(arr, cmap=cmap, vmin=-vlim, vmax=vlim, interpolation='nearest')
            ax.set_title(title, fontsize=11)
            ax.axis('off')
            plt.colorbar(im, ax=ax, shrink=0.75)

        fig.suptitle(f'{label} — ΔYield vs Climate Anomaly', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{sub_dir}/{tag}_spatial_overlay.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {label}")


# =============================================================================
# ANALYSIS 7b — Pixel-level correlation: ΔYield vs ΔClimate
# =============================================================================

def analysis_pixel_correlation(mask, climate_data):
    """
    For each crop × climate variable:
      - Pearson and Spearman correlation at pixel level
      - Scatter plot with regression line, coloured by yield gain/loss
    Outputs a combined summary CSV and per-crop scatter figures.
    """
    print("\n[7b] Pixel-level correlation …")
    sub_dir = f'{OUT_DIR}/7b_correlation'
    os.makedirs(sub_dir, exist_ok=True)

    all_results = []

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        delta_yield = load_delta_yield(tag, mask)
        if delta_yield is None:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'{label} — ΔYield vs Climate Anomaly (pixel level)', fontsize=13, fontweight='bold')

        for ax, (var, c_data) in zip(axes, climate_data.items()):
            d_clim = c_data['delta']

            # Valid pixels: inside mask, both arrays finite, yield nonzero in at least one scenario
            valid = (
                mask
                & np.isfinite(delta_yield)
                & np.isfinite(d_clim)
            )
            y = delta_yield[valid]
            x = d_clim[valid]

            if len(x) < 10:
                ax.set_title(f'Insufficient data for {var}')
                continue

            # Correlations
            pearson_r,  pearson_p  = stats.pearsonr(x, y)
            spearman_r, spearman_p = stats.spearmanr(x, y)

            all_results.append({
                'crop'      : label,
                'variable'  : var,
                'n_pixels'  : int(len(x)),
                'pearson_r' : round(pearson_r,  4),
                'pearson_p' : round(pearson_p,  6),
                'spearman_r': round(spearman_r, 4),
                'spearman_p': round(spearman_p, 6),
            })

            # Scatter (subsample for speed)
            idx = np.random.choice(len(x), min(8000, len(x)), replace=False)
            point_colors = np.where(y[idx] >= 0, '#2166AC', '#D6604D')
            ax.scatter(x[idx], y[idx], c=point_colors, alpha=0.3, s=5, rasterized=True)

            # Regression line
            slope, intercept, *_ = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 200)
            ax.plot(x_line, slope * x_line + intercept, color='black', lw=1.8,
                    label=f'Pearson r={pearson_r:.3f} (p={pearson_p:.2e})\n'
                          f'Spearman r={spearman_r:.3f} (p={spearman_p:.2e})')

            ax.axhline(0, color='grey', lw=0.8, linestyle='--')
            ax.axvline(0, color='grey', lw=0.8, linestyle='--')
            ax.set_xlabel(f'Δ{c_data["label"]}', fontsize=11)
            ax.set_ylabel('ΔYield (kg/ha)', fontsize=11)
            ax.set_title(f'ΔYield vs Δ{var.capitalize()}', fontsize=11)
            ax.legend(fontsize=8, loc='upper left')

            # Colour legend
            ax.legend(handles=[
                Patch(color='#2166AC', label='ΔYield ≥ 0 (gain)'),
                Patch(color='#D6604D', label='ΔYield < 0 (loss)'),
                plt.Line2D([0], [0], color='black', lw=1.8,
                           label=f'Pearson r={pearson_r:.3f} (p={pearson_p:.2e})\n'
                                 f'Spearman r={spearman_r:.3f} (p={spearman_p:.2e})'),
            ], fontsize=8, loc='upper left')

        plt.tight_layout()
        plt.savefig(f'{sub_dir}/{tag}_correlation_scatter.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {label}")

    df = pd.DataFrame(all_results)
    df.to_csv(f'{sub_dir}/correlation_summary.csv', index=False)
    print(f"\n  Correlation summary:\n{df.to_string(index=False)}")
    return df


# =============================================================================
# ANALYSIS 7c — Climate quadrant hotspot map
# =============================================================================

def analysis_climate_quadrants(mask, climate_data, crop):
    """
    Classifies every pixel into one of four climate-change quadrants based on
    whether temperature and precipitation increased or decreased in 1999–2018
    vs 1979–1998, then shows the mean ΔYield in each quadrant for a given crop.

    Quadrants (relative to period-mean threshold):
      Q1 — Warmer & Wetter    Q2 — Warmer & Drier
      Q3 — Cooler & Wetter    Q4 — Cooler & Drier
    """
    tag, label = crop['tag'], crop['label']
    delta_yield = load_delta_yield(tag, mask)
    if delta_yield is None:
        return None

    d_temp   = climate_data['temperature']['delta']
    d_precip = climate_data['precipitation']['delta']

    # Threshold: zero anomaly (warmer/cooler relative to baseline)
    warmer = d_temp   > 0
    wetter = d_precip > 0

    quadrant = np.full(mask.shape, np.nan)
    quadrant[mask & warmer  &  wetter] = 1   # Warmer & Wetter
    quadrant[mask & warmer  & ~wetter] = 2   # Warmer & Drier
    quadrant[mask & ~warmer &  wetter] = 3   # Cooler & Wetter
    quadrant[mask & ~warmer & ~wetter] = 4   # Cooler & Drier

    # Mean ΔYield by quadrant
    quad_labels = {1: 'Warmer\n& Wetter', 2: 'Warmer\n& Drier',
                   3: 'Cooler\n& Wetter', 4: 'Cooler\n& Drier'}
    quad_colors = {1: '#4DAC26', 2: '#D01C8B', 3: '#0571B0', 4: '#CA0020'}

    stats_rows = []
    for q, qlabel in quad_labels.items():
        sel = (quadrant == q) & np.isfinite(delta_yield)
        vals = delta_yield[sel]
        stats_rows.append({
            'quadrant'        : q,
            'label'           : qlabel.replace('\n', ' '),
            'n_pixels'        : len(vals),
            'mean_delta_yield': np.nanmean(vals) if len(vals) > 0 else np.nan,
            'std_delta_yield' : np.nanstd(vals)  if len(vals) > 0 else np.nan,
        })

    return quadrant, stats_rows, quad_colors, quad_labels, delta_yield


def analysis_all_quadrants(mask, climate_data):
    """Run quadrant analysis for all crops and produce summary outputs."""
    print("\n[7c] Climate quadrant hotspot analysis …")
    sub_dir = f'{OUT_DIR}/7c_quadrants'
    os.makedirs(sub_dir, exist_ok=True)

    all_stats = []

    for crop in CROPS:
        result = analysis_climate_quadrants(mask, climate_data, crop)
        if result is None:
            print(f"  ⚠ Skipping {crop['label']} — no yield data.")
            continue

        quadrant, stats_rows, quad_colors, quad_labels, delta_yield = result
        label = crop['label']
        tag   = crop['tag']

        for row in stats_rows:
            row['crop'] = label
        all_stats.extend(stats_rows)

        # Figure: quadrant map + bar chart of mean ΔYield per quadrant
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: spatial quadrant map
        cmap_q = mcolors.ListedColormap(['#4DAC26', '#D01C8B', '#0571B0', '#CA0020'])
        norm_q = mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap_q.N)
        im = axes[0].imshow(quadrant, cmap=cmap_q, norm=norm_q, interpolation='nearest')
        axes[0].axis('off')
        axes[0].set_title('Climate-Change Quadrant Map', fontsize=11)
        legend_patches = [Patch(color=c, label=quad_labels[q].replace('\n', ' '))
                          for q, c in quad_colors.items()]
        axes[0].legend(handles=legend_patches, loc='lower right', fontsize=8)

        # Right: bar chart — mean ΔYield by quadrant
        qs     = [r['quadrant']         for r in stats_rows]
        means  = [r['mean_delta_yield'] for r in stats_rows]
        labels = [quad_labels[q].replace('\n', ' ') for q in qs]
        colors = [quad_colors[q] for q in qs]
        bars   = axes[1].bar(labels, means, color=colors, edgecolor='black', alpha=0.85)
        axes[1].axhline(0, color='black', lw=0.8)
        axes[1].set_ylabel('Mean ΔYield (kg/ha)', fontsize=11)
        axes[1].set_title('Mean ΔYield by Climate Quadrant', fontsize=11)
        axes[1].tick_params(axis='x', labelsize=9)

        # Annotate pixel counts
        for bar, row in zip(bars, stats_rows):
            axes[1].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.5 * np.sign(bar.get_height() or 1),
                         f"n={row['n_pixels']:,}", ha='center', fontsize=8)

        fig.suptitle(f'{label} — ΔYield by Climate-Change Quadrant', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{sub_dir}/{tag}_quadrant_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {label}")

    df = pd.DataFrame(all_stats)
    df.to_csv(f'{sub_dir}/quadrant_summary.csv', index=False)

    # Cross-crop heatmap: mean ΔYield by crop × quadrant
    pivot = df.pivot_table(index='crop', columns='label', values='mean_delta_yield')
    fig, ax = plt.subplots(figsize=(10, 7))
    vlim = np.nanpercentile(np.abs(pivot.values[np.isfinite(pivot.values)]), 98)
    im = ax.imshow(pivot.values, cmap='RdBu', vmin=-vlim, vmax=vlim, aspect='auto')
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_title('Mean ΔYield (kg/ha) by Crop × Climate Quadrant\n(1999–2018 vs 1979–1998)',
                 fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Mean ΔYield (kg/ha)')
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=9,
                        color='white' if abs(val) > vlim * 0.6 else 'black')
    plt.tight_layout()
    plt.savefig(f'{sub_dir}/ALL_CROPS_quadrant_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Cross-crop quadrant heatmap saved.")


# =============================================================================
# ANALYSIS 7d — Cross-crop climate sensitivity summary
# =============================================================================

def analysis_climate_sensitivity_summary(mask, climate_data, corr_df):
    """
    Produces a single summary figure: for each crop, Pearson r values for
    ΔYield ~ ΔTemp and ΔYield ~ ΔPrecip, colour-coded by significance.
    """
    print("\n[7d] Cross-crop climate sensitivity summary …")
    sub_dir = f'{OUT_DIR}/7d_sensitivity'
    os.makedirs(sub_dir, exist_ok=True)

    if corr_df is None or corr_df.empty:
        print("  ⚠ No correlation data available.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Cross-Crop Climate Sensitivity — Pearson r (ΔYield ~ ΔClimate)',
                 fontsize=13, fontweight='bold')

    for ax, var in zip(axes, ['temperature', 'precipitation']):
        sub = corr_df[corr_df['variable'] == var].copy()
        sub = sub.sort_values('pearson_r', ascending=True)
        colors = ['#D6604D' if p < 0.05 else '#BABABA' for p in sub['pearson_p']]
        bars = ax.barh(sub['crop'], sub['pearson_r'], color=colors, edgecolor='white')
        ax.axvline(0, color='black', lw=0.8)
        ax.set_xlabel('Pearson r', fontsize=11)
        ax.set_title(f'ΔYield ~ Δ{var.capitalize()}', fontsize=11)
        ax.set_xlim(-1, 1)

        legend_patches = [
            Patch(color='#D6604D', label='p < 0.05 (significant)'),
            Patch(color='#BABABA', label='p ≥ 0.05'),
        ]
        ax.legend(handles=legend_patches, fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{sub_dir}/cross_crop_climate_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Sensitivity summary saved.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Climate Correlation Analysis — Permafrost Thaw Impact")
    print("=" * 60)

    print("\nLoading mask …")
    mask = load_mask().astype(bool)

    print("\nLoading climate arrays …")
    climate_data = load_climate_arrays()

    # Run analyses
    analysis_spatial_overlay(mask, climate_data)
    corr_df = analysis_pixel_correlation(mask, climate_data)
    analysis_all_quadrants(mask, climate_data)
    analysis_climate_sensitivity_summary(mask, climate_data, corr_df)

    print(f"\n✅ All climate correlation analyses complete.")
    print(f"   Outputs written to: {OUT_DIR}/")