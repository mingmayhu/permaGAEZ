"""
ΔALT vs ΔYield Scatter — Nonlinear Threshold Analysis
======================================================
Plots ΔALT (active layer deepening vs baseline) against ΔYield
(observed - no-thaw) for all 10 crops in one figure.

Fits both a linear and quadratic (inverted-U) curve per crop to test
whether thaw has a threshold effect — beneficial up to a point,
harmful beyond it.

Outputs: ./thaw_analysis_output/13_alt_dyield_scatter/
"""

# =============================================================================
# CONFIGURATION
# =============================================================================

WORK_DIR  = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH = r'./data_input/qilian mask.tif'
ELEV_PATH = r'./data_input/terrain/elevation.npy'

ALT_PATH_PATTERN = r'./data_input/permafrost_yearly/{year}/active_layer_depth.npy'

YEARS_BASELINE   = list(range(1979, 1999))
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

OUT_DIR = './thaw_analysis_output/13_alt_dyield_scatter'

# =============================================================================
# IMPORTS
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from scipy.ndimage import zoom
from scipy.optimize import curve_fit

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

def load_delta_alt(mask):
    """Compute ΔALT = mean ALT (1999-2018) minus mean ALT (1979-1998)."""
    print('Loading ALT files...')

    def _load_period(years):
        stack = []
        for year in years:
            path = ALT_PATH_PATTERN.format(year=year)
            try:
                arr = np.load(path).astype(float)
            except FileNotFoundError:
                continue
            if arr.ndim == 3:
                time_ax = int(np.argmax(arr.shape))
                arr = np.moveaxis(arr, time_ax, 0)
                arr = np.nanmean(arr, axis=0)
            arr = match_grid(arr, mask.shape)
            arr[~mask]    = np.nan
            arr[arr <= 0] = np.nan
            stack.append(arr)
        return np.nanmean(stack, axis=0) if stack else None

    alt_base = _load_period(YEARS_BASELINE)
    alt_comp = _load_period(YEARS_COMPARISON)

    if alt_base is None or alt_comp is None:
        raise RuntimeError('Could not load ALT files.')

    delta = np.where(np.isfinite(alt_base) & np.isfinite(alt_comp),
                     alt_comp - alt_base, np.nan)
    delta[~mask] = np.nan
    print(f'  ΔALT loaded. Mean={np.nanmean(delta):.3f} m  '
          f'Range=[{np.nanmin(delta):.3f}, {np.nanmax(delta):.3f}]')
    return delta

def load_mean_dyield(tag, mask):
    """Mean ΔYield (observed - no-thaw) per pixel over 1999-2018."""
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
    diff = np.nanmean(obs_stack, axis=0) - np.nanmean(cf_stack, axis=0)
    diff[~mask] = np.nan
    return diff

# =============================================================================
# BINNED MEANS — for visualising the nonlinear trend clearly
# =============================================================================

def compute_binned_means(x, y, n_bins=15):
    """
    Bin x into n_bins quantile bins, compute mean y per bin.
    Returns (bin_centers, bin_means, bin_stds, bin_counts).
    """
    valid = np.isfinite(x) & np.isfinite(y)
    x_v, y_v = x[valid], y[valid]
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges     = np.nanpercentile(x_v, quantiles)
    # remove duplicate edges
    edges = np.unique(edges)

    centers, means, stds, counts = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (x_v >= lo) & (x_v < hi)
        if in_bin.sum() < 3:
            continue
        centers.append((lo + hi) / 2)
        means.append(np.nanmean(y_v[in_bin]))
        stds.append(np.nanstd(y_v[in_bin]))
        counts.append(int(in_bin.sum()))

    return np.array(centers), np.array(means), np.array(stds), np.array(counts)

# =============================================================================
# QUADRATIC FIT — tests for inverted-U threshold effect
# =============================================================================

def fit_quadratic(x, y):
    """
    Fit y = a*x^2 + b*x + c.
    Returns (coeffs, r2, threshold_x) where threshold_x is the x at peak/trough.
    """
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 10:
        return None, None, None
    coeffs = np.polyfit(x[valid], y[valid], 2)
    y_pred = np.polyval(coeffs, x[valid])
    ss_res = np.sum((y[valid] - y_pred) ** 2)
    ss_tot = np.sum((y[valid] - np.mean(y[valid])) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    # vertex of parabola = -b / 2a
    threshold = -coeffs[1] / (2 * coeffs[0]) if coeffs[0] != 0 else np.nan
    return coeffs, r2, threshold

# =============================================================================
# MAIN PLOT — all 10 crops in one figure
# =============================================================================

def plot_all_crops(delta_alt, mask, elevation):
    print('\nBuilding scatter figure...')

    n_crops = len(CROPS)
    ncols   = 5
    nrows   = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4.5))
    fig.suptitle('ΔALT (Active Layer Deepening) vs ΔYield (Observed − No-Thaw)\n'
                 'Testing for Threshold Effect of Permafrost Thaw',
                 fontsize=14, fontweight='bold')
    axes = axes.flatten()

    summary_records = []

    for i, crop in enumerate(CROPS):
        tag, label = crop['tag'], crop['label']
        ax = axes[i]

        dy = load_mean_dyield(tag, mask)
        if dy is None:
            ax.set_title(f'{label}\n(no data)')
            ax.axis('off')
            continue

        # Flatten to 1D valid pixels
        valid = (mask & np.isfinite(delta_alt) & np.isfinite(dy) &
                 np.isfinite(elevation))
        x = delta_alt[valid]
        y = dy[valid]
        e = elevation[valid]

        if len(x) < 10:
            ax.set_title(f'{label}\n(insufficient data)')
            continue

        # Scatter coloured by elevation
        sc = ax.scatter(x, y, c=e, cmap='terrain_r', alpha=0.3, s=5,
                        rasterized=True, vmin=e.min(), vmax=e.max())

        # Binned means with error bars — shows the nonlinear trend clearly
        bx, by, bstd, bn = compute_binned_means(x, y, n_bins=12)
        ax.errorbar(bx, by, yerr=bstd, fmt='o', color='black',
                    markersize=5, linewidth=1.5, capsize=3,
                    label='Binned mean ± 1SD', zorder=5)
        ax.plot(bx, by, color='black', linewidth=1.5, zorder=4)

        # Quadratic fit
        coeffs, r2_quad, threshold = fit_quadratic(x, y)
        if coeffs is not None:
            x_line = np.linspace(x.min(), x.max(), 300)
            y_quad  = np.polyval(coeffs, x_line)
            ax.plot(x_line, y_quad, color='#d73027', linewidth=2,
                    linestyle='--', label=f'Quadratic fit (R²={r2_quad:.2f})',
                    zorder=6)
            # Mark threshold if it falls within data range
            if x.min() < threshold < x.max():
                ax.axvline(threshold, color='#d73027', linewidth=1,
                           linestyle=':', alpha=0.7)
                ax.text(threshold, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else y.min(),
                        f' peak\n {threshold:.2f}m',
                        color='#d73027', fontsize=7, va='bottom')

        # Linear fit for comparison
        slope, intercept, r, p, _ = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_line, slope * x_line + intercept, color='#4575b4',
                linewidth=1.5, linestyle='-',
                label=f'Linear (r={r:.2f}, p={p:.3f})', zorder=3)

        ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
        ax.axvline(0, color='grey', linewidth=0.8, linestyle='--')
        ax.set_xlabel('ΔALT (m)', fontsize=9)
        ax.set_ylabel('ΔYield (kg/ha)', fontsize=9)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, loc='upper right')

        plt.colorbar(sc, ax=ax, shrink=0.7, label='Elev (m)',
                     pad=0.02).ax.tick_params(labelsize=7)

        summary_records.append({
            'crop'           : label,
            'n_pixels'       : int(len(x)),
            'linear_r'       : round(float(r), 4),
            'linear_p'       : round(float(p), 4),
            'quad_r2'        : round(float(r2_quad), 4) if r2_quad else np.nan,
            'quad_threshold_m': round(float(threshold), 3) if threshold and
                                x.min() < threshold < x.max() else np.nan,
            'concave_down'   : bool(coeffs[0] < 0) if coeffs is not None else np.nan,
        })

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/all_crops_alt_dyield_scatter.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: all_crops_alt_dyield_scatter.png')

    # Save summary
    df = pd.DataFrame(summary_records)
    df.to_csv(f'{OUT_DIR}/threshold_analysis_summary.csv', index=False)
    print('\nThreshold analysis summary:')
    print(df.to_string(index=False))
    return df

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print('=' * 60)
    print('ΔALT vs ΔYield Scatter — Threshold Analysis')
    print('=' * 60)

    mask      = load_mask().astype(bool)
    elevation = match_grid(np.load(ELEV_PATH), mask.shape)
    delta_alt = load_delta_alt(mask)

    plot_all_crops(delta_alt, mask, elevation)

    print(f'\nAll outputs written to: {OUT_DIR}/')