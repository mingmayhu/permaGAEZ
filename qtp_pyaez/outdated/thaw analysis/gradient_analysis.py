"""
Gradient Trend Analysis — Does permafrost thaw impact scale with climate?
==========================================================================
Tests whether ΔYield (permafrost thaw impact) systematically increases or
decreases as ΔTemperature and ΔPrecipitation increase, using continuous
gradient binning rather than the 4-quadrant binary split.

Analyses:
  1. 2D climate-bin heatmap     — mean ΔYield in a 10×10 ΔTemp × ΔPrecip grid
  2. Marginal gradient lines    — mean ΔYield at each ΔTemp percentile (and ΔPrecip)
  3. Spearman rank correlation  — monotonic trend between ΔClimate and ΔYield
  4. Partial correlation        — ΔYield ~ ΔTemp controlling for ΔPrecip (and vice versa)
  5. Binned OLS slope           — does the slope steepen at climate extremes?

Outputs written to: ./thaw_analysis_output/8_gradient_trends/
"""

# =============================================================================
# CONFIGURATION — mirrors your existing scripts
# =============================================================================

WORK_DIR  = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH = r'./data_input/qilian mask.tif'

CLIMATE = {
    'temperature': {
        'period1': r'./data_input/climate_20yr_averages/1979-1998/TempMean.npy',  # °C
        'period2': r'./data_input/climate_20yr_averages/1999-2018/TempMean.npy',  # °C
        'label'  : 'ΔTemperature (°C)',
        'short'  : 'ΔTemp',
        'color'  : '#D6604D',
    },
    'precipitation': {
        'period1': r'./data_input/climate_20yr_averages/1979-1998/Precip.npy',  # mm
        'period2': r'./data_input/climate_20yr_averages/1999-2018/Precip.npy',  # mm
        'label'  : 'ΔPrecipitation (mm)',
        'short'  : 'ΔPrecip',
        'color'  : '#4393C3',
    },
}

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

N_BINS   = 10   # number of percentile bins per climate axis
OUT_DIR  = './thaw_analysis_output/8_gradient_trends'

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
from scipy import stats
from scipy.ndimage import zoom

try:
    from osgeo import gdal
except ImportError:
    import gdal

os.chdir(WORK_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# HELPERS  (same as your existing scripts)
# =============================================================================

def load_raster(path):
    ds = gdal.Open(path)
    if ds is None:
        return None
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    arr = ds.ReadAsArray().astype(float)
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[arr < -1e10] = np.nan
    return arr

def load_mask():
    return load_raster(MASK_PATH).astype(bool)

def reduce_to_2d(arr, var_name):
    """Collapse 3D (rows, cols, days) or (days, rows, cols) → 2D mean/sum."""
    if arr.ndim == 2:
        return arr
    time_axis = int(np.argmax(arr.shape))
    arr_t = np.moveaxis(arr, time_axis, 0)
    is_precip = 'precip' in var_name.lower() or 'rain' in var_name.lower()
    return np.nansum(arr_t, axis=0) if is_precip else np.nanmean(arr_t, axis=0)

def match_grid(arr, target_shape):
    if arr.shape == target_shape:
        return arr
    zy = target_shape[0] / arr.shape[0]
    zx = target_shape[1] / arr.shape[1]
    return zoom(arr, (zy, zx), order=1)

def load_delta_yield(tag, mask):
    years = list(range(1999, 2019))
    obs_stack, cf_stack = [], []
    for year in years:
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
    mean_obs = np.nanmean(obs_stack, axis=0)
    mean_cf  = np.nanmean(cf_stack,  axis=0)
    delta = np.where(np.isfinite(mean_obs) & np.isfinite(mean_cf),
                     mean_obs - mean_cf, np.nan)
    delta[~mask] = np.nan
    return delta

def load_climate(mask_shape):
    """Load and reduce climate arrays to 2D, resample to mask grid if needed."""
    out = {}
    for var, cfg in CLIMATE.items():
        p1 = reduce_to_2d(np.load(cfg['period1']).astype(float), var)
        p2 = reduce_to_2d(np.load(cfg['period2']).astype(float), var)
        p1 = match_grid(p1, mask_shape)
        p2 = match_grid(p2, mask_shape)
        out[var] = {**cfg, 'delta': p2 - p1}
    return out

def pixel_df(mask, climate, delta_yield):
    """Flatten all arrays to a tidy pixel-level DataFrame."""
    valid = mask & np.isfinite(delta_yield)
    for c in climate.values():
        valid &= np.isfinite(c['delta'])
    rows = {'delta_yield': delta_yield[valid]}
    for var, c in climate.items():
        rows[var] = c['delta'][valid]
    return pd.DataFrame(rows)

# =============================================================================
# ANALYSIS 1 — Marginal gradient: mean ΔYield by climate percentile bin
# =============================================================================

def marginal_gradient(df, climate, n_bins, crop_label, out_dir):
    """
    For each climate variable independently, bin pixels into N percentile
    buckets and compute mean ΔYield per bucket.  Plots the trend line and
    fits a linear regression to test whether the gradient is significant.
    Returns a summary dict.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'{crop_label} — ΔYield along climate gradients',
                 fontsize=13, fontweight='bold')
    results = []

    for ax, (var, cfg) in zip(axes, climate.items()):
        x_all = df[var].values
        y_all = df['delta_yield'].values

        # Percentile bin edges
        edges = np.percentile(x_all, np.linspace(0, 100, n_bins + 1))
        edges[-1] += 1e-9   # ensure last point is included

        bin_mids, bin_means, bin_stds, bin_ns = [], [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (x_all >= lo) & (x_all < hi)
            if sel.sum() < 3:
                continue
            bin_mids.append((lo + hi) / 2)
            bin_means.append(np.nanmean(y_all[sel]))
            bin_stds.append(np.nanstd(y_all[sel]))
            bin_ns.append(sel.sum())

        bin_mids  = np.array(bin_mids)
        bin_means = np.array(bin_means)
        bin_stds  = np.array(bin_stds)
        bin_ns    = np.array(bin_ns)

        # OLS trend across bins
        slope, intercept, r, p, se = stats.linregress(bin_mids, bin_means)
        spearman_r, spearman_p = stats.spearmanr(x_all, y_all)

        # Plot
        ax.fill_between(bin_mids, bin_means - bin_stds, bin_means + bin_stds,
                        alpha=0.15, color=cfg['color'], label='±1 SD')
        ax.plot(bin_mids, bin_means, 'o-', color=cfg['color'],
                linewidth=2, markersize=5, label='Bin mean ΔYield')
        x_line = np.linspace(bin_mids.min(), bin_mids.max(), 200)
        ax.plot(x_line, slope * x_line + intercept, '--', color='#333',
                linewidth=1.4,
                label=f'Trend: slope={slope:.3f} kg/ha/unit\n'
                      f'Spearman r={spearman_r:.3f} (p={spearman_p:.3f})')
        ax.axhline(0, color='grey', lw=0.8, linestyle=':')
        ax.set_xlabel(cfg['label'], fontsize=11)
        ax.set_ylabel('Mean ΔYield (kg/ha)', fontsize=11)
        ax.set_title(f'ΔYield vs {cfg["short"]}', fontsize=11)
        ax.legend(fontsize=8, loc='best')

        results.append({
            'crop'          : crop_label,
            'climate_var'   : var,
            'ols_slope'     : round(slope, 5),
            'ols_r'         : round(r,     4),
            'ols_p'         : round(p,     5),
            'spearman_r'    : round(spearman_r, 4),
            'spearman_p'    : round(spearman_p, 5),
            'n_pixels'      : len(x_all),
            'interpretation': _interpret(slope, spearman_r, spearman_p, var),
        })

    plt.tight_layout()
    tag = crop_label.lower().replace(' ', '_')
    plt.savefig(f'{out_dir}/{tag}_gradient_lines.png', dpi=150, bbox_inches='tight')
    plt.close()
    return results


def _interpret(slope, r, p, var):
    """Plain-language interpretation of a gradient trend."""
    sig  = 'significant' if p < 0.05 else 'not significant'
    dir_ = 'increases' if slope > 0 else 'decreases'
    mag  = 'strongly' if abs(r) > 0.5 else ('moderately' if abs(r) > 0.3 else 'weakly')
    return (f'ΔYield {dir_} {mag} as {var} rises ({sig}, r={r:.3f})')


# =============================================================================
# ANALYSIS 2 — 2D climate-bin heatmap (ΔTemp × ΔPrecip grid)
# =============================================================================

def heatmap_2d(df, climate, n_bins, crop_label, out_dir):
    """
    Divide ΔTemp and ΔPrecip each into N percentile bins.
    Compute mean ΔYield in each (ΔTemp bin, ΔPrecip bin) cell.
    Produces a heatmap showing the joint gradient surface.
    """
    vars_ = list(climate.keys())
    x_var, y_var = vars_[0], vars_[1]   # temp on X, precip on Y
    x_cfg, y_cfg = climate[x_var], climate[y_var]

    x = df[x_var].values
    y = df[y_var].values
    z = df['delta_yield'].values

    x_edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
    y_edges = np.percentile(y, np.linspace(0, 100, n_bins + 1))
    x_edges[-1] += 1e-9
    y_edges[-1] += 1e-9

    grid      = np.full((n_bins, n_bins), np.nan)
    grid_n    = np.zeros((n_bins, n_bins), dtype=int)
    x_mids    = (x_edges[:-1] + x_edges[1:]) / 2
    y_mids    = (y_edges[:-1] + y_edges[1:]) / 2

    for i, (xl, xh) in enumerate(zip(x_edges[:-1], x_edges[1:])):
        for j, (yl, yh) in enumerate(zip(y_edges[:-1], y_edges[1:])):
            sel = (x >= xl) & (x < xh) & (y >= yl) & (y < yh)
            if sel.sum() >= 3:
                grid[j, i]   = np.nanmean(z[sel])
                grid_n[j, i] = sel.sum()

    vlim = np.nanpercentile(np.abs(grid[np.isfinite(grid)]), 95) if np.any(np.isfinite(grid)) else 1

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle(f'{crop_label} — ΔYield across climate-change gradient space',
                 fontsize=13, fontweight='bold')

    # Left: mean ΔYield heatmap
    im = axes[0].imshow(grid, aspect='auto', origin='lower',
                        cmap='RdBu', vmin=-vlim, vmax=vlim,
                        extent=[x_edges[0], x_edges[-1],
                                y_edges[0], y_edges[-1]])
    plt.colorbar(im, ax=axes[0], label='Mean ΔYield (kg/ha)')
    axes[0].set_xlabel(x_cfg['label'], fontsize=11)
    axes[0].set_ylabel(y_cfg['label'], fontsize=11)
    axes[0].set_title('Mean ΔYield per climate bin', fontsize=11)
    axes[0].axvline(0, color='black', lw=0.8, linestyle='--', alpha=0.5)
    axes[0].axhline(0, color='black', lw=0.8, linestyle='--', alpha=0.5)

    # Annotate cells with value (only if ≥3 pixels)
    x_step = (x_edges[-1] - x_edges[0]) / n_bins
    y_step = (y_edges[-1] - y_edges[0]) / n_bins
    for i in range(n_bins):
        for j in range(n_bins):
            if np.isfinite(grid[j, i]) and grid_n[j, i] >= 3:
                axes[0].text(x_mids[i], y_mids[j],
                             f'{grid[j,i]:.1f}',
                             ha='center', va='center', fontsize=7,
                             color='white' if abs(grid[j,i]) > vlim * 0.6 else 'black')

    # Right: pixel count heatmap (transparency guide)
    im2 = axes[1].imshow(grid_n, aspect='auto', origin='lower',
                         cmap='Blues',
                         extent=[x_edges[0], x_edges[-1],
                                 y_edges[0], y_edges[-1]])
    plt.colorbar(im2, ax=axes[1], label='Pixel count')
    axes[1].set_xlabel(x_cfg['label'], fontsize=11)
    axes[1].set_ylabel(y_cfg['label'], fontsize=11)
    axes[1].set_title('Pixel count per bin\n(low n = unreliable)', fontsize=11)
    axes[1].axvline(0, color='black', lw=0.8, linestyle='--', alpha=0.5)
    axes[1].axhline(0, color='black', lw=0.8, linestyle='--', alpha=0.5)

    plt.tight_layout()
    tag = crop_label.lower().replace(' ', '_')
    plt.savefig(f'{out_dir}/{tag}_2d_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {crop_label} heatmap saved')


# =============================================================================
# ANALYSIS 3 — Partial correlations  (ΔYield ~ ΔTemp | ΔPrecip, and vice versa)
# =============================================================================

def partial_correlation(df, climate):
    """
    Pearson partial correlation:
      pcorr(ΔYield, ΔTemp)   controlling for ΔPrecip
      pcorr(ΔYield, ΔPrecip) controlling for ΔTemp

    Uses the residual method: regress both target and control on the
    third variable, correlate the residuals.
    """
    vars_  = list(climate.keys())
    v0, v1 = vars_[0], vars_[1]
    y  = df['delta_yield'].values
    x0 = df[v0].values
    x1 = df[v1].values

    def resid(a, b):
        s, i, *_ = stats.linregress(b, a)
        return a - (s * b + i)

    # partial(y, x0 | x1)
    ry_0  = resid(y,  x1)
    rx0_1 = resid(x0, x1)
    r0, p0 = stats.pearsonr(ry_0, rx0_1)

    # partial(y, x1 | x0)
    ry_1  = resid(y,  x0)
    rx1_0 = resid(x1, x0)
    r1, p1 = stats.pearsonr(ry_1, rx1_0)

    return {
        f'partial_r_yield_{v0}_ctrl_{v1}': round(r0, 4),
        f'partial_p_yield_{v0}_ctrl_{v1}': round(p0, 5),
        f'partial_r_yield_{v1}_ctrl_{v0}': round(r1, 4),
        f'partial_p_yield_{v1}_ctrl_{v0}': round(p1, 5),
    }


# =============================================================================
# ANALYSIS 4 — Does the gradient steepen at climate extremes?
#              Split pixels into terciles of ΔTemp (low / mid / high) and
#              recompute the ΔYield ~ ΔPrecip slope within each tercile.
# =============================================================================

def slope_by_tercile(df, climate, crop_label, out_dir):
    """
    Tests for a non-linear / threshold effect:
    Does the ΔYield vs ΔPrecip slope change depending on how warm it is?
    And vice versa?
    """
    vars_ = list(climate.keys())
    v_temp, v_precip = vars_[0], vars_[1]
    t_cfg, p_cfg = climate[v_temp], climate[v_precip]

    results = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'{crop_label} — Climate gradient slope by tercile',
                 fontsize=13, fontweight='bold')

    pairs = [
        (v_temp,   t_cfg,   v_precip, p_cfg,   axes[0]),  # ΔYield~ΔTemp, split by ΔPrecip
        (v_precip, p_cfg,   v_temp,   t_cfg,   axes[1]),  # ΔYield~ΔPrecip, split by ΔTemp
    ]

    colors_tercile = ['#4393C3', '#2CA25F', '#D6604D']

    for x_var, x_cfg, split_var, split_cfg, ax in pairs:
        split_vals = df[split_var].values
        t33, t66   = np.percentile(split_vals, [33, 67])
        tercile_masks = [
            split_vals < t33,
            (split_vals >= t33) & (split_vals < t66),
            split_vals >= t66,
        ]
        tercile_labels = [
            f'Low {split_cfg["short"]} (<P33)',
            f'Mid {split_cfg["short"]} (P33–P67)',
            f'High {split_cfg["short"]} (>P67)',
        ]

        for mask_t, tlabel, color in zip(tercile_masks, tercile_labels, colors_tercile):
            sub = df[mask_t]
            if len(sub) < 10:
                continue
            x = sub[x_var].values
            y = sub['delta_yield'].values
            slope, intercept, r, p, _ = stats.linregress(x, y)

            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, color=color,
                    linewidth=2,
                    label=f'{tlabel}: slope={slope:.3f}, r={r:.3f} (p={p:.3f})')
            ax.scatter(x, y, color=color, alpha=0.08, s=4, rasterized=True)

            results.append({
                'crop'       : crop_label,
                'x_var'      : x_var,
                'split_var'  : split_var,
                'tercile'    : tlabel,
                'slope'      : round(slope, 5),
                'r'          : round(r, 4),
                'p'          : round(p, 5),
                'n'          : len(sub),
            })

        ax.axhline(0, color='grey', lw=0.8, linestyle=':')
        ax.set_xlabel(x_cfg['label'], fontsize=11)
        ax.set_ylabel('ΔYield (kg/ha)', fontsize=11)
        ax.set_title(f'ΔYield ~ {x_cfg["short"]}\nsplit by {split_cfg["short"]} tercile', fontsize=10)
        ax.legend(fontsize=8, loc='best')

    plt.tight_layout()
    tag = crop_label.lower().replace(' ', '_')
    plt.savefig(f'{out_dir}/{tag}_tercile_slopes.png', dpi=150, bbox_inches='tight')
    plt.close()
    return results


# =============================================================================
# CROSS-CROP SUMMARY  — gradient slopes for all crops in one figure
# =============================================================================

def cross_crop_gradient_summary(all_marginal_results, out_dir):
    """
    Bar chart: OLS slope (ΔYield per unit ΔClimate) for every crop,
    coloured by significance.
    """
    df = pd.DataFrame(all_marginal_results)
    df.to_csv(f'{out_dir}/gradient_slope_summary.csv', index=False)

    vars_ = df['climate_var'].unique()
    fig, axes = plt.subplots(1, len(vars_), figsize=(7 * len(vars_), 6))
    if len(vars_) == 1:
        axes = [axes]

    fig.suptitle('ΔYield gradient slope by crop — does thaw impact scale with climate?',
                 fontsize=13, fontweight='bold')

    for ax, var in zip(axes, vars_):
        sub = df[df['climate_var'] == var].sort_values('ols_slope', ascending=True)
        colors = ['#D6604D' if p < 0.05 else '#BABABA' for p in sub['ols_p']]
        bars   = ax.barh(sub['crop'], sub['ols_slope'], color=colors, edgecolor='white')
        ax.axvline(0, color='black', lw=0.8)
        ax.set_xlabel('OLS slope (kg/ha per unit Δclimate)', fontsize=11)
        ax.set_title(f'ΔYield ~ Δ{var.capitalize()}\nslope per crop', fontsize=11)

        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color='#D6604D', label='p < 0.05'),
            Patch(color='#BABABA', label='p ≥ 0.05'),
        ], fontsize=9)

        # Annotate spearman r
        for bar, (_, row) in zip(bars, sub.iterrows()):
            ax.text(bar.get_width() + 0.0002 * np.sign(bar.get_width() or 1),
                    bar.get_y() + bar.get_height() / 2,
                    f'r={row["spearman_r"]:.2f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{out_dir}/ALL_CROPS_gradient_slopes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ Cross-crop slope summary saved')


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print('=' * 60)
    print('Gradient Trend Analysis — ΔYield vs ΔClimate')
    print('=' * 60)

    mask    = load_mask().astype(bool)
    climate = load_climate(mask.shape)

    all_marginal  = []
    all_partial   = []
    all_tercile   = []

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        print(f'\n── {label} ──')

        dy = load_delta_yield(tag, mask)
        if dy is None:
            print(f'  ⚠ No yield data, skipping.')
            continue

        df = pixel_df(mask, climate, dy)
        if len(df) < 20:
            print(f'  ⚠ Too few valid pixels ({len(df)}), skipping.')
            continue

        print(f'  Valid pixels: {len(df):,}')

        # 1. Marginal gradient lines
        marg = marginal_gradient(df, climate, N_BINS, label, OUT_DIR)
        all_marginal.extend(marg)
        for m in marg:
            print(f'  {m["climate_var"]:15s} → {m["interpretation"]}')

        # 2. 2D heatmap
        heatmap_2d(df, climate, N_BINS, label, OUT_DIR)

        # 3. Partial correlations
        pc = partial_correlation(df, climate)
        pc['crop'] = label
        all_partial.append(pc)

        # 4. Tercile slope analysis
        terc = slope_by_tercile(df, climate, label, OUT_DIR)
        all_tercile.extend(terc)

    # Cross-crop summary
    print('\n── Cross-crop summary ──')
    cross_crop_gradient_summary(all_marginal, OUT_DIR)

    # Save all tables
    pd.DataFrame(all_partial).to_csv(f'{OUT_DIR}/partial_correlations.csv', index=False)
    pd.DataFrame(all_tercile).to_csv(f'{OUT_DIR}/tercile_slopes.csv', index=False)

    print(f'\n✅ All gradient analyses complete → {OUT_DIR}/')

    # Print partial correlation summary
    pc_df = pd.DataFrame(all_partial)
    print('\nPartial correlations (controlling for the other climate variable):')
    print(pc_df.to_string(index=False))