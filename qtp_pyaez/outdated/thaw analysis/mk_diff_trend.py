"""
Mann-Kendall on ΔYield Difference Timeseries
=============================================
Tests whether the *impact* of permafrost thaw on yield is itself
trending significantly over time.

Per pixel, for each crop:
  difference[year] = observed_yield[year] - nothaw_yield[year]
  → Mann-Kendall test on this timeseries

This is more direct than comparing trend proportions between scenarios,
because it explicitly asks: is thaw's effect growing or shrinking?

Outputs:
  - Per-crop maps: trend direction + significance
  - Summary heatmap: % significant positive/negative pixels per crop

Outputs: ./thaw_analysis_output/11_mk_diff_trend/
"""

# =============================================================================
# CONFIGURATION — match your existing scripts
# =============================================================================

WORK_DIR  = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH = r'./data_input/qilian mask.tif'

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

ALPHA   = 0.05
MIN_OBS = 8    # minimum non-nan years to run MK
OUT_DIR = './thaw_analysis_output/11_mk_diff_trend'

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
from matplotlib.patches import Rectangle
import pymannkendall as mk

try:
    from osgeo import gdal
except ImportError:
    import gdal

os.chdir(WORK_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# HELPERS — reuse same pattern as your existing scripts
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

def load_diff_stack(tag, mask):
    """
    Load annual difference timeseries: observed - no-thaw per year.
    Returns array of shape (n_years, rows, cols).
    Years with missing obs or cf data are filled with nan.
    """
    rows, cols = mask.shape
    stack = np.full((len(YEARS_COMPARISON), rows, cols), np.nan)

    for i, year in enumerate(YEARS_COMPARISON):
        obs = load_raster(f'./data_output/final_classification/{tag}/{year}_raw_yield.tif')
        cf  = load_raster(f'./data_output/final_classification_nothaw/{tag}/{year}_raw_yield.tif')
        if obs is None or cf is None:
            continue
        diff = np.where(np.isfinite(obs) & np.isfinite(cf), obs - cf, np.nan)
        diff[~mask] = np.nan
        stack[i] = diff

    return stack

# =============================================================================
# CORE — pixel-wise MK on difference timeseries
# =============================================================================

def run_mk_diff(stack, mask):
    """
    Run Mann-Kendall on the difference timeseries at each pixel.

    Returns two 2D arrays:
      trend_map  : +1 (sig positive), -1 (sig negative), 0 (no sig trend)
      tau_map    : Kendall tau value at each pixel (nan where insufficient data)
    """
    rows, cols = mask.shape
    trend_map  = np.zeros((rows, cols), dtype=float)
    tau_map    = np.full((rows, cols), np.nan)

    active_pixels = list(zip(*np.where(mask)))
    n_total = len(active_pixels)

    for idx, (r, c) in enumerate(active_pixels):
        if idx % 5000 == 0:
            print(f'  Progress: {idx}/{n_total} pixels')

        series = stack[:, r, c]
        finite = np.isfinite(series)

        if finite.sum() < MIN_OBS:
            continue

        result = mk.original_test(series[finite], alpha=ALPHA)
        tau_map[r, c] = result.Tau

        if result.p <= ALPHA:
            trend_map[r, c] = 1.0 if result.trend == 'increasing' else -1.0

    return trend_map, tau_map

# =============================================================================
# PLOT — per-crop map
# =============================================================================

def plot_crop_map(trend_map, tau_map, mask, label, tag):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{label} — MK Trend of ΔYield (Observed − No-Thaw)',
                 fontsize=12, fontweight='bold')

    # Left: trend direction + significance
    ax = axes[0]
    display = np.where(mask, trend_map, np.nan)
    cmap  = mcolors.ListedColormap(['#d73027', '#f7f7f7', '#4575b4'])
    norm  = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    im    = ax.imshow(display, cmap=cmap, norm=norm, interpolation='nearest')
    cb    = plt.colorbar(im, ax=ax, shrink=0.75, ticks=[-1, 0, 1])
    cb.set_ticklabels(['Sig. negative', 'No trend', 'Sig. positive'])
    n_pos = int((trend_map == 1).sum())
    n_neg = int((trend_map == -1).sum())
    n_tot = int(mask.sum())
    ax.set_title(f'Trend direction (p<{ALPHA})\n'
                 f'Sig+: {n_pos} px ({100*n_pos/n_tot:.1f}%)  '
                 f'Sig−: {n_neg} px ({100*n_neg/n_tot:.1f}%)',
                 fontsize=10)
    ax.axis('off')

    # Right: Kendall tau magnitude
    ax = axes[1]
    tau_display = np.where(mask, tau_map, np.nan)
    vlim = np.nanpercentile(np.abs(tau_display[np.isfinite(tau_display)]), 97)
    im2  = ax.imshow(tau_display, cmap='RdBu', vmin=-vlim, vmax=vlim,
                     interpolation='nearest')
    plt.colorbar(im2, ax=ax, shrink=0.75, label='Kendall τ')
    ax.set_title('Kendall τ (effect size)', fontsize=10)
    ax.axis('off')

    plt.tight_layout()
    fname = f'{OUT_DIR}/{tag.replace("combined_","")}_mk_diff.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {fname}')

# =============================================================================
# PLOT — summary heatmap across all crops
# =============================================================================

def plot_summary_heatmap(summary_records):
    df         = pd.DataFrame(summary_records)
    crop_order = [c['label'] for c in CROPS]
    df['crop'] = pd.Categorical(df['crop'], categories=crop_order, ordered=True)
    df         = df.sort_values('crop')

    fig, ax = plt.subplots(figsize=(7, 8))
    fig.suptitle('MK Trend of ΔYield Timeseries — All Crops\n'
                 '(% pixels with significant trend in thaw impact)',
                 fontsize=12, fontweight='bold')

    cols_plot  = ['pct_sig_pos', 'pct_sig_neg']
    col_labels = ['Sig. positive\ntrend (%)', 'Sig. negative\ntrend (%)']
    mat        = df[cols_plot].values

    # Use a neutral sequential colormap — magnitude not direction here
    im = ax.imshow(mat, cmap='YlOrRd', vmin=0, vmax=max(mat.max(), 5),
                   aspect='auto')
    plt.colorbar(im, ax=ax, label='% of pixels', shrink=0.7)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['crop'].tolist(), fontsize=11)

    for i in range(len(df)):
        for j, col in enumerate(cols_plot):
            v = df[col].iloc[i]
            ax.text(j, i, f'{v:.1f}%', ha='center', va='center',
                    fontsize=10,
                    color='white' if v > mat.max() * 0.7 else 'black')

    ax.set_title('Black outline = any sig. pixels present', fontsize=9, pad=8)

    # Outline cells that have any significant pixels
    for i in range(len(df)):
        for j, col in enumerate(cols_plot):
            if df[col].iloc[i] > 0:
                ax.add_patch(Rectangle((j-.5, i-.5), 1, 1,
                                        fill=False, edgecolor='black', lw=1.5))

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/SUMMARY_mk_diff_heatmap.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: SUMMARY_mk_diff_heatmap.png')

    # Also save the numbers
    df.to_csv(f'{OUT_DIR}/mk_diff_summary.csv', index=False)
    print('  Saved: mk_diff_summary.csv')
    print('\nSummary table:')
    print(df.to_string(index=False))

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print('=' * 60)
    print('Mann-Kendall on ΔYield Difference Timeseries')
    print('=' * 60)

    mask = load_mask().astype(bool)
    n_px = int(mask.sum())
    print(f'Mask loaded: {n_px} active pixels')

    summary_records = []

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        print(f'\n--- {label} ---')

        print('  Loading difference stack...')
        stack = load_diff_stack(tag, mask)

        # Check how many years have data
        n_years_with_data = np.sum(np.any(np.isfinite(stack), axis=(1, 2)))
        print(f'  Years with data: {n_years_with_data}/{len(YEARS_COMPARISON)}')

        if n_years_with_data < MIN_OBS:
            print(f'  Skipping — insufficient years')
            continue

        print('  Running pixel-wise Mann-Kendall...')
        trend_map, tau_map = run_mk_diff(stack, mask)

        plot_crop_map(trend_map, tau_map, mask, label, tag)

        n_sig_pos = int((trend_map ==  1).sum())
        n_sig_neg = int((trend_map == -1).sum())
        summary_records.append({
            'crop'       : label,
            'n_pixels'   : n_px,
            'n_sig_pos'  : n_sig_pos,
            'n_sig_neg'  : n_sig_neg,
            'pct_sig_pos': round(100 * n_sig_pos / n_px, 2),
            'pct_sig_neg': round(100 * n_sig_neg / n_px, 2),
        })

    print('\n[Summary heatmap]')
    plot_summary_heatmap(summary_records)

    print(f'\nAll outputs written to: {OUT_DIR}/')