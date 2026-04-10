"""
Cumulative ΔSuitability with Permutation-Based Confidence Intervals
====================================================================
For each crop and overall aggregate, plots the cumulative sum of
annual ΔSuitability (observed − counterfactual) from 1999–2018.

The line starts at zero in 1999 and shows how the accumulated
thaw contribution grows (or shrinks) over time.

Statistical significance is assessed via permutation test:
  - Randomly shuffle the signs of annual differences 1000 times
  - Compute cumulative sum for each permutation
  - 95% CI from 2.5th and 97.5th percentiles of permuted distributions
  - If actual cumulative gap lies outside CI → significant

Also tests three metrics:
  - ΔSuitability (fixed-boundary suitability classes)
  - ΔTotal yield (tonnes)
  - ΔSuitable area (km²)

Outputs written to:
  ./results_analysis/outputs/cumulative_delta/
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
OUT_ROOT  = './results_analysis/outputs/cumulative_delta'

YEARS_CF     = list(range(1999, 2019))
N_PERM       = 1000
PIXEL_AREA_KM2 = 78.0
PIXEL_AREA_HA  = PIXEL_AREA_KM2 * 100

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
    return load_raster(MASK_PATH).astype(bool)

def regional_mean_suit(arr, mask):
    """Mean suitability over ALL mask pixels including zeros."""
    arr = arr.copy()
    arr[arr < 0] = np.nan
    valid = mask & np.isfinite(arr)
    return float(np.nanmean(arr[valid])) if valid.any() else np.nan

def total_yield_tonnes(arr, mask):
    valid = mask & np.isfinite(arr) & (arr > 0)
    return float(np.sum(arr[valid]) * PIXEL_AREA_HA / 1000) if valid.any() else 0.0

def suitable_area_km2(arr, mask):
    valid = mask & np.isfinite(arr) & (arr > 0)
    return float(np.sum(valid)) * PIXEL_AREA_KM2

def obs_suit_path(tag, year):
    return f'./data_output/final_classification_fixed/{tag}/{year}_suitability_class.tif'

def cf_suit_path(tag, year):
    return f'./data_output/final_classification_nothaw_fixed/{tag}/{year}_suitability_class.tif'

def obs_yield_path(tag, year):
    return f'./data_output/final_classification/{tag}/{year}_raw_yield.tif'

def cf_yield_path(tag, year):
    return f'./data_output/final_classification_nothaw/{tag}/{year}_raw_yield.tif'

def permutation_ci(diff_series, n_perm=N_PERM, ci=95):
    """
    Permutation test for cumulative sum of diff_series.
    Randomly flips signs of annual differences to simulate null hypothesis
    of no systematic thaw effect. Returns (lower, upper) CI bands and
    two-sided p-value for the final cumulative gap.
    """
    valid = np.isfinite(diff_series)
    vals  = np.where(valid, diff_series, 0.0)
    n     = len(vals)

    perms = np.zeros((n_perm, n))
    for i in range(n_perm):
        # Randomly flip signs — null hypothesis: no systematic direction
        signs       = np.random.choice([-1, 1], size=n)
        perms[i]    = np.cumsum(vals * signs)

    alpha  = (100 - ci) / 2
    lower  = np.percentile(perms, alpha,      axis=0)
    upper  = np.percentile(perms, 100 - alpha, axis=0)

    # Two-sided p-value: proportion of permutations with final gap
    # as extreme as observed
    actual_final = np.cumsum(vals)[-1]
    p_val = float(np.mean(np.abs(perms[:, -1]) >= np.abs(actual_final)))

    return lower, upper, p_val

def plot_cumulative_delta(ax, years_arr, diff_series, title, ylabel,
                          color='#2166AC'):
    """
    Plot cumulative ΔMetric as a single line from zero with
    permutation CI bands.
    """
    vals     = np.where(np.isfinite(diff_series), diff_series, 0.0)
    cum_diff = np.cumsum(vals)
    lo, hi, p_val = permutation_ci(diff_series)

    # Main cumulative line
    ax.plot(years_arr, cum_diff, color=color, linewidth=2.5,
            marker='o', markersize=4, label='Cumulative Δ (Obs − CF)')

    # Permutation CI band
    ax.fill_between(years_arr, lo, hi, alpha=0.20, color='grey',
                    label=f'95% permutation CI')

    # Zero reference line
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    # Shade where cumulative gap is outside CI
    ax.fill_between(years_arr, cum_diff, 0,
                    where=(cum_diff > hi),
                    alpha=0.25, color=color, label='Sig. positive')
    ax.fill_between(years_arr, cum_diff, 0,
                    where=(cum_diff < lo),
                    alpha=0.25, color='#D6604D', label='Sig. negative')

    # Final value annotation
    final = cum_diff[-1]
    sig_str = f'★ p={p_val:.3f}' if p_val < 0.05 else f'n.s. p={p_val:.3f}'
    ann_color = color if final >= 0 else '#D6604D'
    ax.annotate(
        f'{final:+,.3f}\n{sig_str}',
        xy=(years_arr[-1], final),
        xytext=(-65, 15 if final >= 0 else -30),
        textcoords='offset points',
        fontsize=8.5, color=ann_color, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=ann_color, lw=1.2)
    )

    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xticks(years_arr[::2])
    ax.set_xticklabels(years_arr[::2], rotation=45, ha='right', fontsize=8)


# ── Build annual difference series ────────────────────────────────────────────

def build_diff_series(tag, mask):
    """Build annual ΔSuitability, ΔTotal yield, ΔArea series."""
    d_suit  = []
    d_total = []
    d_area  = []

    for year in YEARS_CF:
        obs_s = load_raster(obs_suit_path(tag, year))
        cf_s  = load_raster(cf_suit_path(tag, year))
        obs_y = load_raster(obs_yield_path(tag, year))
        cf_y  = load_raster(cf_yield_path(tag, year))

        # ΔSuitability
        if obs_s is not None and cf_s is not None:
            d_suit.append(regional_mean_suit(obs_s, mask) -
                          regional_mean_suit(cf_s, mask))
        else:
            d_suit.append(np.nan)

        # ΔTotal yield
        if obs_y is not None and cf_y is not None:
            d_total.append(total_yield_tonnes(obs_y, mask) -
                           total_yield_tonnes(cf_y, mask))
        else:
            d_total.append(np.nan)

        # ΔSuitable area
        if obs_y is not None and cf_y is not None:
            d_area.append(suitable_area_km2(obs_y, mask) -
                          suitable_area_km2(cf_y, mask))
        else:
            d_area.append(np.nan)

    return (np.array(d_suit),
            np.array(d_total),
            np.array(d_area))


# ── Main ──────────────────────────────────────────────────────────────────────

def run(mask):
    years_arr = np.array(YEARS_CF)
    summary   = []

    # Store diff series for overall aggregate
    all_suit  = []
    all_total = []
    all_area  = []

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        print(f'  {label} …')

        d_suit, d_total, d_area = build_diff_series(tag, mask)
        all_suit.append(d_suit)
        all_total.append(d_total)
        all_area.append(d_area)

        # Per-crop figure
        fig, axes = plt.subplots(1, 3, figsize=(21, 5))

        plot_cumulative_delta(axes[0], years_arr, d_suit,
                              'ΔSuitability',
                              'Cumulative ΔSuitability (class units)')
        plot_cumulative_delta(axes[1], years_arr, d_total,
                              'ΔTotal Yield',
                              'Cumulative ΔTotal Yield (tonnes)')
        plot_cumulative_delta(axes[2], years_arr, d_area,
                              'ΔSuitable Area',
                              'Cumulative ΔSuitable Area (km²)')

        fig.suptitle(
            f'{label} — Cumulative Thaw Contribution (1999–2018)\n'
            f'Single line = accumulated difference (Obs − No-Thaw CF), '
            f'Grey band = 95% permutation CI',
            fontsize=12, fontweight='bold'
        )
        plt.tight_layout()
        fig.savefig(f'{OUT_ROOT}/{tag}_cumulative_delta.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

        # Record summary stats
        for diff, metric in [(d_suit, 'suitability'),
                             (d_total, 'total_yield'),
                             (d_area,  'area')]:
            vals     = np.where(np.isfinite(diff), diff, 0.0)
            cum_final = float(np.cumsum(vals)[-1])
            _, _, p  = permutation_ci(diff)
            summary.append({
                'crop'       : label,
                'metric'     : metric,
                'cum_final'  : round(cum_final, 4),
                'p_value'    : round(p, 4),
                'significant': p < 0.05,
            })

    # ── Overall aggregate ──────────────────────────────────────────────────────
    print('  Overall aggregate …')

    # Normalize suitability and yield across crops before averaging
    all_suit  = np.array(all_suit)   # (n_crops, 20)
    all_total = np.array(all_total)
    all_area  = np.array(all_area)

    # Suitability: direct mean (same scale across crops)
    agg_suit  = np.nanmean(all_suit,  axis=0)

    # Normalize yield differences by each crop's mean absolute difference
    # so no single high-yield crop dominates
    crop_scale = np.nanmean(np.abs(all_total), axis=1, keepdims=True)
    crop_scale[crop_scale == 0] = np.nan
    agg_total_norm = np.nanmean(all_total / crop_scale, axis=0)

    # Area: sum across crops (additive)
    agg_area  = np.nansum(all_area, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(21, 5))

    plot_cumulative_delta(axes[0], years_arr, agg_suit,
                          'Overall ΔSuitability (mean across crops)',
                          'Cumulative ΔSuitability (class units)')
    plot_cumulative_delta(axes[1], years_arr, agg_total_norm,
                          'Overall ΔTotal Yield (normalized)',
                          'Cumulative ΔYield (normalized)')
    plot_cumulative_delta(axes[2], years_arr, agg_area,
                          'Overall ΔSuitable Area (sum across crops)',
                          'Cumulative ΔArea (km²)')

    fig.suptitle(
        'Overall Aggregate — Cumulative Thaw Contribution (1999–2018)\n'
        'Grey band = 95% permutation CI  |  ★ = significant at p < 0.05',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    fig.savefig(f'{OUT_ROOT}/OVERALL_cumulative_delta.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    # Overall summary stats
    for diff, metric in [(agg_suit,       'suitability'),
                         (agg_total_norm, 'total_yield_norm'),
                         (agg_area,       'area')]:
        vals      = np.where(np.isfinite(diff), diff, 0.0)
        cum_final = float(np.cumsum(vals)[-1])
        _, _, p   = permutation_ci(diff)
        summary.append({
            'crop'       : 'OVERALL',
            'metric'     : metric,
            'cum_final'  : round(cum_final, 4),
            'p_value'    : round(p, 4),
            'significant': p < 0.05,
        })

    # ── Save summary ──────────────────────────────────────────────────────────
    df = pd.DataFrame(summary)
    df.to_csv(f'{OUT_ROOT}/cumulative_delta_summary.csv', index=False)

    print(f'\n✓ All outputs saved to: {OUT_ROOT}/')
    print('\nSummary:')
    print(df.to_string(index=False))


if __name__ == '__main__':
    np.random.seed(42)   # reproducibility
    mask = load_mask()
    run(mask)