"""
Cumulative 40-Year Trend Analysis (1979–2018)
=============================================
Plots cumulative sums of:
  - Total yield (tonnes)
  - Mean yield (kg/ha)
  - Suitable area (km²)

For both observed and no-thaw counterfactual scenarios.
The gap between the two lines represents the accumulated
difference attributable to permafrost thaw over time.

Pre-1999: counterfactual = observed (identical by construction)
Post-1999: scenarios diverge as permafrost data differs

Outputs per crop:
  - {tag}_cumulative.png — 3-panel cumulative plot

Outputs overall:
  - overall_cumulative.png — 3-panel cumulative aggregate

Outputs written to: ./results_analysis/outputs/3_trend_40yr_output/cumulative/
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
OUT_ROOT  = './results_analysis/outputs/3_trend_40yr_output/cumulative'

YEARS_ALL       = list(range(1979, 2019))
DIVERGENCE_YEAR = 1999

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

def spatial_mean_yield(arr, mask):
    valid = mask & np.isfinite(arr) & (arr > 0)
    return float(np.nanmean(arr[valid])) if valid.any() else np.nan

def total_yield_tonnes(arr, mask):
    valid = mask & np.isfinite(arr) & (arr > 0)
    return float(np.sum(arr[valid]) * PIXEL_AREA_HA / 1000) if valid.any() else 0.0

def suitable_area_km2(arr, mask):
    valid = mask & np.isfinite(arr) & (arr > 0)
    return float(np.sum(valid)) * PIXEL_AREA_KM2

def obs_path(tag, year):
    return f'./data_output/final_classification/{tag}/{year}_raw_yield.tif'

def cf_path(tag, year):
    return f'./data_output/final_classification_nothaw/{tag}/{year}_raw_yield.tif'

def build_series(tag, mask):
    """Build annual time series for all three metrics, both scenarios."""
    obs_yield, cf_yield = [], []
    obs_total, cf_total = [], []
    obs_area,  cf_area  = [], []

    for year in YEARS_ALL:
        obs = load_raster(obs_path(tag, year))
        obs_yield.append(spatial_mean_yield(obs, mask) if obs is not None else np.nan)
        obs_total.append(total_yield_tonnes(obs, mask) if obs is not None else 0.0)
        obs_area.append(suitable_area_km2(obs, mask)   if obs is not None else 0.0)

        if year < DIVERGENCE_YEAR:
            cf_yield.append(obs_yield[-1])
            cf_total.append(obs_total[-1])
            cf_area.append(obs_area[-1])
        else:
            cf = load_raster(cf_path(tag, year))
            cf_yield.append(spatial_mean_yield(cf, mask) if cf is not None else np.nan)
            cf_total.append(total_yield_tonnes(cf, mask) if cf is not None else 0.0)
            cf_area.append(suitable_area_km2(cf, mask)   if cf is not None else 0.0)

    return {
        'obs_yield': np.array(obs_yield),
        'cf_yield' : np.array(cf_yield),
        'obs_total': np.array(obs_total),
        'cf_total' : np.array(cf_total),
        'obs_area' : np.array(obs_area),
        'cf_area'  : np.array(cf_area),
    }

N_BOOTSTRAP = 1000   # number of bootstrap iterations

def cumsum_valid(series):
    """Cumulative sum treating NaN as zero (so gaps don't break the series)."""
    s = np.where(np.isfinite(series), series, 0.0)
    return np.cumsum(s)

def bootstrap_cumsum_ci(diff_series, n_boot=N_BOOTSTRAP, ci=95):
    """
    Bootstrap confidence bands for the cumulative sum of diff_series.
    At each year t, resamples the annual differences WITH replacement
    and accumulates them, giving a distribution of possible cumulative
    trajectories. Returns (lower, upper) bands at each time step.

    Note: resampling breaks temporal order intentionally — we are
    asking "given this distribution of annual differences, what range
    of cumulative totals is plausible?" not testing autocorrelation.
    """
    valid = np.isfinite(diff_series)
    n     = valid.sum()
    if n < 4:
        nans = np.full(len(diff_series), np.nan)
        return nans, nans

    vals  = diff_series[valid]
    boots = np.zeros((n_boot, len(diff_series)))

    for i in range(n_boot):
        # Resample annual differences
        resampled = np.random.choice(vals, size=len(diff_series), replace=True)
        boots[i]  = np.cumsum(resampled)

    alpha = (100 - ci) / 2
    lower = np.percentile(boots, alpha,     axis=0)
    upper = np.percentile(boots, 100-alpha, axis=0)
    return lower, upper

def plot_cumulative(axes, years_arr, obs_s, cf_s,
                    ylabel, title, divergence_year=DIVERGENCE_YEAR):
    """
    Plot cumulative observed vs CF with:
      - shaded gap between scenarios
      - bootstrap 95% CI band around the cumulative gap
    """
    cum_obs  = cumsum_valid(obs_s)
    cum_cf   = cumsum_valid(cf_s)
    cum_gap  = cum_obs - cum_cf   # cumulative accumulated difference

    # Bootstrap CI on the cumulative gap
    diff_series = np.where(np.isfinite(obs_s) & np.isfinite(cf_s),
                           obs_s - cf_s, np.nan)
    ci_lo, ci_hi = bootstrap_cumsum_ci(diff_series)

    # ── Main lines ────────────────────────────────────────────────────────────
    axes.plot(years_arr, cum_obs, color='#2166AC', linewidth=2,
              marker='o', markersize=3, label='Observed')
    axes.plot(years_arr, cum_cf,  color='#D6604D', linewidth=2,
              marker='s', markersize=3, linestyle='--', label='No-Thaw CF')

    # Shade gap between scenarios
    axes.fill_between(years_arr, cum_obs, cum_cf,
                      where=(cum_obs >= cum_cf),
                      alpha=0.12, color='#2166AC', label='Thaw benefit')
    axes.fill_between(years_arr, cum_obs, cum_cf,
                      where=(cum_obs < cum_cf),
                      alpha=0.12, color='#D6604D', label='Thaw deficit')

    # ── Bootstrap CI on gap ───────────────────────────────────────────────────
    # CI is centered on the counterfactual line + cumulative gap
    ci_center = cum_cf   # reference line
    axes.fill_between(years_arr,
                      ci_center + ci_lo,
                      ci_center + ci_hi,
                      alpha=0.20, color='grey',
                      label='95% CI (bootstrap)')

    # Mark divergence point
    axes.axvline(divergence_year, color='grey', linestyle='--',
                 linewidth=1.2, label=f'Divergence ({divergence_year})')

    # Annotate final gap and whether CI excludes zero
    final_gap = cum_gap[-1]
    ci_lo_final = ci_lo[-1]
    ci_hi_final = ci_hi[-1]
    sig_str = '★ sig' if (ci_lo_final > 0 or ci_hi_final < 0) else 'n.s.'
    gap_color = '#2166AC' if final_gap >= 0 else '#D6604D'
    axes.annotate(
        f'Gap: {final_gap:+,.1f}\n95% CI [{ci_lo_final:+,.1f}, {ci_hi_final:+,.1f}]\n{sig_str}',
        xy=(years_arr[-1], cum_obs[-1]),
        xytext=(-80, 15), textcoords='offset points',
        fontsize=7.5, color=gap_color,
        arrowprops=dict(arrowstyle='->', color=gap_color, lw=1.2)
    )

    axes.set_xlabel('Year', fontsize=10)
    axes.set_ylabel(ylabel, fontsize=10)
    axes.set_title(title, fontsize=11, fontweight='bold')
    axes.legend(fontsize=7.5, loc='upper left')
    axes.set_xticks(years_arr[::4])
    axes.set_xticklabels(years_arr[::4], rotation=45, ha='right', fontsize=8)


# ── Per-crop cumulative plots ─────────────────────────────────────────────────

def run_per_crop(mask):
    years_arr = np.array(YEARS_ALL)
    all_series = {}

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        print(f'  {label} …')

        s = build_series(tag, mask)
        all_series[label] = s

        fig, axes = plt.subplots(1, 3, figsize=(21, 5))

        plot_cumulative(axes[0], years_arr,
                        s['obs_yield'], s['cf_yield'],
                        'Cumulative Mean Yield (kg/ha)', 'Yield Intensity')

        plot_cumulative(axes[1], years_arr,
                        s['obs_total'], s['cf_total'],
                        'Cumulative Total Yield (tonnes)', 'Total Yield')

        plot_cumulative(axes[2], years_arr,
                        s['obs_area'], s['cf_area'],
                        'Cumulative Suitable Area (km²)', 'Suitable Area')

        fig.suptitle(
            f'{label} — Cumulative Observed vs. No-Thaw (1979–2018)\n'
            f'Shading = accumulated difference attributable to thaw',
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout()
        fig.savefig(f'{OUT_ROOT}/{tag}_cumulative.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

    print('  ✓ Per-crop cumulative plots saved.')
    return all_series


# ── Overall aggregate cumulative plot ─────────────────────────────────────────

def run_overall(mask, all_series):
    years_arr = np.array(YEARS_ALL)

    # Collect arrays
    obs_yield_all = np.array([s['obs_yield'] for s in all_series.values()])
    cf_yield_all  = np.array([s['cf_yield']  for s in all_series.values()])
    obs_total_all = np.array([s['obs_total'] for s in all_series.values()])
    cf_total_all  = np.array([s['cf_total']  for s in all_series.values()])
    obs_area_all  = np.array([s['obs_area']  for s in all_series.values()])
    cf_area_all   = np.array([s['cf_area']   for s in all_series.values()])

    # Normalize yield before averaging (different scales per crop)
    crop_means = np.nanmean(obs_yield_all, axis=1, keepdims=True)
    crop_means[crop_means == 0] = np.nan
    obs_yield_norm = np.nanmean(obs_yield_all / crop_means, axis=0)
    cf_yield_norm  = np.nanmean(cf_yield_all  / crop_means, axis=0)

    # Total and area are additive
    obs_total_agg = np.nansum(obs_total_all, axis=0)
    cf_total_agg  = np.nansum(cf_total_all,  axis=0)
    obs_area_agg  = np.nansum(obs_area_all,  axis=0)
    cf_area_agg   = np.nansum(cf_area_all,   axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(21, 5))

    plot_cumulative(axes[0], years_arr,
                    obs_yield_norm, cf_yield_norm,
                    'Cumulative Normalized Yield', 'Overall Yield Intensity')

    plot_cumulative(axes[1], years_arr,
                    obs_total_agg, cf_total_agg,
                    'Cumulative Total Yield (tonnes)', 'Overall Total Yield')

    plot_cumulative(axes[2], years_arr,
                    obs_area_agg, cf_area_agg,
                    'Cumulative Suitable Area (km²)', 'Overall Suitable Area')

    fig.suptitle(
        'Overall Aggregate — Cumulative Observed vs. No-Thaw (1979–2018)\n'
        'Shading = accumulated difference attributable to thaw (all crops combined)',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    fig.savefig(f'{OUT_ROOT}/overall_cumulative.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ Overall cumulative plot saved.')

    # Print final accumulated gap
    print('\nFinal accumulated gap (Observed − Counterfactual):')
    print(f'  Normalized yield: {cumsum_valid(obs_yield_norm)[-1] - cumsum_valid(cf_yield_norm)[-1]:+.4f}')
    print(f'  Total yield:      {cumsum_valid(obs_total_agg)[-1] - cumsum_valid(cf_total_agg)[-1]:+,.0f} tonnes')
    print(f'  Suitable area:    {cumsum_valid(obs_area_agg)[-1] - cumsum_valid(cf_area_agg)[-1]:+,.0f} km²')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    mask = load_mask()
    print('Building per-crop series …')
    all_series = run_per_crop(mask)
    print('\nBuilding overall aggregate …')
    run_overall(mask, all_series)
    print(f'\n✓ All outputs saved to: {OUT_ROOT}/')