"""
40-Year Trend Analysis: Observed vs. No-Thaw Counterfactual (1979–2018)
========================================================================
Metrics tracked per crop per year:
  - Mean yield (kg/ha) — yield intensity over suitable pixels
  - Total yield (tonnes) — sum of yield × pixel area over suitable pixels
  - Suitable area (km²) — count of pixels with yield > 0

Outputs written to: ./results_analysis/outputs/3_trend_40yr_output/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from osgeo import gdal
from pymannkendall import original_test as mk_test

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR  = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH = r'./data_input/qilian mask.tif'
OUT_ROOT  = './results_analysis/outputs/3_trend_40yr_output'

YEARS_ALL       = list(range(1979, 2019))
DIVERGENCE_YEAR = 1999

# Pixel area at ~37.8°N, 0.1° grid
PIXEL_AREA_KM2 = 78.0
PIXEL_AREA_HA  = PIXEL_AREA_KM2 * 100   # 1 km² = 100 ha

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
    """Mean yield over pixels with positive yield only (intensity)."""
    valid = mask & np.isfinite(arr) & (arr > 0)
    return float(np.nanmean(arr[valid])) if valid.any() else np.nan

def total_yield_tonnes(arr, mask):
    """
    Total yield in tonnes across all suitable pixels.
    yield (kg/ha) × pixel_area (ha) / 1000 = tonnes per pixel
    """
    valid = mask & np.isfinite(arr) & (arr > 0)
    return float(np.sum(arr[valid]) * PIXEL_AREA_HA / 1000) if valid.any() else 0.0

def suitable_area_km2(arr, mask):
    """Area of pixels with yield > 0 in km²."""
    valid = mask & np.isfinite(arr) & (arr > 0)
    return float(np.sum(valid)) * PIXEL_AREA_KM2

def obs_path(tag, year):
    return f'./data_output/final_classification/{tag}/{year}_raw_yield.tif'

def cf_path(tag, year):
    return f'./data_output/final_classification_nothaw/{tag}/{year}_raw_yield.tif'

def run_mk(series, years):
    s = np.array(series, dtype=float)
    valid = np.isfinite(s)
    if valid.sum() < 4:
        return None
    mk = mk_test(s[valid])
    y_hat = mk.intercept + mk.slope * np.arange(len(s[valid]))
    line  = np.full(len(series), np.nan)
    line[valid] = y_hat
    return {
        'tau'        : round(mk.Tau, 3),
        'p_value'    : round(mk.p, 4),
        'slope_sen'  : round(mk.slope, 4),
        'trend'      : mk.trend,
        'significant': mk.p < 0.05,
        'n_years'    : int(valid.sum()),
        'sen_line'   : line,
    }

def append_mk_result(results, label, period, metric, mk_obs, mk_cf):
    if mk_obs and mk_cf:
        results.append({
            'crop'            : label,
            'period'          : period,
            'metric'          : metric,
            'obs_tau'         : mk_obs['tau'],
            'obs_p'           : mk_obs['p_value'],
            'obs_slope'       : mk_obs['slope_sen'],
            'obs_trend'       : mk_obs['trend'],
            'obs_significant' : mk_obs['significant'],
            'cf_tau'          : mk_cf['tau'],
            'cf_p'            : mk_cf['p_value'],
            'cf_slope'        : mk_cf['slope_sen'],
            'cf_trend'        : mk_cf['trend'],
            'cf_significant'  : mk_cf['significant'],
            'slope_difference': round(mk_obs['slope_sen'] - mk_cf['slope_sen'], 4),
        })

def plot_series(ax, years_arr, obs_s, cf_s, mk_obs, mk_cf, ylabel,
                title, slope_unit):
    ax.plot(years_arr, obs_s, color='#2166AC', linewidth=2,
            marker='o', markersize=4, label='Observed')
    ax.plot(years_arr, cf_s,  color='#D6604D', linewidth=2,
            marker='s', markersize=4, linestyle='--', label='No-Thaw CF')
    if mk_obs:
        ax.plot(years_arr, mk_obs['sen_line'], color='#2166AC',
                linewidth=1.5, linestyle=':', alpha=0.8,
                label=f"Obs slope: {mk_obs['slope_sen']:.3f} {slope_unit} "
                      f"(p={mk_obs['p_value']:.3f})")
    if mk_cf:
        ax.plot(years_arr, mk_cf['sen_line'], color='#D6604D',
                linewidth=1.5, linestyle=':', alpha=0.8,
                label=f"CF slope: {mk_cf['slope_sen']:.3f} {slope_unit} "
                      f"(p={mk_cf['p_value']:.3f})")
    ax.axvline(DIVERGENCE_YEAR, color='grey', linestyle='--',
               linewidth=1.2, label='Divergence (1999)')
    ax.axvspan(DIVERGENCE_YEAR, YEARS_ALL[-1], alpha=0.04, color='grey')
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xticks(years_arr[::4])
    ax.set_xticklabels(years_arr[::4], rotation=45, ha='right', fontsize=8)


# ── Per-crop analysis ─────────────────────────────────────────────────────────

def run(mask):
    all_results = []
    years_arr   = np.array(YEARS_ALL)
    post_mask   = years_arr >= DIVERGENCE_YEAR

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        print(f'\n── {label} ──')

        obs_yield_s, cf_yield_s = [], []
        obs_total_s, cf_total_s = [], []
        obs_area_s,  cf_area_s  = [], []

        for year in YEARS_ALL:
            obs = load_raster(obs_path(tag, year))
            obs_yield_s.append(spatial_mean_yield(obs, mask)  if obs is not None else np.nan)
            obs_total_s.append(total_yield_tonnes(obs, mask)  if obs is not None else np.nan)
            obs_area_s.append(suitable_area_km2(obs, mask)    if obs is not None else np.nan)

            if year < DIVERGENCE_YEAR:
                cf_yield_s.append(obs_yield_s[-1])
                cf_total_s.append(obs_total_s[-1])
                cf_area_s.append(obs_area_s[-1])
            else:
                cf = load_raster(cf_path(tag, year))
                cf_yield_s.append(spatial_mean_yield(cf, mask) if cf is not None else np.nan)
                cf_total_s.append(total_yield_tonnes(cf, mask) if cf is not None else np.nan)
                cf_area_s.append(suitable_area_km2(cf, mask)   if cf is not None else np.nan)

        obs_yield_s = np.array(obs_yield_s)
        cf_yield_s  = np.array(cf_yield_s)
        obs_total_s = np.array(obs_total_s)
        cf_total_s  = np.array(cf_total_s)
        obs_area_s  = np.array(obs_area_s)
        cf_area_s   = np.array(cf_area_s)

        # ── MK tests for all three metrics ────────────────────────────────────
        for period, idx in [('1979-2018', slice(None)), ('1999-2018', post_mask)]:
            for metric, obs_s, cf_s in [
                ('mean_yield_kg_ha',  obs_yield_s, cf_yield_s),
                ('total_yield_tonnes', obs_total_s, cf_total_s),
                ('suitable_area_km2', obs_area_s,  cf_area_s),
            ]:
                mk_obs = run_mk(obs_s[idx], years_arr[idx])
                mk_cf  = run_mk(cf_s[idx],  years_arr[idx])
                append_mk_result(all_results, label, period, metric, mk_obs, mk_cf)

        # ── Per-crop plot: 3 panels ────────────────────────────────────────────
        mk_y_obs = run_mk(obs_yield_s, years_arr)
        mk_y_cf  = run_mk(cf_yield_s,  years_arr)
        mk_t_obs = run_mk(obs_total_s, years_arr)
        mk_t_cf  = run_mk(cf_total_s,  years_arr)
        mk_a_obs = run_mk(obs_area_s,  years_arr)
        mk_a_cf  = run_mk(cf_area_s,   years_arr)

        fig, axes = plt.subplots(1, 3, figsize=(21, 5))

        plot_series(axes[0], years_arr, obs_yield_s, cf_yield_s,
                    mk_y_obs, mk_y_cf,
                    'Mean Yield (kg/ha)', 'Yield Intensity', 'kg/ha/yr')
        plot_series(axes[1], years_arr, obs_total_s, cf_total_s,
                    mk_t_obs, mk_t_cf,
                    'Total Yield (tonnes)', 'Total Yield', 'tonnes/yr')
        plot_series(axes[2], years_arr, obs_area_s, cf_area_s,
                    mk_a_obs, mk_a_cf,
                    'Suitable Area (km²)', 'Suitable Area', 'km²/yr')

        fig.suptitle(f'{label} — Yield Intensity, Total Yield & Suitable Area (1979–2018)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        fig.savefig(f'{OUT_ROOT}/{tag}_three_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  ✓ {label} saved')

    # ── Save CSV ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    df.to_csv(f'{OUT_ROOT}/trend_all_metrics.csv', index=False)

    # ── Summary figures: slope comparison per metric ───────────────────────────
    for metric, ylabel, unit in [
        ('mean_yield_kg_ha',   "Sen's Slope (kg/ha/yr)",  'kg/ha/yr'),
        ('total_yield_tonnes', "Sen's Slope (tonnes/yr)", 'tonnes/yr'),
        ('suitable_area_km2',  "Sen's Slope (km²/yr)",    'km²/yr'),
    ]:
        df_m = df[df['metric'] == metric]
        for period in ['1979-2018', '1999-2018']:
            df_p = df_m[df_m['period'] == period].copy().sort_values('obs_slope')
            if df_p.empty:
                continue
            x     = np.arange(len(df_p))
            width = 0.35
            fig, ax = plt.subplots(figsize=(13, 6))
            bars_obs = ax.bar(x - width/2, df_p['obs_slope'], width,
                              label='Observed', color='#2166AC', alpha=0.85)
            bars_cf  = ax.bar(x + width/2, df_p['cf_slope'],  width,
                              label='No-Thaw CF', color='#D6604D', alpha=0.85)
            for bars, sig_col in [(bars_obs, 'obs_significant'),
                                  (bars_cf,  'cf_significant')]:
                for bar, (_, row) in zip(bars, df_p.iterrows()):
                    if row[sig_col]:
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + abs(bar.get_height()) * 0.02,
                                '★', ha='center', va='bottom', fontsize=11)
            ax.axhline(0, color='black', linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(df_p['crop'], rotation=30, ha='right', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(
                f'{metric.replace("_", " ").title()} Trend: Observed vs. No-Thaw ({period})\n'
                f'★ = significant at p < 0.05',
                fontsize=12, fontweight='bold'
            )
            ax.legend(fontsize=11)
            plt.tight_layout()
            fig.savefig(
                f'{OUT_ROOT}/slope_{metric}_{period.replace("-","_")}.png',
                dpi=150, bbox_inches='tight'
            )
            plt.close()

    print(f'\n✓ All outputs saved to: {OUT_ROOT}/')
    print(df.to_string(index=False))


# ── Overall aggregate ─────────────────────────────────────────────────────────

def run_overall(mask):
    print('\n══ Overall Aggregate Analysis ══')
    years_arr = np.array(YEARS_ALL)
    post_mask = years_arr >= DIVERGENCE_YEAR

    obs_yield_all, cf_yield_all = [], []
    obs_total_all, cf_total_all = [], []
    obs_area_all,  cf_area_all  = [], []

    for crop in CROPS:
        tag = crop['tag']
        obs_y, cf_y = [], []
        obs_t, cf_t = [], []
        obs_a, cf_a = [], []

        for year in YEARS_ALL:
            obs = load_raster(obs_path(tag, year))
            obs_y.append(spatial_mean_yield(obs, mask) if obs is not None else np.nan)
            obs_t.append(total_yield_tonnes(obs, mask) if obs is not None else np.nan)
            obs_a.append(suitable_area_km2(obs, mask)  if obs is not None else np.nan)

            if year < DIVERGENCE_YEAR:
                cf_y.append(obs_y[-1])
                cf_t.append(obs_t[-1])
                cf_a.append(obs_a[-1])
            else:
                cf = load_raster(cf_path(tag, year))
                cf_y.append(spatial_mean_yield(cf, mask) if cf is not None else np.nan)
                cf_t.append(total_yield_tonnes(cf, mask) if cf is not None else np.nan)
                cf_a.append(suitable_area_km2(cf, mask)  if cf is not None else np.nan)

        obs_yield_all.append(np.array(obs_y))
        cf_yield_all.append(np.array(cf_y))
        obs_total_all.append(np.array(obs_t))
        cf_total_all.append(np.array(cf_t))
        obs_area_all.append(np.array(obs_a))
        cf_area_all.append(np.array(cf_a))

    obs_yield_all = np.array(obs_yield_all)
    cf_yield_all  = np.array(cf_yield_all)

    # Normalize yield before averaging (different scales per crop)
    crop_means = np.nanmean(obs_yield_all, axis=1, keepdims=True)
    crop_means[crop_means == 0] = np.nan
    obs_yield_norm = np.nanmean(obs_yield_all / crop_means, axis=0)
    cf_yield_norm  = np.nanmean(cf_yield_all  / crop_means, axis=0)

    # Total yield and area are additive across crops
    obs_total_agg = np.nansum(obs_total_all, axis=0)
    cf_total_agg  = np.nansum(cf_total_all,  axis=0)
    obs_area_agg  = np.nansum(obs_area_all,  axis=0)
    cf_area_agg   = np.nansum(cf_area_all,   axis=0)

    results = []
    for period, idx in [('1979-2018', slice(None)), ('1999-2018', post_mask)]:
        for metric, obs_s, cf_s in [
            ('normalized_yield',   obs_yield_norm, cf_yield_norm),
            ('total_yield_tonnes', obs_total_agg,  cf_total_agg),
            ('total_area_km2',     obs_area_agg,   cf_area_agg),
        ]:
            mk_obs = run_mk(obs_s[idx], years_arr[idx])
            mk_cf  = run_mk(cf_s[idx],  years_arr[idx])
            if mk_obs and mk_cf:
                results.append({
                    'metric'          : metric,
                    'period'          : period,
                    'obs_tau'         : mk_obs['tau'],
                    'obs_p'           : mk_obs['p_value'],
                    'obs_slope'       : mk_obs['slope_sen'],
                    'obs_significant' : mk_obs['significant'],
                    'cf_tau'          : mk_cf['tau'],
                    'cf_p'            : mk_cf['p_value'],
                    'cf_slope'        : mk_cf['slope_sen'],
                    'cf_significant'  : mk_cf['significant'],
                    'slope_difference': round(mk_obs['slope_sen'] - mk_cf['slope_sen'], 4),
                })
                print(f'  [{metric}] {period} | '
                      f'obs: τ={mk_obs["tau"]:.3f} p={mk_obs["p_value"]:.4f} '
                      f'slope={mk_obs["slope_sen"]:.3f} | '
                      f'cf: τ={mk_cf["tau"]:.3f} p={mk_cf["p_value"]:.4f} '
                      f'slope={mk_cf["slope_sen"]:.3f} | '
                      f'Δslope={mk_obs["slope_sen"]-mk_cf["slope_sen"]:.3f}')

    # ── Overall plot: 3 panels ─────────────────────────────────────────────────
    mk_y_obs = run_mk(obs_yield_norm, years_arr)
    mk_y_cf  = run_mk(cf_yield_norm,  years_arr)
    mk_t_obs = run_mk(obs_total_agg,  years_arr)
    mk_t_cf  = run_mk(cf_total_agg,   years_arr)
    mk_a_obs = run_mk(obs_area_agg,   years_arr)
    mk_a_cf  = run_mk(cf_area_agg,    years_arr)

    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    plot_series(axes[0], years_arr, obs_yield_norm, cf_yield_norm,
                mk_y_obs, mk_y_cf,
                'Normalized Yield', 'Overall Yield Intensity (Normalized)', '')
    plot_series(axes[1], years_arr, obs_total_agg, cf_total_agg,
                mk_t_obs, mk_t_cf,
                'Total Yield (tonnes)', 'Overall Total Yield', 'tonnes/yr')
    plot_series(axes[2], years_arr, obs_area_agg, cf_area_agg,
                mk_a_obs, mk_a_cf,
                'Total Suitable Area (km²)', 'Overall Suitable Area', 'km²/yr')

    fig.suptitle('Overall Aggregate — Yield Intensity, Total Yield & Suitable Area (1979–2018)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{OUT_ROOT}/overall_three_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

    pd.DataFrame(results).to_csv(f'{OUT_ROOT}/overall_trend_results.csv', index=False)
    print(f'\n✓ Overall results saved to: {OUT_ROOT}/overall_trend_results.csv')


if __name__ == '__main__':
    mask = load_mask()
    run(mask)
    run_overall(mask)