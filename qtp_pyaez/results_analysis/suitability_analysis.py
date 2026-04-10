"""
Suitability Score Analysis
==========================
Using fixed-boundary suitability classes (0–5) from reclassification script.

Analyses:
  1. Annual ΔSuitability maps (observed − counterfactual, 1999–2018)
     + small-multiples panel per crop
  2. Wilcoxon signed-rank test on regional mean ΔSuitability (1999–2018)
     — is observed suitability consistently above counterfactual?
  3. Mann-Kendall trend test on regional mean ΔSuitability (1999–2018)
     — is thaw's contribution growing over time?
  4. 40-year trend analysis on mean suitability score across ALL mask pixels
     — includes zeros so frontier expansion is captured

Key difference from raw yield analysis:
  - Mean is computed over ALL mask pixels (including class 0)
  - This captures frontier expansion (0→1) as well as within-suitable changes

Outputs written to: ./results_analysis/outputs/4_suitability_analysis/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from osgeo import gdal
from pymannkendall import original_test as mk_test
from scipy.stats import wilcoxon

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR  = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH = r'./data_input/qilian mask.tif'
OUT_ROOT  = './results_analysis/outputs/4_suitability_analysis'

YEARS_ALL  = list(range(1979, 2019))
YEARS_CF   = list(range(1999, 2019))
DIVERGENCE_YEAR = 1999

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

DIVERGING_CMAP = 'RdBu'

os.chdir(WORK_DIR)
os.makedirs(OUT_ROOT, exist_ok=True)
for sub in ['1_delta_maps', '2_wilcoxon', '3_mk_delta', '4_trend_40yr']:
    os.makedirs(f'{OUT_ROOT}/{sub}', exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_raster(path):
    ds = gdal.Open(path)
    if ds is None:
        return None, None
    band   = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    arr    = ds.ReadAsArray().astype(float)
    arr[arr < -1e10] = np.nan
    if nodata is not None:
        arr[arr == nodata] = np.nan
    geo_info = (ds.GetGeoTransform(), ds.GetProjection(),
                ds.RasterXSize, ds.RasterYSize)
    return arr, geo_info

def save_raster(path, arr, geo_info):
    geo, proj, nx, ny = geo_info
    driver = gdal.GetDriverByName('GTiff')
    ds_out = driver.Create(path, nx, ny, 1, gdal.GDT_Float32)
    ds_out.SetGeoTransform(geo)
    ds_out.SetProjection(proj)
    band = ds_out.GetRasterBand(1)
    band.WriteArray(arr.astype(np.float32))
    band.SetNoDataValue(-9999.0)
    ds_out.FlushCache()

def load_mask():
    arr, _ = load_raster(MASK_PATH)
    return arr.astype(bool)

def obs_suit_path(tag, year):
    return f'./data_output/final_classification_fixed/{tag}/{year}_suitability_class.tif'

def cf_suit_path(tag, year):
    return f'./data_output/final_classification_nothaw_fixed/{tag}/{year}_suitability_class.tif'

def regional_mean_suit(arr, mask):
    """
    Mean suitability across ALL mask pixels including class 0.
    NaN pixels (outside mask or nodata) excluded.
    Class 0 (no yield) included as zero — captures frontier expansion.
    """
    arr_clean = arr.copy()
    arr_clean[arr_clean < 0] = np.nan   # remove nodata sentinels
    valid = mask & np.isfinite(arr_clean)
    return float(np.nanmean(arr_clean[valid])) if valid.any() else np.nan

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
        'slope_sen'  : round(mk.slope, 6),
        'trend'      : mk.trend,
        'significant': mk.p < 0.05,
        'n_years'    : int(valid.sum()),
        'sen_line'   : line,
    }


# ── Analysis 1: Annual ΔSuitability maps ──────────────────────────────────────

def analysis_delta_maps(mask):
    print('\n[Analysis 1] Annual ΔSuitability maps …')
    out_dir = f'{OUT_ROOT}/1_delta_maps'
    tif_dir = f'{out_dir}/tif'
    os.makedirs(tif_dir, exist_ok=True)

    all_ts = []

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        print(f'  {label}')

        annual_deltas = {}
        ts_records    = []

        for year in YEARS_CF:
            obs, geo_info = load_raster(obs_suit_path(tag, year))
            cf,  _        = load_raster(cf_suit_path(tag, year))

            if obs is None or cf is None:
                print(f'    ⚠ missing {year}')
                continue

            obs[~mask] = np.nan
            cf[~mask]  = np.nan
            obs[obs < 0] = np.nan
            cf[cf < 0]   = np.nan

            # ΔSuitability — include all pixels (zeros count)
            delta = np.where(np.isfinite(obs) & np.isfinite(cf),
                             obs - cf, np.nan)
            delta[~mask] = np.nan
            annual_deltas[year] = delta

            # Save GeoTIFF
            out_arr = np.where(np.isfinite(delta), delta, -9999.0)
            crop_tif_dir = f'{tif_dir}/{tag}'
            os.makedirs(crop_tif_dir, exist_ok=True)
            save_raster(f'{crop_tif_dir}/{year}_delta_suit.tif', out_arr, geo_info)

            # Regional mean ΔSuitability (all pixels)
            valid = mask & np.isfinite(delta)
            mean_delta = float(np.nanmean(delta[valid])) if valid.any() else np.nan
            pct_pos    = float(np.nanmean(delta[valid] > 0) * 100) if valid.any() else np.nan
            pct_neg    = float(np.nanmean(delta[valid] < 0) * 100) if valid.any() else np.nan

            ts_records.append({
                'year'        : year,
                'crop'        : label,
                'mean_delta'  : mean_delta,
                'pct_positive': pct_pos,
                'pct_negative': pct_neg,
            })

        if not annual_deltas:
            continue

        all_ts.extend(ts_records)

        # Consistent colour scale across all years
        all_vals = np.concatenate([
            d[mask & np.isfinite(d)] for d in annual_deltas.values()
        ])
        vlim = np.nanpercentile(np.abs(all_vals), 98)
        if vlim == 0 or np.isnan(vlim):
            vlim = 0.5

        # Small-multiples panel
        years_sorted = sorted(annual_deltas.keys())
        ncols = 5
        nrows = -(-len(years_sorted) // ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 3.5, nrows * 3.2))
        axes = axes.flatten()

        for i, year in enumerate(years_sorted):
            im = axes[i].imshow(annual_deltas[year],
                                cmap=DIVERGING_CMAP,
                                vmin=-vlim, vmax=vlim)
            axes[i].set_title(str(year), fontsize=10, fontweight='bold')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], shrink=0.75, label='ΔClass')

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        fig.suptitle(
            f'{label} — Annual ΔSuitability (Observed − No-Thaw)\n'
            f'Colour scale: ±{vlim:.2f} class units (98th pct)',
            fontsize=13, fontweight='bold', y=1.01
        )
        plt.tight_layout()
        fig.savefig(f'{out_dir}/{tag}_delta_suit_panel.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

        # Time series bar chart
        ts_df = pd.DataFrame(ts_records).sort_values('year')
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.bar(ts_df['year'], ts_df['mean_delta'],
                color=np.where(ts_df['mean_delta'] >= 0, '#2166AC', '#D6604D'),
                edgecolor='white', width=0.8)
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_xlabel('Year', fontsize=11)
        ax2.set_ylabel('Regional Mean ΔSuitability (class units)', fontsize=11)
        ax2.set_title(f'{label} — Annual Mean ΔSuitability (Observed − No-Thaw)',
                      fontsize=12, fontweight='bold')
        ax2.set_xticks(YEARS_CF)
        ax2.set_xticklabels(YEARS_CF, rotation=45, ha='right', fontsize=8)
        plt.tight_layout()
        fig2.savefig(f'{out_dir}/{tag}_delta_suit_timeseries.png',
                     dpi=150, bbox_inches='tight')
        plt.close()

    pd.DataFrame(all_ts).to_csv(f'{out_dir}/annual_delta_suit_all_crops.csv', index=False)
    print(f'  ✓ Delta maps complete. CSV saved.')
    return pd.DataFrame(all_ts)


# ── Analysis 2: Wilcoxon signed-rank test ─────────────────────────────────────

def analysis_wilcoxon(delta_df):
    print('\n[Analysis 2] Wilcoxon signed-rank test …')
    out_dir = f'{OUT_ROOT}/2_wilcoxon'
    results = []

    for crop in CROPS:
        label = crop['label']
        series = delta_df[delta_df['crop'] == label].sort_values('year')['mean_delta'].values
        valid  = series[np.isfinite(series)]

        if len(valid) < 4:
            print(f'  ⚠ {label}: too few valid years')
            continue

        stat, p_two = wilcoxon(valid, alternative='two-sided')
        _,    p_pos = wilcoxon(valid, alternative='greater')
        _,    p_neg = wilcoxon(valid, alternative='less')

        results.append({
            'crop'              : label,
            'median_delta'      : round(float(np.median(valid)), 6),
            'pct_years_positive': round(float(np.mean(valid > 0) * 100), 1),
            'wilcoxon_stat'     : round(stat, 2),
            'p_two_sided'       : round(p_two, 4),
            'p_greater_zero'    : round(p_pos, 4),
            'p_less_zero'       : round(p_neg, 4),
            'sig_positive'      : p_pos < 0.05,
            'sig_negative'      : p_neg < 0.05,
            'n_years'           : len(valid),
        })
        print(f'  {label}: median={np.median(valid):.5f}, '
              f'p(greater)={p_pos:.4f}, {np.mean(valid>0)*100:.0f}% yrs positive')

    df = pd.DataFrame(results)
    df.to_csv(f'{out_dir}/wilcoxon_suit_results.csv', index=False)

    # Summary figure
    df_s = df.sort_values('median_delta')
    colors = ['#2166AC' if s else '#AAAAAA' for s in df_s['sig_positive']]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    bars = axes[0].barh(df_s['crop'], df_s['median_delta'],
                        color=colors, edgecolor='white')
    axes[0].axvline(0, color='black', linewidth=0.8)
    for bar, (_, row) in zip(bars, df_s.iterrows()):
        x = bar.get_width()
        axes[0].text(x + 0.0001 * np.sign(x) if x != 0 else 0.0001,
                     bar.get_y() + bar.get_height() / 2,
                     f'p={row["p_greater_zero"]:.3f}'
                     f'{"★" if row["sig_positive"] else ""}',
                     va='center', fontsize=9,
                     ha='left' if x >= 0 else 'right')
    axes[0].set_xlabel('Median ΔSuitability (class units)', fontsize=11)
    axes[0].set_title('Median Annual ΔSuitability\n(Blue = significantly > 0)',
                      fontsize=11, fontweight='bold')

    axes[1].barh(df_s['crop'], df_s['pct_years_positive'],
                 color=colors, edgecolor='white')
    axes[1].axvline(50, color='black', linewidth=0.8, linestyle='--',
                    label='50% reference')
    axes[1].set_xlabel('% of Years with Positive ΔSuitability', fontsize=11)
    axes[1].set_title('% Years Observed > Counterfactual\n(Blue = significantly > 0)',
                      fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].set_xlim(0, 100)

    fig.suptitle('Is Permafrost Thaw Consistently Improving Suitability?\n'
                 'Wilcoxon Signed-Rank Test on Annual ΔSuitability (1999–2018)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{out_dir}/wilcoxon_suit_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(df[['crop', 'median_delta', 'pct_years_positive',
              'p_greater_zero', 'sig_positive']].to_string(index=False))
    return df


# ── Analysis 3: Mann-Kendall on ΔSuitability trend ────────────────────────────

def analysis_mk_delta(delta_df, mask):
    print('\n[Analysis 3] Mann-Kendall on ΔSuitability trend …')
    out_dir = f'{OUT_ROOT}/3_mk_delta'
    results = []

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        series = delta_df[delta_df['crop'] == label].sort_values('year')['mean_delta'].values
        years  = np.array(YEARS_CF)
        valid_mask = np.isfinite(series)
        valid_years  = years[valid_mask]
        valid_series = series[valid_mask]

        if len(valid_series) < 4:
            continue

        mk = mk_test(valid_series)
        results.append({
            'crop'       : label,
            'tau'        : round(mk.Tau, 3),
            'p_value'    : round(mk.p, 4),
            'slope_sen'  : round(mk.slope, 6),
            'trend'      : mk.trend,
            'significant': mk.p < 0.05,
            'n_years'    : int(valid_mask.sum()),
        })

        # Per-crop time series plot
        sen_line = mk.intercept + mk.slope * np.arange(len(valid_series))
        colors   = ['#2166AC' if v >= 0 else '#D6604D' for v in valid_series]
        fig, ax  = plt.subplots(figsize=(10, 4))
        ax.bar(valid_years, valid_series, color=colors,
               edgecolor='white', width=0.8, alpha=0.85)
        ax.plot(valid_years, sen_line, color='black', linewidth=1.8,
                linestyle='--',
                label=f"Sen's slope: {mk.slope:.6f} class/yr")
        ax.axhline(0, color='black', linewidth=0.8)
        sig_str = f'p = {mk.p:.3f} {"★ significant" if mk.p < 0.05 else "(not significant)"}'
        ax.set_title(
            f'{label} — Temporal Trend in Regional Mean ΔSuitability\n'
            f'τ = {mk.Tau:.3f}, {sig_str}',
            fontsize=12, fontweight='bold'
        )
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Regional Mean ΔSuitability (class units)', fontsize=11)
        ax.legend(fontsize=10)
        ax.set_xticks(YEARS_CF)
        ax.set_xticklabels(YEARS_CF, rotation=45, ha='right', fontsize=8)
        plt.tight_layout()
        fig.savefig(f'{out_dir}/{tag}_mk_delta_suit.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  ✓ {label}: τ={mk.Tau:.3f}, p={mk.p:.4f}, slope={mk.slope:.6f}')

    df = pd.DataFrame(results)
    df.to_csv(f'{out_dir}/mk_delta_suit_results.csv', index=False)

    # Summary bar chart
    df_s = df.sort_values('tau')
    colors = ['#2166AC' if s else '#AAAAAA' for s in df_s['significant']]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df_s['crop'], df_s['tau'], color=colors, edgecolor='white')
    ax.axvline(0, color='black', linewidth=0.8)
    for bar, (_, row) in zip(bars, df_s.iterrows()):
        x = bar.get_width()
        ax.text(x + 0.005 * np.sign(x),
                bar.get_y() + bar.get_height() / 2,
                f'p={row["p_value"]:.3f}{"★" if row["significant"] else ""}',
                va='center', fontsize=9,
                ha='left' if x >= 0 else 'right')
    ax.set_xlabel("Kendall's τ", fontsize=12)
    ax.set_title(
        'Mann-Kendall Trend in Regional Mean ΔSuitability (1999–2018)\n'
        'Blue = significant (p < 0.05), Grey = not significant',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    fig.savefig(f'{out_dir}/mk_delta_suit_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(df[['crop', 'tau', 'p_value', 'slope_sen', 'trend', 'significant']].to_string(index=False))
    return df


# ── Analysis 4: 40-year trend on mean suitability (all pixels) ────────────────

def analysis_trend_40yr(mask):
    print('\n[Analysis 4] 40-year trend on mean suitability score …')
    out_dir  = f'{OUT_ROOT}/4_trend_40yr'
    years_arr = np.array(YEARS_ALL)
    post_mask = years_arr >= DIVERGENCE_YEAR
    all_results = []

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        print(f'  {label}')

        obs_suit_s = []
        cf_suit_s  = []

        for year in YEARS_ALL:
            arr, _ = load_raster(obs_suit_path(tag, year))
            obs_suit_s.append(regional_mean_suit(arr, mask) if arr is not None else np.nan)

            if year < DIVERGENCE_YEAR:
                cf_suit_s.append(obs_suit_s[-1])
            else:
                arr_cf, _ = load_raster(cf_suit_path(tag, year))
                cf_suit_s.append(regional_mean_suit(arr_cf, mask)
                                 if arr_cf is not None else np.nan)

        obs_suit_s = np.array(obs_suit_s)
        cf_suit_s  = np.array(cf_suit_s)

        for period, idx in [('1979-2018', slice(None)), ('1999-2018', post_mask)]:
            mk_obs = run_mk(obs_suit_s[idx], years_arr[idx])
            mk_cf  = run_mk(cf_suit_s[idx],  years_arr[idx])
            if mk_obs and mk_cf:
                all_results.append({
                    'crop'            : label,
                    'period'          : period,
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
                    'slope_difference': round(mk_obs['slope_sen'] - mk_cf['slope_sen'], 6),
                })

        # Per-crop plot
        mk_obs_40 = run_mk(obs_suit_s, years_arr)
        mk_cf_40  = run_mk(cf_suit_s,  years_arr)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(years_arr, obs_suit_s, color='#2166AC', linewidth=2,
                marker='o', markersize=4, label='Observed')
        ax.plot(years_arr, cf_suit_s,  color='#D6604D', linewidth=2,
                marker='s', markersize=4, linestyle='--', label='No-Thaw CF')
        if mk_obs_40:
            ax.plot(years_arr, mk_obs_40['sen_line'], color='#2166AC',
                    linewidth=1.5, linestyle=':', alpha=0.8,
                    label=f"Obs slope: {mk_obs_40['slope_sen']:.5f} class/yr "
                          f"(p={mk_obs_40['p_value']:.3f})")
        if mk_cf_40:
            ax.plot(years_arr, mk_cf_40['sen_line'], color='#D6604D',
                    linewidth=1.5, linestyle=':', alpha=0.8,
                    label=f"CF slope: {mk_cf_40['slope_sen']:.5f} class/yr "
                          f"(p={mk_cf_40['p_value']:.3f})")
        ax.axvline(DIVERGENCE_YEAR, color='grey', linestyle='--',
                   linewidth=1.2, label='Divergence (1999)')
        ax.axvspan(DIVERGENCE_YEAR, YEARS_ALL[-1], alpha=0.04, color='grey')
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Mean Suitability Score (all pixels)', fontsize=11)
        ax.set_title(
            f'{label} — Mean Suitability Score: Observed vs. No-Thaw (1979–2018)',
            fontsize=12, fontweight='bold'
        )
        ax.legend(fontsize=9, loc='upper left')
        ax.set_xticks(years_arr[::4])
        ax.set_xticklabels(years_arr[::4], rotation=45, ha='right', fontsize=8)
        plt.tight_layout()
        fig.savefig(f'{out_dir}/{tag}_suit_trend.png', dpi=150, bbox_inches='tight')
        plt.close()

    df = pd.DataFrame(all_results)
    df.to_csv(f'{out_dir}/trend_suit_results.csv', index=False)

    # Summary slope comparison figure
    for period in ['1979-2018', '1999-2018']:
        df_p = df[df['period'] == period].sort_values('obs_slope')
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
        ax.set_ylabel("Sen's Slope (class units/yr)", fontsize=12)
        ax.set_title(
            f'Mean Suitability Trend: Observed vs. No-Thaw ({period})\n'
            f'★ = significant at p < 0.05',
            fontsize=12, fontweight='bold'
        )
        ax.legend(fontsize=11)
        plt.tight_layout()
        fig.savefig(f'{out_dir}/slope_suit_{period.replace("-","_")}.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

    print(df.to_string(index=False))
    return df


# ── Overall aggregate for Analysis 4 ─────────────────────────────────────────

def analysis_trend_40yr_overall(mask):
    print('\n  [Overall Aggregate]')
    out_dir   = f'{OUT_ROOT}/4_trend_40yr'
    years_arr = np.array(YEARS_ALL)
    post_mask = years_arr >= DIVERGENCE_YEAR

    obs_all = []
    cf_all  = []

    for crop in CROPS:
        tag = crop['tag']
        obs_s, cf_s = [], []

        for year in YEARS_ALL:
            arr, _ = load_raster(obs_suit_path(tag, year))
            obs_s.append(regional_mean_suit(arr, mask) if arr is not None else np.nan)

            if year < DIVERGENCE_YEAR:
                cf_s.append(obs_s[-1])
            else:
                arr_cf, _ = load_raster(cf_suit_path(tag, year))
                cf_s.append(regional_mean_suit(arr_cf, mask)
                            if arr_cf is not None else np.nan)

        obs_all.append(np.array(obs_s))
        cf_all.append(np.array(cf_s))

    obs_all = np.array(obs_all)
    cf_all  = np.array(cf_all)

    # Average across crops (suitability scores are on same scale so no normalization needed)
    obs_mean = np.nanmean(obs_all, axis=0)
    cf_mean  = np.nanmean(cf_all,  axis=0)

    results = []
    for period, idx in [('1979-2018', slice(None)), ('1999-2018', post_mask)]:
        mk_obs = run_mk(obs_mean[idx], years_arr[idx])
        mk_cf  = run_mk(cf_mean[idx],  years_arr[idx])
        if mk_obs and mk_cf:
            results.append({
                'period'          : period,
                'obs_tau'         : mk_obs['tau'],
                'obs_p'           : mk_obs['p_value'],
                'obs_slope'       : mk_obs['slope_sen'],
                'obs_significant' : mk_obs['significant'],
                'cf_tau'          : mk_cf['tau'],
                'cf_p'            : mk_cf['p_value'],
                'cf_slope'        : mk_cf['slope_sen'],
                'cf_significant'  : mk_cf['significant'],
                'slope_difference': round(mk_obs['slope_sen'] - mk_cf['slope_sen'], 6),
            })
            print(f'  [{period}] obs: τ={mk_obs["tau"]:.3f} p={mk_obs["p_value"]:.4f} '
                  f'slope={mk_obs["slope_sen"]:.5f} | '
                  f'cf: τ={mk_cf["tau"]:.3f} p={mk_cf["p_value"]:.4f} '
                  f'slope={mk_cf["slope_sen"]:.5f} | '
                  f'Δslope={mk_obs["slope_sen"]-mk_cf["slope_sen"]:.5f}')

    # Overall plot
    mk_obs_40 = run_mk(obs_mean, years_arr)
    mk_cf_40  = run_mk(cf_mean,  years_arr)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(years_arr, obs_mean, color='#2166AC', linewidth=2,
            marker='o', markersize=4, label='Observed')
    ax.plot(years_arr, cf_mean,  color='#D6604D', linewidth=2,
            marker='s', markersize=4, linestyle='--', label='No-Thaw CF')
    if mk_obs_40:
        ax.plot(years_arr, mk_obs_40['sen_line'], color='#2166AC',
                linewidth=1.5, linestyle=':', alpha=0.8,
                label=f"Obs slope: {mk_obs_40['slope_sen']:.5f} class/yr "
                      f"(p={mk_obs_40['p_value']:.3f})")
    if mk_cf_40:
        ax.plot(years_arr, mk_cf_40['sen_line'], color='#D6604D',
                linewidth=1.5, linestyle=':', alpha=0.8,
                label=f"CF slope: {mk_cf_40['slope_sen']:.5f} class/yr "
                      f"(p={mk_cf_40['p_value']:.3f})")
    ax.axvline(DIVERGENCE_YEAR, color='grey', linestyle='--',
               linewidth=1.2, label='Divergence (1999)')
    ax.axvspan(DIVERGENCE_YEAR, YEARS_ALL[-1], alpha=0.04, color='grey')
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Mean Suitability Score — All Crops (class units)', fontsize=11)
    ax.set_title('Overall Mean Suitability: Observed vs. No-Thaw (1979–2018)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xticks(years_arr[::4])
    ax.set_xticklabels(years_arr[::4], rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    fig.savefig(f'{out_dir}/overall_suit_trend.png', dpi=150, bbox_inches='tight')
    plt.close()

    pd.DataFrame(results).to_csv(f'{out_dir}/overall_suit_trend_results.csv', index=False)
    print(f'  ✓ Overall aggregate saved.')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    mask = load_mask()

    delta_df = analysis_delta_maps(mask)
    analysis_wilcoxon(delta_df)
    analysis_mk_delta(delta_df, mask)
    analysis_trend_40yr(mask)
    analysis_trend_40yr_overall(mask)

    print(f'\n✓ All analyses complete. Outputs in: {OUT_ROOT}/')