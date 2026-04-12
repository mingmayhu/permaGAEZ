"""
Suitability Score Analysis
==========================
Using fixed-boundary suitability classes (0–5) from reclassification script.

Analyses:
  1. Annual ΔSuitability maps (Thaw − counterfactual, 1999–2018)
     + small-multiples panel per crop
  2. Wilcoxon signed-rank test on regional mean ΔSuitability (1999–2018)
     — is Thaw suitability consistently above counterfactual?
  3. Mann-Kendall trend test on regional mean ΔSuitability (1999–2018)
     — is thaw's contribution growing over time?
  4. 40-year trend analysis on mean suitability score and % suitable (>=2)
     — classes 0 and 1 combined into class 1, range 1-5
     — includes bootstrap slope difference test

Outputs written to: ./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/
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
OUT_ROOT  = './results/permafrost_thaw_impact/thaw_vs_nothaw/outputs'

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

def remap(arr_int):
    """Combine class 0 into class 1. Returns array with values in 1-5."""
    out = arr_int.copy()
    out[out == 0] = 1
    return out

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

PERMAFROST_PATH = r'./data_input/permafrost_qilian.tif'

def load_mask():
    arr, _ = load_raster(MASK_PATH)
    mask = arr.astype(bool)
    # Exclude lake pixels (nodata or 0 in the permafrost map)
    pf_arr, _ = load_raster(PERMAFROST_PATH)
    if pf_arr is not None:
        lake_mask = ((pf_arr == 0) | ~np.isfinite(pf_arr)) & mask
        mask[lake_mask] = False
        print(f'  Excluded {lake_mask.sum()} lake/nodata pixels from mask')
    return mask

def obs_suit_path(tag, year):
    return f'./data_output/final_classification_fixed/{tag}/{year}_suitability_class.tif'

def cf_suit_path(tag, year):
    return f'./data_output/final_classification_nothaw_fixed/{tag}/{year}_suitability_class.tif'

def regional_mean_suit(arr, mask):
    """Mean suitability across ALL mask pixels.
    Classes 0 and 1 combined into class 1. Range is 1-5.
    """
    arr_clean = arr.copy()
    arr_clean[arr_clean < 0] = np.nan
    arr_int = np.where(np.isfinite(arr_clean), arr_clean, np.nan)
    arr_int = np.where(np.isfinite(arr_int),
                       remap(np.where(np.isfinite(arr_int), arr_int, 0).astype(int)).astype(float),
                       np.nan)
    valid = mask & np.isfinite(arr_int)
    return float(np.nanmean(arr_int[valid])) if valid.any() else np.nan

def regional_pct_ge2(arr, mask):
    """% of ALL mask pixels with class >= 2 (marginally suitable or better).
    Computed over all mask pixels to capture spatial expansion.
    """
    arr_clean = arr.copy()
    arr_clean[arr_clean < 0] = np.nan
    arr_int = np.where(np.isfinite(arr_clean),
                       np.where(np.isfinite(arr_clean), arr_clean, 0).astype(int),
                       0).astype(int)
    arr_int = np.clip(arr_int, 0, 5)
    if not mask.any():
        return np.nan
    return float(np.mean(arr_int[mask] >= 2) * 100)

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

def test_slope_difference(obs_s, cf_s, n_iter=1000):
    """Bootstrap and permutation tests for whether slope difference
    (obs - CF) is significantly different from zero.

    Bootstrap: resamples years with replacement independently for each series.
    Permutation: randomly swaps obs/cf labels for each year, preserving
                 the paired temporal structure.

    Returns dict with keys:
        boot_mean, boot_ci_lo, boot_ci_hi, boot_p
        perm_mean, perm_ci_lo, perm_ci_hi, perm_p
    """
    obs = np.array(obs_s, dtype=float)
    cf  = np.array(cf_s,  dtype=float)
    valid = np.isfinite(obs) & np.isfinite(cf)
    if valid.sum() < 4:
        nan6 = (np.nan, np.nan, np.nan, np.nan)
        return dict(boot_mean=np.nan, boot_ci_lo=np.nan, boot_ci_hi=np.nan, boot_p=np.nan,
                    perm_mean=np.nan, perm_ci_lo=np.nan, perm_ci_hi=np.nan, perm_p=np.nan)

    obs_v = obs[valid]
    cf_v  = cf[valid]
    n     = len(obs_v)
    actual_diff = mk_test(obs_v).slope - mk_test(cf_v).slope

    # ── Bootstrap (resample with replacement) ─────────────────────────────────
    boot_diffs = []
    for _ in range(n_iter):
        idx  = np.random.choice(n, size=n, replace=True)
        mk_o = mk_test(obs_v[idx])
        mk_c = mk_test(cf_v[idx])
        boot_diffs.append(mk_o.slope - mk_c.slope)
    boot_diffs = np.array(boot_diffs)
    boot_p     = float(np.mean(np.abs(boot_diffs) >= np.abs(actual_diff)))

    # ── Permutation (swap obs/cf labels per year) ─────────────────────────────
    perm_diffs = []
    for _ in range(n_iter):
        swap     = np.random.rand(n) > 0.5
        obs_perm = np.where(swap, cf_v, obs_v)
        cf_perm  = np.where(swap, obs_v, cf_v)
        mk_o     = mk_test(obs_perm)
        mk_c     = mk_test(cf_perm)
        perm_diffs.append(mk_o.slope - mk_c.slope)
    perm_diffs = np.array(perm_diffs)
    perm_p     = float(np.mean(np.abs(perm_diffs) >= np.abs(actual_diff)))

    return dict(
        boot_mean  = round(float(np.mean(boot_diffs)), 6),
        boot_ci_lo = round(float(np.percentile(boot_diffs, 2.5)), 6),
        boot_ci_hi = round(float(np.percentile(boot_diffs, 97.5)), 6),
        boot_p     = round(boot_p, 4),
        perm_mean  = round(float(np.mean(perm_diffs)), 6),
        perm_ci_lo = round(float(np.percentile(perm_diffs, 2.5)), 6),
        perm_ci_hi = round(float(np.percentile(perm_diffs, 97.5)), 6),
        perm_p     = round(perm_p, 4),
    )


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

            # Remap class 0 -> 1 before computing delta
            obs_r = np.where(np.isfinite(obs),
                             remap(np.where(np.isfinite(obs), obs, 0).astype(int)).astype(float),
                             np.nan)
            cf_r  = np.where(np.isfinite(cf),
                             remap(np.where(np.isfinite(cf), cf, 0).astype(int)).astype(float),
                             np.nan)

            delta = np.where(np.isfinite(obs_r) & np.isfinite(cf_r),
                             obs_r - cf_r, np.nan)
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
            f'{label} — Annual ΔSuitability (Thaw − No-Thaw)\n'
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
        ax2.set_title(f'{label} — Annual Mean ΔSuitability (Thaw − No-Thaw)',
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
    axes[1].set_title('% Years Thaw > Counterfactual\n(Blue = significantly > 0)',
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
        valid_mask   = np.isfinite(series)
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


# ── Analysis 4: 40-year trend on mean suitability and % suitable ──────────────

def analysis_trend_40yr(mask):
    print('\n[Analysis 4] 40-year trend on mean suitability and % suitable …')
    out_dir   = f'{OUT_ROOT}/4_trend_40yr'
    years_arr = np.array(YEARS_ALL)
    post_mask = years_arr >= DIVERGENCE_YEAR
    all_results = []

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        print(f'  {label}')

        obs_suit_s = []
        cf_suit_s  = []
        obs_pct_s  = []
        cf_pct_s   = []

        for year in YEARS_ALL:
            arr, _ = load_raster(obs_suit_path(tag, year))
            obs_suit_s.append(regional_mean_suit(arr, mask) if arr is not None else np.nan)
            obs_pct_s.append(regional_pct_ge2(arr, mask)   if arr is not None else np.nan)

            if year < DIVERGENCE_YEAR:
                cf_suit_s.append(obs_suit_s[-1])
                cf_pct_s.append(obs_pct_s[-1])
            else:
                arr_cf, _ = load_raster(cf_suit_path(tag, year))
                cf_suit_s.append(regional_mean_suit(arr_cf, mask) if arr_cf is not None else np.nan)
                cf_pct_s.append(regional_pct_ge2(arr_cf, mask)   if arr_cf is not None else np.nan)

        obs_suit_s = np.array(obs_suit_s)
        cf_suit_s  = np.array(cf_suit_s)
        obs_pct_s  = np.array(obs_pct_s)
        cf_pct_s   = np.array(cf_pct_s)

        # MK + bootstrap for both metrics, both periods
        for period, idx in [('1979-2018', slice(None)), ('1999-2018', post_mask)]:
            for metric, obs_s, cf_s, unit in [
                ('mean_suit', obs_suit_s, cf_suit_s, 'class/yr'),
                ('pct_ge2',   obs_pct_s,  cf_pct_s,  '%/yr'),
            ]:
                mk_obs = run_mk(obs_s[idx], years_arr[idx])
                mk_cf  = run_mk(cf_s[idx],  years_arr[idx])
                sd = test_slope_difference(obs_s[idx], cf_s[idx])

                if mk_obs and mk_cf:
                    all_results.append({
                        'crop'              : label,
                        'period'            : period,
                        'metric'            : metric,
                        'obs_tau'           : mk_obs['tau'],
                        'obs_p'             : mk_obs['p_value'],
                        'obs_slope'         : mk_obs['slope_sen'],
                        'obs_trend'         : mk_obs['trend'],
                        'obs_significant'   : mk_obs['significant'],
                        'cf_tau'            : mk_cf['tau'],
                        'cf_p'             : mk_cf['p_value'],
                        'cf_slope'          : mk_cf['slope_sen'],
                        'cf_trend'          : mk_cf['trend'],
                        'cf_significant'    : mk_cf['significant'],
                        'slope_difference'  : round(mk_obs['slope_sen'] - mk_cf['slope_sen'], 6),
                        'boot_ci_lo'        : sd['boot_ci_lo'],
                        'boot_ci_hi'        : sd['boot_ci_hi'],
                        'boot_p'            : sd['boot_p'],
                        'boot_sig'          : (not np.isnan(sd['boot_p'])) and sd['boot_p'] < 0.05,
                        'perm_ci_lo'        : sd['perm_ci_lo'],
                        'perm_ci_hi'        : sd['perm_ci_hi'],
                        'perm_p'            : sd['perm_p'],
                        'perm_sig'          : (not np.isnan(sd['perm_p'])) and sd['perm_p'] < 0.05,
                    })

        # Per-crop plot: 2 panels
        mk_obs_suit = run_mk(obs_suit_s, years_arr)
        mk_cf_suit  = run_mk(cf_suit_s,  years_arr)
        mk_obs_pct  = run_mk(obs_pct_s,  years_arr)
        mk_cf_pct   = run_mk(cf_pct_s,   years_arr)

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        for ax, obs_s, cf_s, mk_obs, mk_cf, ylabel, title, unit in [
            (axes[0], obs_suit_s, cf_suit_s, mk_obs_suit, mk_cf_suit,
             'Mean Suitability Score (1–5)', 'Mean Suitability', 'class/yr'),
            (axes[1], obs_pct_s,  cf_pct_s,  mk_obs_pct,  mk_cf_pct,
             '% Pixels with Class ≥ 2', '% Suitable Land', '%/yr'),
        ]:
            ax.plot(years_arr, obs_s, color='#2166AC', linewidth=2,
                    marker='o', markersize=4, label='Thaw')
            ax.plot(years_arr, cf_s, color='#D6604D', linewidth=2,
                    marker='s', markersize=4, linestyle='--', label='No-Thaw CF')
            if mk_obs:
                ax.plot(years_arr, mk_obs['sen_line'], color='#2166AC',
                        linewidth=1.5, linestyle=':', alpha=0.8,
                        label=f"Thaw slope: {mk_obs['slope_sen']:.5f} {unit} "
                              f"(p={mk_obs['p_value']:.3f})")
            if mk_cf:
                ax.plot(years_arr, mk_cf['sen_line'], color='#D6604D',
                        linewidth=1.5, linestyle=':', alpha=0.8,
                        label=f"No-Thaw slope: {mk_cf['slope_sen']:.5f} {unit} "
                              f"(p={mk_cf['p_value']:.3f})")
            ax.axvline(DIVERGENCE_YEAR, color='grey', linestyle='--',
                       linewidth=1.2, label='Divergence (1999)')
            ax.axvspan(DIVERGENCE_YEAR, YEARS_ALL[-1], alpha=0.04, color='grey')
            ax.set_xlabel('Year', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f'{label} — {title}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9, loc='upper left')
            ax.set_xticks(years_arr[::4])
            ax.set_xticklabels(years_arr[::4], rotation=45, ha='right', fontsize=8)

        fig.suptitle(f'{label} — Thaw vs. No-Thaw Suitability (1979–2018)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        fig.savefig(f'{out_dir}/{tag}_suit_trend.png', dpi=150, bbox_inches='tight')
        plt.close()

    df = pd.DataFrame(all_results)
    df.to_csv(f'{out_dir}/trend_suit_results.csv', index=False)

    # Summary slope comparison figures — one per metric per period
    for metric, ylabel in [('mean_suit', "Sen's Slope (class/yr)"),
                            ('pct_ge2',  "Sen's Slope (%/yr)")]:
        for period in ['1979-2018', '1999-2018']:
            df_p = df[(df['period'] == period) & (df['metric'] == metric)].sort_values('obs_slope')
            if df_p.empty:
                continue
            x     = np.arange(len(df_p))
            width = 0.35
            fig, ax = plt.subplots(figsize=(13, 6))
            bars_obs = ax.bar(x - width/2, df_p['obs_slope'], width,
                              label='Thaw', color='#2166AC', alpha=0.85)
            bars_cf  = ax.bar(x + width/2, df_p['cf_slope'],  width,
                              label='No-Thaw CF', color='#D6604D', alpha=0.85)
            for bars, sig_col in [(bars_obs, 'obs_significant'),
                                  (bars_cf,  'cf_significant')]:
                for bar, (_, row) in zip(bars, df_p.iterrows()):
                    if row[sig_col]:
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + abs(bar.get_height()) * 0.02,
                                '★', ha='center', va='bottom', fontsize=11)
            # Mark bootstrap and permutation significant slope differences
            for xi, (_, row) in zip(x, df_p.iterrows()):
                markers = []
                if row.get('boot_sig', False): markers.append('b')
                if row.get('perm_sig', False): markers.append('p')
                if markers:
                    ax.text(xi, max(row['obs_slope'], row['cf_slope']) + 0.0001,
                            '†', ha='center', va='bottom', fontsize=11, color='green')
            ax.axhline(0, color='black', linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(df_p['crop'], rotation=30, ha='right', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(
                f'Suitability Trend ({metric}): Thaw vs. No-Thaw ({period})\n'
                f'★ = MK significant (p < 0.05), † = bootstrap slope diff significant',
                fontsize=12, fontweight='bold'
            )
            ax.legend(fontsize=11)
            plt.tight_layout()
            fig.savefig(f'{out_dir}/slope_{metric}_{period.replace("-","_")}.png',
                        dpi=150, bbox_inches='tight')
            plt.close()

    # Print bootstrap and permutation results
    # Note: overall is appended by analysis_trend_40yr_overall — reload after that call
    print('\nSlope difference results (1979-2018, per crop):')
    df_boot = df[df['period'] == '1979-2018'][
        ['crop', 'metric', 'slope_difference',
         'boot_ci_lo', 'boot_ci_hi', 'boot_p', 'boot_sig',
         'perm_ci_lo', 'perm_ci_hi', 'perm_p', 'perm_sig']]
    print(df_boot.to_string(index=False))

    return df


# ── Overall aggregate for Analysis 4 ─────────────────────────────────────────

def analysis_trend_40yr_overall(mask):
    print('\n  [Overall Aggregate]')
    out_dir   = f'{OUT_ROOT}/4_trend_40yr'
    years_arr = np.array(YEARS_ALL)
    post_mask = years_arr >= DIVERGENCE_YEAR

    obs_suit_all = []
    cf_suit_all  = []
    obs_pct_all  = []
    cf_pct_all   = []

    for crop in CROPS:
        tag = crop['tag']
        obs_s, cf_s, obs_p, cf_p = [], [], [], []

        for year in YEARS_ALL:
            arr, _ = load_raster(obs_suit_path(tag, year))
            obs_s.append(regional_mean_suit(arr, mask) if arr is not None else np.nan)
            obs_p.append(regional_pct_ge2(arr, mask)   if arr is not None else np.nan)

            if year < DIVERGENCE_YEAR:
                cf_s.append(obs_s[-1])
                cf_p.append(obs_p[-1])
            else:
                arr_cf, _ = load_raster(cf_suit_path(tag, year))
                cf_s.append(regional_mean_suit(arr_cf, mask) if arr_cf is not None else np.nan)
                cf_p.append(regional_pct_ge2(arr_cf, mask)   if arr_cf is not None else np.nan)

        obs_suit_all.append(np.array(obs_s))
        cf_suit_all.append(np.array(cf_s))
        obs_pct_all.append(np.array(obs_p))
        cf_pct_all.append(np.array(cf_p))

    obs_suit_mean = np.nanmean(np.array(obs_suit_all), axis=0)
    cf_suit_mean  = np.nanmean(np.array(cf_suit_all),  axis=0)
    obs_pct_mean  = np.nanmean(np.array(obs_pct_all),  axis=0)
    cf_pct_mean   = np.nanmean(np.array(cf_pct_all),   axis=0)

    results = []
    for period, idx in [('1979-2018', slice(None)), ('1999-2018', post_mask)]:
        for metric, obs_s, cf_s in [
            ('mean_suit', obs_suit_mean, cf_suit_mean),
            ('pct_ge2',   obs_pct_mean,  cf_pct_mean),
        ]:
            mk_obs = run_mk(obs_s[idx], years_arr[idx])
            mk_cf  = run_mk(cf_s[idx],  years_arr[idx])
            sd = test_slope_difference(obs_s[idx], cf_s[idx])

            if mk_obs and mk_cf:
                results.append({
                    'period'           : period,
                    'metric'           : metric,
                    'obs_tau'          : mk_obs['tau'],
                    'obs_p'            : mk_obs['p_value'],
                    'obs_slope'        : mk_obs['slope_sen'],
                    'obs_significant'  : mk_obs['significant'],
                    'cf_tau'           : mk_cf['tau'],
                    'cf_p'             : mk_cf['p_value'],
                    'cf_slope'         : mk_cf['slope_sen'],
                    'cf_significant'   : mk_cf['significant'],
                    'slope_difference' : round(mk_obs['slope_sen'] - mk_cf['slope_sen'], 6),
                    'boot_ci_lo'       : sd['boot_ci_lo'],
                    'boot_ci_hi'       : sd['boot_ci_hi'],
                    'boot_p'           : sd['boot_p'],
                    'boot_sig'         : (not np.isnan(sd['boot_p'])) and sd['boot_p'] < 0.05,
                    'perm_ci_lo'       : sd['perm_ci_lo'],
                    'perm_ci_hi'       : sd['perm_ci_hi'],
                    'perm_p'           : sd['perm_p'],
                    'perm_sig'         : (not np.isnan(sd['perm_p'])) and sd['perm_p'] < 0.05,
                })
                print(f'  [{period}] {metric}: obs slope={mk_obs["slope_sen"]:.5f} '
                      f'(p={mk_obs["p_value"]:.4f}) | cf slope={mk_cf["slope_sen"]:.5f} '
                      f'(p={mk_cf["p_value"]:.4f}) | Δslope={mk_obs["slope_sen"]-mk_cf["slope_sen"]:.5f} '
                      f'boot_p={sd["boot_p"]} perm_p={sd["perm_p"]}')

    # Overall plot: 2 panels
    mk_obs_suit = run_mk(obs_suit_mean, years_arr)
    mk_cf_suit  = run_mk(cf_suit_mean,  years_arr)
    mk_obs_pct  = run_mk(obs_pct_mean,  years_arr)
    mk_cf_pct   = run_mk(cf_pct_mean,   years_arr)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, obs_s, cf_s, mk_obs, mk_cf, ylabel, title, unit in [
        (axes[0], obs_suit_mean, cf_suit_mean, mk_obs_suit, mk_cf_suit,
         'Mean Suitability Score (1–5)', 'Overall Mean Suitability', 'class/yr'),
        (axes[1], obs_pct_mean,  cf_pct_mean,  mk_obs_pct,  mk_cf_pct,
         '% Pixels with Class ≥ 2', 'Overall % Suitable Land', '%/yr'),
    ]:
        ax.plot(years_arr, obs_s, color='#2166AC', linewidth=2,
                marker='o', markersize=4, label='Thaw')
        ax.plot(years_arr, cf_s, color='#D6604D', linewidth=2,
                marker='s', markersize=4, linestyle='--', label='No-Thaw CF')
        if mk_obs:
            ax.plot(years_arr, mk_obs['sen_line'], color='#2166AC',
                    linewidth=1.5, linestyle=':', alpha=0.8,
                    label=f"Thaw slope: {mk_obs['slope_sen']:.5f} {unit} "
                          f"(p={mk_obs['p_value']:.3f})")
        if mk_cf:
            ax.plot(years_arr, mk_cf['sen_line'], color='#D6604D',
                    linewidth=1.5, linestyle=':', alpha=0.8,
                    label=f"No-Thaw slope: {mk_cf['slope_sen']:.5f} {unit} "
                          f"(p={mk_cf['p_value']:.3f})")
        ax.axvline(DIVERGENCE_YEAR, color='grey', linestyle='--',
                   linewidth=1.2, label='Divergence (1999)')
        ax.axvspan(DIVERGENCE_YEAR, YEARS_ALL[-1], alpha=0.04, color='grey')
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper left')
        ax.set_xticks(years_arr[::4])
        ax.set_xticklabels(years_arr[::4], rotation=45, ha='right', fontsize=8)

    fig.suptitle('Overall Suitability: Thaw vs. No-Thaw (1979–2018)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{out_dir}/overall_suit_trend.png', dpi=150, bbox_inches='tight')
    plt.close()

    df_overall = pd.DataFrame(results)
    df_overall.insert(0, 'crop', 'OVERALL')
    df_overall.to_csv(f'{out_dir}/overall_suit_trend_results.csv', index=False)

    # Append overall to main trend results CSV
    main_csv = f'{out_dir}/trend_suit_results.csv'
    if os.path.exists(main_csv):
        df_main = pd.read_csv(main_csv)
        df_combined = pd.concat([df_main, df_overall], ignore_index=True)
        df_combined.to_csv(main_csv, index=False)
    print(f'  ✓ Overall aggregate saved and appended to trend_suit_results.csv')


# ── Main ──────────────────────────────────────────────────────────────────────

def plot_slope_diff(out_dir):
    """Plot bootstrap and permutation dot plots including overall.
    Called after both analysis_trend_40yr and analysis_trend_40yr_overall
    so the CSV contains per-crop + overall rows.
    """
    main_csv = f'{out_dir}/trend_suit_results.csv'
    if not os.path.exists(main_csv):
        print('  ⚠ trend_suit_results.csv not found, skipping dot plots')
        return

    df_all  = pd.read_csv(main_csv)
    df_boot = df_all[df_all['period'] == '1979-2018'][
        ['crop', 'metric', 'slope_difference',
         'boot_ci_lo', 'boot_ci_hi', 'boot_p', 'boot_sig',
         'perm_ci_lo', 'perm_ci_hi', 'perm_p', 'perm_sig']]

    title_map = {'mean_suit': 'Mean Suitability Score',
                 'pct_ge2':   '% Suitable Land (≥ Class 2)'}

    for test, ci_lo, ci_hi, p_col, sig_col, fname in [
        ('Bootstrap',   'boot_ci_lo', 'boot_ci_hi', 'boot_p', 'boot_sig', 'bootstrap_slope_diff.png'),
        ('Permutation', 'perm_ci_lo', 'perm_ci_hi', 'perm_p', 'perm_sig', 'permutation_slope_diff.png'),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        for ax, metric, xlabel in [
            (axes[0], 'mean_suit', "Sen's Slope Difference (class/yr)"),
            (axes[1], 'pct_ge2',   "Sen's Slope Difference (%/yr)"),
        ]:
            df_crops   = df_boot[(df_boot['metric'] == metric) & (df_boot['crop'] != 'OVERALL')].copy()
            df_ov      = df_boot[(df_boot['metric'] == metric) & (df_boot['crop'] == 'OVERALL')].copy()
            df_m       = pd.concat([df_ov, df_crops.sort_values('slope_difference')], ignore_index=True)

            y      = np.arange(len(df_m))
            colors = ['#000000' if c == 'OVERALL' else ('#2166AC' if s else '#AAAAAA')
                      for c, s in zip(df_m['crop'], df_m[sig_col])]

            ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
            if not df_ov.empty:
                ax.axhline(0.5, color='grey', linewidth=0.8, linestyle=':')

            for yi, (col, (_, row)) in enumerate(zip(colors, df_m.iterrows())):
                ax.plot([row[ci_lo], row[ci_hi]], [yi, yi],
                        color=col, linewidth=2, alpha=0.7)
                ax.scatter(row['slope_difference'], yi, color=col, zorder=5, s=70)
                sig = '★' if row[sig_col] else ''
                ax.text(max(row[ci_hi], row['slope_difference']) * 1.05 + 1e-6,
                        yi, f"p={row[p_col]:.3f}{sig}",
                        va='center', fontsize=8, color=col)

            ax.set_yticks(y)
            ax.set_yticklabels(df_m['crop'], fontsize=10)
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_title(title_map[metric], fontsize=12, fontweight='bold')

        fig.suptitle(f"Does Permafrost Thaw Amplify the Suitability Trend?"
                     f"{test} Test on Sen's Slope Difference (Thaw - No-Thaw)"
                     f"Black = Overall, Blue = significant crop, Grey = not significant",
                     fontsize=12, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.91])
        fig.savefig(f'{out_dir}/{fname}', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  ✓ {test} dot plot saved')


if __name__ == '__main__':
    np.random.seed(42)
    mask = load_mask()

    delta_df = analysis_delta_maps(mask)
    analysis_wilcoxon(delta_df)
    analysis_mk_delta(delta_df, mask)
    analysis_trend_40yr(mask)
    analysis_trend_40yr_overall(mask)
    plot_slope_diff(f'{OUT_ROOT}/4_trend_40yr')

    print(f'\n✓ All analyses complete. Outputs in: {OUT_ROOT}/')