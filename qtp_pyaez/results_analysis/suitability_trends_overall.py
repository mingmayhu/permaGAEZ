"""
40-Year Suitability Score Trend Analysis (1979–2018)
=====================================================
Tracks four suitability-based metrics for both observed and
no-thaw counterfactual scenarios:

  1. Mean suitability score (all pixels, including class 0)
  2. % pixels with class >= 1 (any suitability — frontier expansion)
  3. % pixels with class >= 3 (moderately suitable or better)
  4. Class distribution over time (stacked area chart)

For each metric:
  - Mann-Kendall trend test on both scenarios (full 40 years + post-1999)
  - Sen's slope with significance annotation
  - Side-by-side observed vs counterfactual plots
  - Slope difference (thaw's contribution to trend)

Pre-1999: counterfactual = observed by construction
Post-1999: scenarios diverge

Outputs written to:
  ./results_analysis/outputs/7_suitability_40yr/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pymannkendall import original_test as mk_test
from osgeo import gdal

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR  = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH = r'./data_input/qilian mask.tif'
OUT_ROOT  = './results_analysis/outputs/7_suitability_40yr'

YEARS_ALL       = list(range(1979, 2019))
YEARS_CF        = list(range(1999, 2019))
DIVERGENCE_YEAR = 1999
ALPHA           = 0.05

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

# Class colours for stacked area chart
CLASS_COLORS = {
    0: '#d9d9d9',   # grey   — no suitability
    1: '#fc8d59',   # orange — not suitable
    2: '#fee08b',   # yellow — marginal
    3: '#d9ef8b',   # light green — moderate
    4: '#91cf60',   # green — suitable
    5: '#1a9850',   # dark green — very suitable
}
CLASS_LABELS = {
    0: 'Class 0 (none)',
    1: 'Class 1 (not suitable)',
    2: 'Class 2 (marginal)',
    3: 'Class 3 (moderate)',
    4: 'Class 4 (suitable)',
    5: 'Class 5 (very suitable)',
}

os.chdir(WORK_DIR)
os.makedirs(OUT_ROOT, exist_ok=True)
for sub in ['per_crop', 'overall', 'class_distribution']:
    os.makedirs(f'{OUT_ROOT}/{sub}', exist_ok=True)


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

def obs_suit_path(tag, year):
    return f'./data_output/final_classification_fixed/{tag}/{year}_suitability_class.tif'

def cf_suit_path(tag, year):
    return f'./data_output/final_classification_nothaw_fixed/{tag}/{year}_suitability_class.tif'

def clean(arr, mask):
    out = arr.copy()
    out[~mask] = np.nan
    out[out < 0] = np.nan
    return out

def compute_metrics(arr, mask):
    """
    Compute all four suitability metrics from a class raster.
    Returns dict of scalar values.
    """
    n_total = int(mask.sum())
    if n_total == 0:
        return {m: np.nan for m in
                ['mean_suit', 'pct_ge1', 'pct_ge3',
                 'pct_0','pct_1','pct_2','pct_3','pct_4','pct_5']}

    arr_c = clean(arr, mask)
    # treat NaN as class 0 for percentage calculations
    arr_int = np.where(mask & np.isfinite(arr_c), arr_c, 0).astype(int)
    arr_int = np.clip(arr_int, 0, 5)

    mean_suit = float(np.nanmean(arr_c[mask & np.isfinite(arr_c)]))
    pct_ge1   = float(np.mean(arr_int[mask] >= 1) * 100)
    pct_ge3   = float(np.mean(arr_int[mask] >= 3) * 100)
    pcts      = {f'pct_{c}': float(np.mean(arr_int[mask] == c) * 100)
                 for c in range(6)}

    return {'mean_suit': mean_suit, 'pct_ge1': pct_ge1,
            'pct_ge3': pct_ge3, **pcts}

from scipy.stats import wilcoxon as scipy_wilcoxon

def run_mk(series):
    """Run MK on finite values. Returns dict."""
    s = np.array(series, dtype=float)
    valid = np.isfinite(s)
    if valid.sum() < 4:
        return None
    mk = mk_test(s[valid])
    y_hat = mk.intercept + mk.slope * np.arange(valid.sum())
    line  = np.full(len(s), np.nan)
    line[valid] = y_hat
    return {
        'tau'        : round(mk.Tau, 3),
        'p'          : round(mk.p, 4),
        'slope'      : round(mk.slope, 6),
        'trend'      : mk.trend,
        'significant': mk.p < ALPHA,
        'sen_line'   : line,
    }

def test_slope_difference(obs_s, cf_s, n_boot=1000):
    """
    Bootstrap test for whether the slope difference (obs - CF) is
    significantly different from zero.
    Resamples years with replacement, computes Sen's slope for each
    bootstrap sample in both scenarios, then tests if the distribution
    of slope differences excludes zero.
    Returns (mean_diff, ci_lo, ci_hi, p_value).
    """
    obs = np.array(obs_s, dtype=float)
    cf  = np.array(cf_s,  dtype=float)
    valid = np.isfinite(obs) & np.isfinite(cf)
    if valid.sum() < 4:
        return np.nan, np.nan, np.nan, np.nan

    obs_v = obs[valid]
    cf_v  = cf[valid]
    n     = len(obs_v)

    boot_diffs = []
    for _ in range(n_boot):
        idx       = np.random.choice(n, size=n, replace=True)
        mk_o      = mk_test(obs_v[idx])
        mk_c      = mk_test(cf_v[idx])
        boot_diffs.append(mk_o.slope - mk_c.slope)

    boot_diffs = np.array(boot_diffs)
    mean_diff  = float(np.mean(boot_diffs))
    ci_lo      = float(np.percentile(boot_diffs, 2.5))
    ci_hi      = float(np.percentile(boot_diffs, 97.5))
    # Two-sided p-value: proportion of bootstrap diffs on wrong side of zero
    actual_diff = mk_test(obs_v).slope - mk_test(cf_v).slope
    p_val = float(np.mean(np.abs(boot_diffs) >= np.abs(actual_diff)))

    return round(mean_diff, 6), round(ci_lo, 6), round(ci_hi, 6), round(p_val, 4)

def run_wilcoxon_pair(obs_s, cf_s):
    """
    Wilcoxon signed-rank test on paired annual values (obs vs CF).
    Tests whether observed is consistently higher than counterfactual.
    Only uses post-divergence years (1999-2018) since pre-1999 obs == CF.
    Returns (median_diff, pct_years_positive, p_greater, p_two_sided, significant).
    """
    obs = np.array(obs_s, dtype=float)
    cf  = np.array(cf_s,  dtype=float)
    diff = obs - cf
    valid = np.isfinite(diff)
    if valid.sum() < 4:
        return np.nan, np.nan, np.nan, np.nan, False

    d = diff[valid]
    # Need non-zero differences for Wilcoxon
    nonzero = d[d != 0]
    if len(nonzero) < 4:
        return float(np.median(d)), float(np.mean(d > 0) * 100), np.nan, np.nan, False

    try:
        _, p_two    = scipy_wilcoxon(d, alternative='two-sided')
        _, p_greater = scipy_wilcoxon(d, alternative='greater')
    except Exception:
        return float(np.median(d)), float(np.mean(d > 0) * 100), np.nan, np.nan, False

    return (round(float(np.median(d)), 6),
            round(float(np.mean(d > 0) * 100), 1),
            round(p_greater, 4),
            round(p_two, 4),
            p_greater < ALPHA)

def plot_metric(ax, years_arr, obs_s, cf_s, mk_obs, mk_cf,
                ylabel, title, slope_unit):
    """Plot observed vs CF with Sen's slope lines."""
    ax.plot(years_arr, obs_s, color='#2166AC', linewidth=2,
            marker='o', markersize=3, label='Observed')
    ax.plot(years_arr, cf_s, color='#D6604D', linewidth=2,
            marker='s', markersize=3, linestyle='--', label='No-Thaw CF')

    if mk_obs:
        sig = '★' if mk_obs['significant'] else ''
        ax.plot(years_arr, mk_obs['sen_line'], color='#2166AC',
                linewidth=1.5, linestyle=':', alpha=0.85,
                label=f"Obs slope: {mk_obs['slope']:.5f} {slope_unit} "
                      f"(p={mk_obs['p']:.3f}){sig}")
    if mk_cf:
        sig = '★' if mk_cf['significant'] else ''
        ax.plot(years_arr, mk_cf['sen_line'], color='#D6604D',
                linewidth=1.5, linestyle=':', alpha=0.85,
                label=f"CF slope: {mk_cf['slope']:.5f} {slope_unit} "
                      f"(p={mk_cf['p']:.3f}){sig}")

    ax.axvline(DIVERGENCE_YEAR, color='grey', linestyle='--',
               linewidth=1.2, label='Divergence (1999)')
    ax.axvspan(DIVERGENCE_YEAR, YEARS_ALL[-1], alpha=0.04, color='grey')
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=7.5, loc='upper left')
    ax.set_xticks(years_arr[::4])
    ax.set_xticklabels(years_arr[::4], rotation=45, ha='right', fontsize=8)


# ── Build annual series ───────────────────────────────────────────────────────

def build_series(tag, mask):
    """
    Build annual time series for all metrics, both scenarios.
    Returns dict of arrays length 40.
    """
    metrics = ['mean_suit', 'pct_ge1', 'pct_ge3',
               'pct_0','pct_1','pct_2','pct_3','pct_4','pct_5']
    obs_data = {m: [] for m in metrics}
    cf_data  = {m: [] for m in metrics}

    for year in YEARS_ALL:
        obs = load_raster(obs_suit_path(tag, year))
        m   = compute_metrics(obs, mask) if obs is not None else \
              {k: np.nan for k in metrics}
        for k in metrics:
            obs_data[k].append(m[k])

        if year < DIVERGENCE_YEAR:
            for k in metrics:
                cf_data[k].append(obs_data[k][-1])
        else:
            cf = load_raster(cf_suit_path(tag, year))
            m  = compute_metrics(cf, mask) if cf is not None else \
                 {k: np.nan for k in metrics}
            for k in metrics:
                cf_data[k].append(m[k])

    return ({k: np.array(v) for k, v in obs_data.items()},
            {k: np.array(v) for k, v in cf_data.items()})


# ── Main analysis ─────────────────────────────────────────────────────────────

def run(mask):
    years_arr = np.array(YEARS_ALL)
    post_mask = years_arr >= DIVERGENCE_YEAR
    all_results = []

    # Store for overall aggregate
    obs_all = {m: [] for m in ['mean_suit','pct_ge1','pct_ge3']}
    cf_all  = {m: [] for m in ['mean_suit','pct_ge1','pct_ge3']}
    obs_cls_all = {c: [] for c in range(6)}
    cf_cls_all  = {c: [] for c in range(6)}

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        print(f'  {label} …')

        obs_data, cf_data = build_series(tag, mask)

        # Store for aggregate
        for m in ['mean_suit','pct_ge1','pct_ge3']:
            obs_all[m].append(obs_data[m])
            cf_all[m].append(cf_data[m])
        for c in range(6):
            obs_cls_all[c].append(obs_data[f'pct_{c}'])
            cf_cls_all[c].append(cf_data[f'pct_{c}'])

        # ── MK tests + slope difference + Wilcoxon for three metrics ─────────
        for period, idx, years_idx in [
            ('1979-2018', slice(None), years_arr),
            ('1999-2018', post_mask,   years_arr[post_mask]),
        ]:
            for metric, obs_s, cf_s, unit in [
                ('mean_suit', obs_data['mean_suit'], cf_data['mean_suit'], 'class/yr'),
                ('pct_ge1',   obs_data['pct_ge1'],  cf_data['pct_ge1'],   '%/yr'),
                ('pct_ge3',   obs_data['pct_ge3'],  cf_data['pct_ge3'],   '%/yr'),
            ]:
                mk_obs = run_mk(obs_s[idx])
                mk_cf  = run_mk(cf_s[idx])

                # Bootstrap slope difference test
                sd_mean, sd_lo, sd_hi, sd_p = test_slope_difference(
                    obs_s[idx], cf_s[idx])

                # Wilcoxon on post-divergence paired values only
                wil_med, wil_pct, wil_p_gt, wil_p_2s, wil_sig = \
                    run_wilcoxon_pair(
                        obs_s[post_mask], cf_s[post_mask])

                if mk_obs and mk_cf:
                    all_results.append({
                        'crop'              : label,
                        'period'            : period,
                        'metric'            : metric,
                        'obs_tau'           : mk_obs['tau'],
                        'obs_p'             : mk_obs['p'],
                        'obs_slope'         : mk_obs['slope'],
                        'obs_trend'         : mk_obs['trend'],
                        'obs_significant'   : mk_obs['significant'],
                        'cf_tau'            : mk_cf['tau'],
                        'cf_p'              : mk_cf['p'],
                        'cf_slope'          : mk_cf['slope'],
                        'cf_trend'          : mk_cf['trend'],
                        'cf_significant'    : mk_cf['significant'],
                        'slope_difference'  : round(mk_obs['slope'] - mk_cf['slope'], 6),
                        'slope_diff_ci_lo'  : sd_lo,
                        'slope_diff_ci_hi'  : sd_hi,
                        'slope_diff_p'      : sd_p,
                        'slope_diff_sig'    : (not np.isnan(sd_p)) and sd_p < ALPHA,
                        'wilcoxon_median_diff'  : wil_med,
                        'wilcoxon_pct_pos'      : wil_pct,
                        'wilcoxon_p_greater'    : wil_p_gt,
                        'wilcoxon_p_two_sided'  : wil_p_2s,
                        'wilcoxon_sig'          : wil_sig,
                    })

        # ── Per-crop figure: 3 metric panels ─────────────────────────────────
        mk_ms_obs = run_mk(obs_data['mean_suit'])
        mk_ms_cf  = run_mk(cf_data['mean_suit'])
        mk_g1_obs = run_mk(obs_data['pct_ge1'])
        mk_g1_cf  = run_mk(cf_data['pct_ge1'])
        mk_g3_obs = run_mk(obs_data['pct_ge3'])
        mk_g3_cf  = run_mk(cf_data['pct_ge3'])

        fig, axes = plt.subplots(1, 3, figsize=(21, 5))
        plot_metric(axes[0], years_arr,
                    obs_data['mean_suit'], cf_data['mean_suit'],
                    mk_ms_obs, mk_ms_cf,
                    'Mean Suitability Score (all pixels)', 'Mean Suitability', 'class/yr')
        plot_metric(axes[1], years_arr,
                    obs_data['pct_ge1'], cf_data['pct_ge1'],
                    mk_g1_obs, mk_g1_cf,
                    '% Pixels with Class ≥ 1', '% Suitable (any)', '%/yr')
        plot_metric(axes[2], years_arr,
                    obs_data['pct_ge3'], cf_data['pct_ge3'],
                    mk_g3_obs, mk_g3_cf,
                    '% Pixels with Class ≥ 3', '% Moderately Suitable+', '%/yr')

        fig.suptitle(f'{label} — Suitability Metrics: Observed vs. No-Thaw (1979–2018)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        fig.savefig(f'{OUT_ROOT}/per_crop/{tag}_suit_trends.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

        # ── Class distribution stacked area chart ─────────────────────────────
        fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5))
        for ax, data, title in [
            (axes2[0], obs_data, 'Observed'),
            (axes2[1], cf_data,  'No-Thaw CF'),
        ]:
            bottom = np.zeros(len(years_arr))
            for c in range(6):
                vals = np.array(data[f'pct_{c}'])
                vals = np.where(np.isfinite(vals), vals, 0.0)
                ax.fill_between(years_arr, bottom, bottom + vals,
                                color=CLASS_COLORS[c], alpha=0.85,
                                label=CLASS_LABELS[c])
                bottom += vals
            ax.axvline(DIVERGENCE_YEAR, color='black', linestyle='--',
                       linewidth=1.2)
            ax.set_xlabel('Year', fontsize=10)
            ax.set_ylabel('% of Mask Pixels', fontsize=10)
            ax.set_title(f'{title}', fontsize=11, fontweight='bold')
            ax.set_ylim(0, 100)
            ax.set_xticks(years_arr[::4])
            ax.set_xticklabels(years_arr[::4], rotation=45, ha='right', fontsize=8)

        handles = [plt.Rectangle((0,0),1,1, color=CLASS_COLORS[c])
                   for c in range(6)]
        labels  = [CLASS_LABELS[c] for c in range(6)]
        fig2.legend(handles, labels, loc='lower center', ncol=6,
                    fontsize=8, bbox_to_anchor=(0.5, -0.05))
        fig2.suptitle(f'{label} — Suitability Class Distribution (1979–2018)',
                      fontsize=13, fontweight='bold')
        plt.tight_layout()
        fig2.savefig(f'{OUT_ROOT}/class_distribution/{tag}_class_dist.png',
                     dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  ✓ {label} saved')

    # ── Overall aggregate ──────────────────────────────────────────────────────
    print('\n  Overall aggregate …')

    for m in ['mean_suit', 'pct_ge1', 'pct_ge3']:
        obs_all[m] = np.nanmean(np.array(obs_all[m]), axis=0)
        cf_all[m]  = np.nanmean(np.array(cf_all[m]),  axis=0)
    for c in range(6):
        obs_cls_all[c] = np.nanmean(np.array(obs_cls_all[c]), axis=0)
        cf_cls_all[c]  = np.nanmean(np.array(cf_cls_all[c]),  axis=0)

    # MK for overall
    for period, idx in [('1979-2018', slice(None)), ('1999-2018', post_mask)]:
        for metric, unit in [('mean_suit','class/yr'),
                              ('pct_ge1','%/yr'), ('pct_ge3','%/yr')]:
            mk_obs = run_mk(obs_all[metric][idx])
            mk_cf  = run_mk(cf_all[metric][idx])

            sd_mean, sd_lo, sd_hi, sd_p = test_slope_difference(
                obs_all[metric][idx], cf_all[metric][idx])

            wil_med, wil_pct, wil_p_gt, wil_p_2s, wil_sig = \
                run_wilcoxon_pair(
                    obs_all[metric][post_mask], cf_all[metric][post_mask])

            if mk_obs and mk_cf:
                all_results.append({
                    'crop'              : 'OVERALL',
                    'period'            : period,
                    'metric'            : metric,
                    'obs_tau'           : mk_obs['tau'],
                    'obs_p'             : mk_obs['p'],
                    'obs_slope'         : mk_obs['slope'],
                    'obs_trend'         : mk_obs['trend'],
                    'obs_significant'   : mk_obs['significant'],
                    'cf_tau'            : mk_cf['tau'],
                    'cf_p'              : mk_cf['p'],
                    'cf_slope'          : mk_cf['slope'],
                    'cf_trend'          : mk_cf['trend'],
                    'cf_significant'    : mk_cf['significant'],
                    'slope_difference'  : round(mk_obs['slope'] - mk_cf['slope'], 6),
                    'slope_diff_ci_lo'  : sd_lo,
                    'slope_diff_ci_hi'  : sd_hi,
                    'slope_diff_p'      : sd_p,
                    'slope_diff_sig'    : (not np.isnan(sd_p)) and sd_p < ALPHA,
                    'wilcoxon_median_diff'  : wil_med,
                    'wilcoxon_pct_pos'      : wil_pct,
                    'wilcoxon_p_greater'    : wil_p_gt,
                    'wilcoxon_p_two_sided'  : wil_p_2s,
                    'wilcoxon_sig'          : wil_sig,
                })

    # Overall 3-metric plot
    mk_ms_obs = run_mk(obs_all['mean_suit'])
    mk_ms_cf  = run_mk(cf_all['mean_suit'])
    mk_g1_obs = run_mk(obs_all['pct_ge1'])
    mk_g1_cf  = run_mk(cf_all['pct_ge1'])
    mk_g3_obs = run_mk(obs_all['pct_ge3'])
    mk_g3_cf  = run_mk(cf_all['pct_ge3'])

    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    plot_metric(axes[0], years_arr,
                obs_all['mean_suit'], cf_all['mean_suit'],
                mk_ms_obs, mk_ms_cf,
                'Mean Suitability Score', 'Overall Mean Suitability', 'class/yr')
    plot_metric(axes[1], years_arr,
                obs_all['pct_ge1'], cf_all['pct_ge1'],
                mk_g1_obs, mk_g1_cf,
                '% Pixels Class ≥ 1', 'Overall % Suitable (any)', '%/yr')
    plot_metric(axes[2], years_arr,
                obs_all['pct_ge3'], cf_all['pct_ge3'],
                mk_g3_obs, mk_g3_cf,
                '% Pixels Class ≥ 3', 'Overall % Moderately Suitable+', '%/yr')

    fig.suptitle('Overall Aggregate — Suitability Metrics: Observed vs. No-Thaw (1979–2018)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{OUT_ROOT}/overall/overall_suit_trends.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    # Overall class distribution
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5))
    for ax, cls_data, title in [
        (axes2[0], obs_cls_all, 'Observed'),
        (axes2[1], cf_cls_all,  'No-Thaw CF'),
    ]:
        bottom = np.zeros(len(years_arr))
        for c in range(6):
            vals = np.where(np.isfinite(cls_data[c]), cls_data[c], 0.0)
            ax.fill_between(years_arr, bottom, bottom + vals,
                            color=CLASS_COLORS[c], alpha=0.85,
                            label=CLASS_LABELS[c])
            bottom += vals
        ax.axvline(DIVERGENCE_YEAR, color='black', linestyle='--', linewidth=1.2)
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel('% of Mask Pixels', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_xticks(years_arr[::4])
        ax.set_xticklabels(years_arr[::4], rotation=45, ha='right', fontsize=8)

    handles = [plt.Rectangle((0,0),1,1, color=CLASS_COLORS[c]) for c in range(6)]
    labels  = [CLASS_LABELS[c] for c in range(6)]
    fig2.legend(handles, labels, loc='lower center', ncol=6,
                fontsize=8, bbox_to_anchor=(0.5, -0.05))
    fig2.suptitle('Overall — Suitability Class Distribution (1979–2018)',
                  fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig2.savefig(f'{OUT_ROOT}/overall/overall_class_dist.png',
                 dpi=150, bbox_inches='tight')
    plt.close()

    # ── Save CSV ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    df.to_csv(f'{OUT_ROOT}/suitability_trend_results.csv', index=False)

    print(f'\n✓ All outputs saved to: {OUT_ROOT}/')
    print('\nOverall 40-year results:')
    df_o = df[(df['crop'] == 'OVERALL') & (df['period'] == '1979-2018')]
    print(df_o[['metric','obs_tau','obs_p','obs_slope','obs_significant',
                'cf_tau','cf_p','cf_slope','cf_significant',
                'slope_difference']].to_string(index=False))


if __name__ == '__main__':
    np.random.seed(42)
    mask = load_mask()
    run(mask)