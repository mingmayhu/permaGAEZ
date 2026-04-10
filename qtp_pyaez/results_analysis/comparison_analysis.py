"""
Three-Mask Comparison Wrapper
==============================
Runs four key analyses across three mask types:
  1. Overall (full study area mask)
  2. Permafrost only (value == 1 in permafrost_qilian.tif)
  3. Seasonally frozen only (value == 2 in permafrost_qilian.tif)

Analyses run for each mask:
  A. Wilcoxon on regional mean ΔSuitability (1999–2018)
  B. Cumulative ΔSuitability with permutation CI
  C. 40-year suitability trend (mean_suit, pct_ge1, pct_ge3)
  D. Elevation & slope stratification of ΔSuitability

All outputs saved to:
  ./results_analysis/outputs/mask_comparison/{mask_type}/

Final comparison summary table saved to:
  ./results_analysis/outputs/mask_comparison/comparison_summary.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon as scipy_wilcoxon
from pymannkendall import original_test as mk_test
from osgeo import gdal

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR      = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH     = r'./data_input/qilian mask.tif'
PERM_MAP_PATH = r'./data_input/permafrost_qilian.tif'
ELEV_PATH     = r'./data_input/terrain/elevation.npy'
SLOPE_PATH    = r'./data_input/terrain/slope.tif'
OUT_ROOT      = r'./results_analysis/outputs/mask_comparison'

YEARS_ALL       = list(range(1979, 2019))
YEARS_CF        = list(range(1999, 2019))
YEARS_PRE       = list(range(1979, 1999))
YEARS_POST      = list(range(1999, 2019))
DIVERGENCE_YEAR = 1999
N_PERM          = 1000
PIXEL_AREA_KM2  = 78.0
ALPHA           = 0.05
ELEV_BINS       = list(range(2000, 6000, 500))

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

PERM_DIR   = r'./data_input/permafrost_yearly'
CLIM_DIR   = r'./data_input/climate_yearly'
CLASS_DIR  = r'./results_analysis/outputs/6_spatial_analysis/2_sign_consistency'
YEARS_PRE  = list(range(1979, 1999))

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

def load_masks():
    """Load all three masks."""
    base = load_raster(MASK_PATH).astype(bool)
    perm = load_raster(PERM_MAP_PATH)
    perm = np.where(np.isfinite(perm), perm, 0)
    return {
        'overall'   : base,
        'permafrost': base & (perm == 1),
        'seasonal'  : base & (perm == 2),
    }

def obs_suit_path(tag, year):
    return f'./data_output/final_classification_fixed/{tag}/{year}_suitability_class.tif'

def cf_suit_path(tag, year):
    return f'./data_output/final_classification_nothaw_fixed/{tag}/{year}_suitability_class.tif'

def regional_mean_suit(arr, mask):
    arr = arr.copy()
    arr[arr < 0] = np.nan
    valid = mask & np.isfinite(arr)
    return float(np.nanmean(arr[valid])) if valid.any() else np.nan

def run_mk(series):
    s = np.array(series, dtype=float)
    valid = np.isfinite(s)
    if valid.sum() < 4:
        return None
    mk = mk_test(s[valid])
    line = np.full(len(s), np.nan)
    line[valid] = mk.intercept + mk.slope * np.arange(valid.sum())
    return {'tau': round(mk.Tau, 3), 'p': round(mk.p, 4),
            'slope': round(mk.slope, 6), 'significant': mk.p < ALPHA,
            'trend': mk.trend, 'sen_line': line}

def permutation_ci(diff_series, n_perm=N_PERM):
    vals  = np.where(np.isfinite(diff_series), diff_series, 0.0)
    perms = np.zeros((n_perm, len(vals)))
    for i in range(n_perm):
        signs    = np.random.choice([-1, 1], size=len(vals))
        perms[i] = np.cumsum(vals * signs)
    lo    = np.percentile(perms, 2.5,  axis=0)
    hi    = np.percentile(perms, 97.5, axis=0)
    final = float(np.cumsum(vals)[-1])
    p_val = float(np.mean(np.abs(perms[:, -1]) >= np.abs(final)))
    return lo, hi, p_val

def build_annual_delta_suit(tag, mask):
    """Annual regional mean ΔSuitability series (1999–2018)."""
    series = []
    for year in YEARS_CF:
        obs = load_raster(obs_suit_path(tag, year))
        cf  = load_raster(cf_suit_path(tag, year))
        if obs is None or cf is None:
            series.append(np.nan)
            continue
        obs[~mask] = np.nan; obs[obs < 0] = np.nan
        cf[~mask]  = np.nan; cf[cf < 0]   = np.nan
        delta = np.where(np.isfinite(obs) & np.isfinite(cf), obs - cf, np.nan)
        valid = mask & np.isfinite(delta)
        series.append(float(np.nanmean(delta[valid])) if valid.any() else np.nan)
    return np.array(series)

def build_suit_series(tag, mask):
    """Annual mean suitability for obs and CF across full 40 years."""
    obs_s, cf_s = [], []
    for year in YEARS_ALL:
        obs = load_raster(obs_suit_path(tag, year))
        obs_s.append(regional_mean_suit(obs, mask) if obs is not None else np.nan)
        if year < DIVERGENCE_YEAR:
            cf_s.append(obs_s[-1])
        else:
            cf = load_raster(cf_suit_path(tag, year))
            cf_s.append(regional_mean_suit(cf, mask) if cf is not None else np.nan)
    return np.array(obs_s), np.array(cf_s)


# ── Analysis A: Wilcoxon ──────────────────────────────────────────────────────

def analysis_wilcoxon(mask, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        series = build_annual_delta_suit(tag, mask)
        valid  = series[np.isfinite(series)]
        if len(valid) < 4:
            results.append({'crop': label, 'median_delta': np.nan,
                            'pct_years_positive': np.nan,
                            'p_greater_zero': np.nan, 'significant': False})
            continue
        nonzero = valid[valid != 0]
        if len(nonzero) < 4:
            results.append({'crop': label,
                            'median_delta': round(float(np.median(valid)), 5),
                            'pct_years_positive': round(float(np.mean(valid > 0)*100), 1),
                            'p_greater_zero': np.nan, 'significant': False})
            continue
        _, p_two = scipy_wilcoxon(valid, alternative='two-sided')
        _, p_pos = scipy_wilcoxon(valid, alternative='greater')
        results.append({
            'crop'               : label,
            'median_delta'       : round(float(np.median(valid)), 5),
            'pct_years_positive' : round(float(np.mean(valid > 0)*100), 1),
            'p_two_sided'        : round(p_two, 4),
            'p_greater_zero'     : round(p_pos, 4),
            'significant'        : p_pos < ALPHA,
        })
    df = pd.DataFrame(results)
    df.to_csv(f'{out_dir}/wilcoxon_results.csv', index=False)
    return df


# ── Analysis B: Cumulative ΔSuitability ──────────────────────────────────────

def analysis_cumulative(mask, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    years_arr = np.array(YEARS_CF)
    results   = []

    all_diff = []
    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        diff = build_annual_delta_suit(tag, mask)
        all_diff.append(diff)

        vals     = np.where(np.isfinite(diff), diff, 0.0)
        cum_diff = np.cumsum(vals)
        lo, hi, p_val = permutation_ci(diff)

        results.append({
            'crop'       : label,
            'cum_final'  : round(float(cum_diff[-1]), 4),
            'p_value'    : round(p_val, 4),
            'significant': p_val < ALPHA,
        })

        # Per-crop plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(years_arr, cum_diff, color='#2166AC', linewidth=2,
                marker='o', markersize=4)
        ax.fill_between(years_arr, lo, hi, alpha=0.2, color='grey',
                        label='95% permutation CI')
        ax.fill_between(years_arr, cum_diff, 0,
                        where=(cum_diff > hi), alpha=0.25,
                        color='#2166AC', label='Sig. positive')
        ax.fill_between(years_arr, cum_diff, 0,
                        where=(cum_diff < lo), alpha=0.25,
                        color='#D6604D', label='Sig. negative')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        sig_str = f'★ p={p_val:.3f}' if p_val < ALPHA else f'n.s. p={p_val:.3f}'
        ax.set_title(f'{label} — Cumulative ΔSuitability\n{sig_str}',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Year'); ax.set_ylabel('Cumulative ΔSuitability')
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(f'{out_dir}/{tag}_cumulative.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Overall aggregate
    agg_diff = np.nanmean(all_diff, axis=0)
    vals     = np.where(np.isfinite(agg_diff), agg_diff, 0.0)
    cum_agg  = np.cumsum(vals)
    lo, hi, p_val = permutation_ci(agg_diff)

    results.append({
        'crop': 'OVERALL', 'cum_final': round(float(cum_agg[-1]), 4),
        'p_value': round(p_val, 4), 'significant': p_val < ALPHA,
    })

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(years_arr, cum_agg, color='#2166AC', linewidth=2,
            marker='o', markersize=4, label='Cumulative Δ')
    ax.fill_between(years_arr, lo, hi, alpha=0.2, color='grey',
                    label='95% permutation CI')
    ax.fill_between(years_arr, cum_agg, 0,
                    where=(cum_agg > hi), alpha=0.25, color='#2166AC')
    ax.fill_between(years_arr, cum_agg, 0,
                    where=(cum_agg < lo), alpha=0.25, color='#D6604D')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    sig_str = f'★ p={p_val:.3f}' if p_val < ALPHA else f'n.s. p={p_val:.3f}'
    ax.set_title(f'Overall — Cumulative ΔSuitability\n{sig_str}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Year'); ax.set_ylabel('Cumulative ΔSuitability')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(f'{out_dir}/OVERALL_cumulative.png', dpi=150, bbox_inches='tight')
    plt.close()

    df = pd.DataFrame(results)
    df.to_csv(f'{out_dir}/cumulative_results.csv', index=False)
    return df


# ── Analysis C: 40-year suitability trend ────────────────────────────────────

def analysis_trend(mask, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    years_arr = np.array(YEARS_ALL)
    post_mask = years_arr >= DIVERGENCE_YEAR
    results   = []

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        obs_s, cf_s = build_suit_series(tag, mask)

        for period, idx in [('1979-2018', slice(None)),
                             ('1999-2018', post_mask)]:
            mk_obs = run_mk(obs_s[idx])
            mk_cf  = run_mk(cf_s[idx])
            if mk_obs and mk_cf:
                results.append({
                    'crop'            : label,
                    'period'          : period,
                    'obs_tau'         : mk_obs['tau'],
                    'obs_p'           : mk_obs['p'],
                    'obs_slope'       : mk_obs['slope'],
                    'obs_significant' : mk_obs['significant'],
                    'cf_tau'          : mk_cf['tau'],
                    'cf_p'            : mk_cf['p'],
                    'cf_slope'        : mk_cf['slope'],
                    'cf_significant'  : mk_cf['significant'],
                    'slope_difference': round(mk_obs['slope'] - mk_cf['slope'], 6),
                })

    df = pd.DataFrame(results)
    df.to_csv(f'{out_dir}/trend_results.csv', index=False)
    return df


# ── Analysis D: Elevation & slope stratification ──────────────────────────────

def analysis_elevation_slope(mask, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    elevation = np.load(ELEV_PATH)
    slope     = load_raster(SLOPE_PATH)
    slope[~mask] = np.nan

    elev_bins  = np.array(ELEV_BINS)
    elev_mids  = elev_bins[:-1] + np.diff(elev_bins) / 2

    # Compute mask-specific mean ΔSuitability per crop
    delta_arrays = {}
    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        stack = []
        for year in YEARS_CF:
            obs = load_raster(obs_suit_path(tag, year))
            cf  = load_raster(cf_suit_path(tag, year))
            if obs is None or cf is None:
                continue
            obs[~mask] = np.nan; obs[obs < 0] = np.nan
            cf[~mask]  = np.nan; cf[cf < 0]   = np.nan
            delta = np.where(np.isfinite(obs) & np.isfinite(cf),
                             obs - cf, np.nan)
            delta[~mask] = np.nan
            stack.append(delta)
        if stack:
            mean_delta = np.nanmean(stack, axis=0)
            mean_delta[~mask] = np.nan
            delta_arrays[label] = mean_delta

    if not delta_arrays:
        return pd.DataFrame()

    agg_delta = np.nanmean(list(delta_arrays.values()), axis=0)
    agg_delta[~mask] = np.nan

    records = []

    # Determine slope bins from data
    slope_masked = slope[mask & np.isfinite(slope)]
    if len(slope_masked) > 0:
        s_max      = np.percentile(slope_masked, 99)
        slope_bins = np.linspace(0, s_max, 7)
        slope_bins = np.append(slope_bins, slope_masked.max() + 1)
        slope_mids = slope_bins[:-1] + np.diff(slope_bins) / 2
    else:
        slope_bins = None

    for var_name, var_arr, bins, mids in [
        ('elevation', elevation, elev_bins, elev_mids),
        *([('slope', slope, slope_bins, slope_mids)]
          if slope_bins is not None else []),
    ]:
        for label, delta in {**delta_arrays, 'OVERALL': agg_delta}.items():
            for b_lo, b_hi in zip(bins[:-1], bins[1:]):
                in_bin = mask & (var_arr >= b_lo) & \
                         (var_arr < b_hi) & np.isfinite(delta)
                val  = float(np.nanmean(delta[in_bin])) \
                       if in_bin.any() else np.nan
                n_px = int(in_bin.sum())
                records.append({
                    'crop'      : label,
                    'variable'  : var_name,
                    'bin_lo'    : round(b_lo, 2),
                    'bin_hi'    : round(b_hi, 2),
                    'bin_mid'   : round((b_lo + b_hi) / 2, 2),
                    'mean_delta': round(val, 5) if not np.isnan(val) else np.nan,
                    'n_pixels'  : n_px,
                })

    # Plot overall elevation profile
    df = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, var_name, ylabel in [
        (axes[0], 'elevation', 'Elevation (m)'),
        (axes[1], 'slope',     'Slope (°)'),
    ]:
        df_v = df[(df['variable'] == var_name) & (df['crop'] == 'OVERALL')]
        if df_v.empty:
            ax.axis('off')
            continue
        valid = df_v[df_v['mean_delta'].notna()]
        colors = ['#2166AC' if v >= 0 else '#D6604D'
                  for v in valid['mean_delta']]
        ax.barh(range(len(valid)), valid['mean_delta'],
                color=colors, edgecolor='white', alpha=0.85)
        ax.set_yticks(range(len(valid)))
        ax.set_yticklabels(
            [f'{r.bin_lo:.0f}–{r.bin_hi:.0f}'
             for _, r in valid.iterrows()], fontsize=8)
        for i, (_, row) in enumerate(valid.iterrows()):
            ax.text(row['mean_delta'] + 0.00005 * np.sign(row['mean_delta']),
                    i, f'n={row["n_pixels"]}', va='center', fontsize=7)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Mean ΔSuitability', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f'Overall ΔSuitability by {var_name.title()} Band',
                     fontsize=11, fontweight='bold')

    plt.tight_layout()
    fig.savefig(f'{out_dir}/elevation_slope_overall.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    df.to_csv(f'{out_dir}/elevation_slope_results.csv', index=False)
    return df


# ── Analysis E: Box plots of pixel characteristics ───────────────────────────

def load_env_vars(mask):
    """Load all environmental variables."""
    from scipy.stats import kruskal as scipy_kruskal
    elev  = np.load(ELEV_PATH)
    slope = load_raster(SLOPE_PATH)
    slope[~mask] = np.nan

    def perm_mean(f, yrs, agg='max'):
        stack = []
        for y in yrs:
            p = f'{PERM_DIR}/{y}/{f}'
            if not os.path.exists(p):
                continue
            a = np.load(p).astype(float)
            if a.ndim == 3:
                a = np.nanmax(a,axis=2) if agg=='max' else np.nanmean(a,axis=2)
            a[~mask] = np.nan
            stack.append(a)
        return np.nanmean(stack,axis=0) if stack else np.full(mask.shape,np.nan)

    def clim_mean(f, yrs, agg='mean'):
        stack = []
        for y in yrs:
            p = f'{CLIM_DIR}/{y}/{f}'
            if not os.path.exists(p):
                continue
            a = np.load(p).astype(float)
            if a.ndim == 3:
                a = np.nansum(a,axis=2) if agg=='sum' else np.nanmean(a,axis=2)
            a[~mask] = np.nan
            stack.append(a)
        return np.nanmean(stack,axis=0) if stack else np.full(mask.shape,np.nan)

    alt_post = perm_mean('active_layer_depth.npy', YEARS_POST, 'max')
    alt_pre  = perm_mean('active_layer_depth.npy', YEARS_PRE,  'max')
    sm_post  = perm_mean('avail_soil_moisture.npy', YEARS_POST, 'mean')
    sm_pre   = perm_mean('avail_soil_moisture.npy', YEARS_PRE,  'mean')
    tm_post  = clim_mean('TempMax.npy', YEARS_POST, 'mean')
    tm_pre   = clim_mean('TempMax.npy', YEARS_PRE,  'mean')
    tn_post  = clim_mean('TempMin.npy', YEARS_POST, 'mean')
    tn_pre   = clim_mean('TempMin.npy', YEARS_PRE,  'mean')
    pr_post  = clim_mean('Precip.npy',  YEARS_POST, 'sum')
    pr_pre   = clim_mean('Precip.npy',  YEARS_PRE,  'sum')

    return {
        'Elevation (m)'     : elev,
        'Slope (°)'         : slope,
        'Mean ALT (m)'      : alt_post,
        'ΔALT (m)'          : alt_post - alt_pre,
        'Mean Soil Moist.'  : sm_post,
        'ΔSoil Moist.'      : sm_post - sm_pre,
        'Mean TempMax (°C)' : tm_post,
        'ΔTempMax (°C)'     : tm_post - tm_pre,
        'Mean TempMin (°C)' : tn_post,
        'ΔTempMin (°C)'     : tn_post - tn_pre,
        'Mean Precip (mm)'  : pr_post,
        'ΔPrecip (mm)'      : pr_post - pr_pre,
    }

def analysis_boxplots(mask, out_dir):
    from scipy.stats import kruskal as scipy_kruskal
    from matplotlib.patches import Patch
    os.makedirs(out_dir, exist_ok=True)

    env_vars = load_env_vars(mask)

    # Recompute net score using mask-specific ΔSuitability
    print('  Computing pixel classifications for this mask …')
    sig_pos_stack, sig_neg_stack = [], []

    for crop in CROPS:
        tag = crop['tag']
        delta_stack = []
        for year in YEARS_CF:
            obs = load_raster(obs_suit_path(tag, year))
            cf  = load_raster(cf_suit_path(tag, year))
            if obs is None or cf is None:
                continue
            obs[~mask] = np.nan; obs[obs < 0] = np.nan
            cf[~mask]  = np.nan; cf[cf < 0]   = np.nan
            delta = np.where(np.isfinite(obs) & np.isfinite(cf),
                             obs - cf, np.nan)
            delta[~mask] = np.nan
            delta_stack.append(delta)

        if not delta_stack:
            continue

        delta_stack = np.array(delta_stack)   # (n_years, rows, cols)
        rows, cols  = mask.shape

        # Pixel-wise sign consistency within this mask
        sig_pos = np.zeros(mask.shape, dtype=float)
        sig_neg = np.zeros(mask.shape, dtype=float)

        for r in range(rows):
            for c in range(cols):
                if not mask[r, c]:
                    continue
                series  = delta_stack[:, r, c]
                valid   = series[np.isfinite(series)]
                nonzero = valid[valid != 0]
                if len(nonzero) < 3:
                    continue
                pct_pos = float(np.mean(nonzero > 0))
                med     = float(np.median(nonzero))
                if len(nonzero) >= 4:
                    try:
                        from scipy.stats import wilcoxon as _wil
                        _, p = _wil(valid, alternative='two-sided')
                        if p < ALPHA and med > 0:
                            sig_pos[r, c] = 1
                        elif p < ALPHA and med < 0:
                            sig_neg[r, c] = 1
                    except Exception:
                        pass

        sig_pos_stack.append(sig_pos)
        sig_neg_stack.append(sig_neg)

    if not sig_pos_stack:
        print('  ⚠ No data for box plots — skipping.')
        return

    net = np.nansum(sig_pos_stack, axis=0) - np.nansum(sig_neg_stack, axis=0)

    cat_codes  = [4, 2, 0]
    cat_labels = {4: 'Net+ (≥1)', 2: 'Neutral', 0: 'Net− (≤-1)'}
    cat_colors = {4: '#1a6faf', 2: '#f0f0f0', 0: '#c0392b'}

    # Use threshold of 1 instead of 2 since permafrost mask has fewer pixels
    cls_map = np.full(mask.shape, np.nan)
    cls_map[mask & (net >= 1)]  = 4
    cls_map[mask & (net <= -1)] = 0
    cls_map[mask & (net > -1) & (net < 1)] = 2

    counts = {c: int(np.sum(mask & (cls_map == c))) for c in cat_codes}
    print(f'  Box plot pixel counts: ' +
          ' | '.join([f'{cat_labels[c]}: {counts[c]}' for c in cat_codes]))

    # Check we have enough pixels to plot
    if all(counts[c] == 0 for c in [4, 0]):
        print('  ⚠ No significant pixels found — skipping box plots.')
        return

    n_vars = len(env_vars)
    ncols  = 4
    nrows  = -(-n_vars // ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 4))
    axes = axes.flatten()
    kw_results = []

    for i, (var_name, var_arr) in enumerate(env_vars.items()):
        ax = axes[i]
        groups, plot_data, plot_labels, plot_colors = {}, [], [], []

        for code in cat_codes:
            if counts[code] == 0:
                continue
            vals = var_arr[mask & (cls_map == code)]
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            groups[code] = vals
            plot_data.append(vals)
            plot_labels.append(f'{cat_labels[code]}\n(n={len(vals)})')
            plot_colors.append(cat_colors[code])

        if len(plot_data) < 2:
            ax.axis('off')
            continue

        bp = ax.boxplot(plot_data, patch_artist=True,
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(linewidth=1.2),
                        flierprops=dict(marker='o', markersize=2,
                                        alpha=0.4, linestyle='none'))
        for patch, color in zip(bp['boxes'], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

        ax.set_xticklabels(plot_labels, fontsize=8)
        ax.set_title(var_name, fontsize=10, fontweight='bold')

        clean = [g[np.isfinite(g)] for g in groups.values()
                 if len(g[np.isfinite(g)]) > 1]
        try:
            H, p = scipy_kruskal(*clean) if len(clean) >= 2 \
                   else (np.nan, np.nan)
        except Exception:
            H, p = np.nan, np.nan

        sig = '★' if (not np.isnan(p) and p < ALPHA) else ''
        ax.text(0.98, 0.97, f'KW p={p:.3f}{sig}',
                transform=ax.transAxes, fontsize=7.5, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        kw_results.append({
            'variable'   : var_name,
            'kruskal_H'  : round(H, 3) if not np.isnan(H) else np.nan,
            'kruskal_p'  : round(p, 4) if not np.isnan(p) else np.nan,
            'kruskal_sig': (not np.isnan(p)) and p < ALPHA,
        })

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    legend_elements = [Patch(facecolor=cat_colors[c], label=cat_labels[c])
                       for c in cat_codes if counts[c] > 0]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Environmental Characteristics by Net Thaw Score\n'
                 '★ = Kruskal-Wallis p < 0.05',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(f'{out_dir}/boxplots.png', dpi=150, bbox_inches='tight')
    plt.close()

    pd.DataFrame(kw_results).to_csv(f'{out_dir}/kruskal.csv', index=False)
    print(f'  ✓ Box plots saved')

def run():
    np.random.seed(42)
    masks = load_masks()

    mask_info = {
        'overall'   : f'Overall (n={masks["overall"].sum()})',
        'permafrost': f'Permafrost only (n={masks["permafrost"].sum()})',
        'seasonal'  : f'Seasonally frozen only (n={masks["seasonal"].sum()})',
    }

    print('Mask sizes:')
    for k, v in mask_info.items():
        print(f'  {v}')

    # Store results for comparison summary
    wilcoxon_all = {}
    cumulative_all = {}
    trend_all = {}

    for mask_type, mask in masks.items():
        print(f'\n{"="*50}')
        print(f'Running analyses for: {mask_info[mask_type]}')
        print(f'{"="*50}')

        out_dir = f'{OUT_ROOT}/{mask_type}'

        print('  [A] Wilcoxon …')
        wilcoxon_all[mask_type] = analysis_wilcoxon(
            mask, f'{out_dir}/wilcoxon')

        print('  [B] Cumulative ΔSuitability …')
        cumulative_all[mask_type] = analysis_cumulative(
            mask, f'{out_dir}/cumulative')

        print('  [C] 40-year trend …')
        trend_all[mask_type] = analysis_trend(
            mask, f'{out_dir}/trend')

        print('  [D] Elevation stratification …')
        analysis_elevation_slope(mask, f'{out_dir}/elevation')

        print(f'  ✓ {mask_type} complete')

    # ── Comparison summary table ───────────────────────────────────────────────
    print('\nBuilding comparison summary …')
    summary_rows = []

    for crop_info in CROPS + [{'label': 'OVERALL', 'tag': None}]:
        label = crop_info['label']
        row   = {'crop': label}

        for mask_type in ['overall', 'permafrost', 'seasonal']:
            # Wilcoxon
            wdf = wilcoxon_all[mask_type]
            w   = wdf[wdf['crop'] == label]
            if not w.empty:
                row[f'{mask_type}_wil_p']   = w['p_greater_zero'].values[0]
                row[f'{mask_type}_wil_sig'] = w['significant'].values[0]
            else:
                row[f'{mask_type}_wil_p']   = np.nan
                row[f'{mask_type}_wil_sig'] = False

            # Cumulative
            cdf = cumulative_all[mask_type]
            c   = cdf[cdf['crop'] == label]
            if not c.empty:
                row[f'{mask_type}_cum_final'] = c['cum_final'].values[0]
                row[f'{mask_type}_cum_p']     = c['p_value'].values[0]
                row[f'{mask_type}_cum_sig']   = c['significant'].values[0]

            # Trend slope difference (40yr)
            tdf = trend_all[mask_type]
            t   = tdf[(tdf['crop'] == label) & (tdf['period'] == '1979-2018')]
            if not t.empty:
                row[f'{mask_type}_slope_diff'] = t['slope_difference'].values[0]
                row[f'{mask_type}_obs_sig']    = t['obs_significant'].values[0]

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f'{OUT_ROOT}/comparison_summary.csv', index=False)

    # ── Comparison figure: Wilcoxon p-values across three masks ───────────────
    crops_list = [c['label'] for c in CROPS]
    x          = np.arange(len(crops_list))
    width      = 0.25
    colors     = {'overall': '#2166AC', 'permafrost': '#1a9850',
                  'seasonal': '#D6604D'}

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, mask_type in enumerate(['overall', 'permafrost', 'seasonal']):
        wdf  = wilcoxon_all[mask_type]
        wdf  = wdf[wdf['crop'].isin(crops_list)].set_index('crop')
        vals = [wdf.loc[c, 'p_greater_zero']
                if c in wdf.index else np.nan for c in crops_list]
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      label=mask_info[mask_type],
                      color=colors[mask_type], alpha=0.85)
        # Mark significant bars
        for bar, v in zip(bars, vals):
            if not np.isnan(v) and v < ALPHA:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.005,
                        '★', ha='center', fontsize=10)

    ax.axhline(ALPHA, color='black', linewidth=0.8, linestyle='--',
               label=f'p = {ALPHA}')
    ax.set_xticks(x)
    ax.set_xticklabels(crops_list, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Wilcoxon p-value (one-sided)', fontsize=11)
    ax.set_title('Wilcoxon Test: Is Observed Suitability Consistently > Counterfactual?\n'
                 'Comparison across three mask types  ★ = p < 0.05',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(f'{OUT_ROOT}/wilcoxon_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── Comparison figure: cumulative final gap ────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, mask_type in enumerate(['overall', 'permafrost', 'seasonal']):
        cdf  = cumulative_all[mask_type]
        cdf  = cdf[cdf['crop'].isin(crops_list)].set_index('crop')
        vals = [cdf.loc[c, 'cum_final']
                if c in cdf.index else np.nan for c in crops_list]
        sigs = [cdf.loc[c, 'significant']
                if c in cdf.index else False for c in crops_list]
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      label=mask_info[mask_type],
                      color=colors[mask_type], alpha=0.85)
        for bar, sig in zip(bars, sigs):
            if sig:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.0005 * np.sign(bar.get_height()),
                        '★', ha='center', fontsize=10)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(crops_list, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Cumulative ΔSuitability (1999–2018)', fontsize=11)
    ax.set_title('Cumulative Thaw Contribution to Suitability\n'
                 'Comparison across three mask types  ★ = p < 0.05 (permutation)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(f'{OUT_ROOT}/cumulative_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f'\n✓ All outputs saved to: {OUT_ROOT}/')
    print('\nComparison summary (Wilcoxon significance):')
    cols = ['crop'] + [f'{m}_wil_sig' for m in ['overall','permafrost','seasonal']]
    print(summary_df[cols].to_string(index=False))


if __name__ == '__main__':
    run()