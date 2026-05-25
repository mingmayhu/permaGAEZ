"""
FAO Comparison Analysis
=======================
Compares the permafrost-considered observed scenario with the original
FAO GAEZ methodology (no permafrost inputs).

Class 0 and class 1 are combined into class 1 (remap) before all calculations.
Lake pixels excluded via permafrost_qilian.tif.

Analyses:
  1. Difference maps — observed minus FAO original per crop and overall
  2. Regional mean comparison — time series of mean suitability and
     suitable land area (km²) for both scenarios (1979-2018)
  3. Suitability class transition matrix — how pixels shift between
     the two methodologies

Outputs written to:
  ./results/permafrost_thaw_impact/fao_comparison/outputs/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, BoundaryNorm, ListedColormap
import matplotlib.patches as mpatches
from pymannkendall import original_test as mk_test
from osgeo import gdal

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR        = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH       = r'./data_input/qilian mask.tif'
PERMAFROST_PATH = r'./data_input/permafrost_qilian.tif'
OUT_ROOT        = r'./results/permafrost_thaw_impact/permafrost_vs_fao/outputs'

YEARS_ALL  = list(range(1979, 2019))
YEARS_PRE  = list(range(1979, 1999))
YEARS_POST = list(range(1999, 2019))

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

CLASS_COLORS = {
    1: '#fc8d59', 2: '#fee08b', 3: '#d9ef8b', 4: '#91cf60', 5: '#1a9850',
}

os.chdir(WORK_DIR)
os.makedirs(OUT_ROOT, exist_ok=True)
for sub in ['1_diff_maps', '2_time_series', '3_transition']:
    os.makedirs(f'{OUT_ROOT}/{sub}', exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def remap(arr_int):
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

def load_mask():
    arr, _ = load_raster(MASK_PATH)
    mask = arr.astype(bool)
    pf_arr, _ = load_raster(PERMAFROST_PATH)
    if pf_arr is not None:
        lake_mask = ((pf_arr == 0) | ~np.isfinite(pf_arr)) & mask
        mask[lake_mask] = False
        print(f'  Excluded {lake_mask.sum()} lake/nodata pixels from mask')
    return mask

def build_pixel_area_km2(mask):
    ds  = gdal.Open(MASK_PATH)
    gt  = ds.GetGeoTransform()
    nrows, ncols = mask.shape
    lats = gt[3] + gt[5] * (np.arange(nrows) + 0.5)
    pixel_side_km = abs(gt[5]) * 111.32
    area_2d = np.outer(
        pixel_side_km * pixel_side_km * np.cos(np.deg2rad(lats)),
        np.ones(ncols)
    )
    area_2d[~mask] = 0.0
    return area_2d

def obs_path(tag, year):
    return f'./data_output/final_classification_fixed/{tag}/{year}_suitability_class.tif'

def fao_path(tag, year):
    return f'./data_output/original/final_classification_fixed/{tag}/{year}_suitability_class.tif'

def apply_remap(arr, mask):
    arr_c = arr.copy()
    arr_c[arr_c < 0] = np.nan
    arr_c[~mask] = np.nan
    return np.where(
        np.isfinite(arr_c),
        remap(np.where(np.isfinite(arr_c), arr_c, 0).astype(int)).astype(float),
        np.nan
    )

def regional_mean_suit(arr, mask):
    arr_r = apply_remap(arr, mask)
    valid = mask & np.isfinite(arr_r)
    return float(np.nanmean(arr_r[valid])) if valid.any() else np.nan

def regional_area_ge2_km2(arr, mask, pixel_area_km2):
    """Sum pixel areas where suitability class >= 2."""
    arr_c = arr.copy()
    arr_c[arr_c < 0] = np.nan
    arr_int = np.where(np.isfinite(arr_c),
                       np.clip(np.where(np.isfinite(arr_c), arr_c, 0).astype(int), 0, 5), 0)
    arr_int[arr_int == 0] = 1  # remap class 0 -> 1
    suitable = mask & (arr_int >= 2)
    return float(np.sum(pixel_area_km2[suitable]))

def run_mk(series):
    s = np.array(series, dtype=float)
    valid = np.isfinite(s)
    if valid.sum() < 4:
        return None
    mk = mk_test(s[valid])
    line = np.full(len(s), np.nan)
    line[valid] = mk.intercept + mk.slope * np.arange(valid.sum())
    return {'tau': round(mk.Tau, 3), 'p': round(mk.p, 4),
            'slope': round(mk.slope, 6), 'intercept': mk.intercept,
            'significant': mk.p < 0.05, 'sen_line': line}

def bootstrap_sen_ci(series, n_boot=1000, ci=95):
    s         = np.array(series, dtype=float)
    valid_idx = np.where(np.isfinite(s))[0]
    if len(valid_idx) < 4:
        return (np.nan, np.nan)
    s_valid = s[valid_idx]
    slopes  = []
    rng     = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = np.sort(rng.choice(len(s_valid), size=len(s_valid), replace=True))
        slopes.append(mk_test(s_valid[idx]).slope)
    lo = np.percentile(slopes, (100 - ci) / 2)
    hi = np.percentile(slopes, 100 - (100 - ci) / 2)
    return (lo, hi)

def plot_map(arr, mask, title, cmap, vmin, vmax, out_path,
             cbar_label='', vcenter=None):
    plot_arr = arr.copy()
    plot_arr[~mask] = np.nan
    fig, ax = plt.subplots(figsize=(10, 6))
    if vcenter is not None:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        im = ax.imshow(plot_arr, cmap=cmap, norm=norm)
    else:
        im = ax.imshow(plot_arr, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, shrink=0.7, label=cbar_label)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_panel(arrays, labels, mask, suptitle, cmap, vmin, vmax,
               out_path, cbar_label='', vcenter=None, ncols=4):
    nrows = -(-len(arrays) // ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.5, nrows * 3.2))
    axes = axes.flatten()
    for i, (arr, label) in enumerate(zip(arrays, labels)):
        plot_arr = arr.copy()
        plot_arr[~mask] = np.nan
        if vcenter is not None:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
            im = axes[i].imshow(plot_arr, cmap=cmap, norm=norm)
        else:
            im = axes[i].imshow(plot_arr, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=axes[i], shrink=0.7, label=cbar_label)
        axes[i].set_title(label, fontsize=10, fontweight='bold')
        axes[i].axis('off')
    for j in range(len(arrays), len(axes)):
        axes[j].axis('off')
    fig.suptitle(suptitle, fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


# ── Analysis 1: Difference maps ───────────────────────────────────────────────

def analysis_diff_maps(mask):
    print('\n[Analysis 1] Difference maps (Observed minus FAO Original) ...')
    out_dir  = f'{OUT_ROOT}/1_diff_maps'
    geo_info = None

    mean_diffs = []
    labels     = []

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        print(f'  {label}')

        obs_stack, fao_stack = [], []
        for year in YEARS_POST:
            obs, gi = load_raster(obs_path(tag, year))
            fao, _  = load_raster(fao_path(tag, year))
            if obs is None or fao is None:
                continue
            if geo_info is None:
                geo_info = gi
            obs_stack.append(apply_remap(obs, mask))
            fao_stack.append(apply_remap(fao, mask))

        if not obs_stack:
            print(f'    Warning: no data for {label}')
            continue

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            obs_mean = np.nanmean(np.stack(obs_stack), axis=0)
            fao_mean = np.nanmean(np.stack(fao_stack), axis=0)

        diff = np.where(np.isfinite(obs_mean) & np.isfinite(fao_mean),
                        obs_mean - fao_mean, np.nan)
        diff[~mask] = np.nan

        mean_diffs.append(diff)
        labels.append(label)

        save_raster(f'{out_dir}/{tag}_diff.tif',
                    np.where(np.isfinite(diff), diff, -9999.0), geo_info)

        pos_pct = float(np.nanmean(diff[mask & np.isfinite(diff)] > 0) * 100)
        neg_pct = float(np.nanmean(diff[mask & np.isfinite(diff)] < 0) * 100)
        print(f'    {pos_pct:.1f}% pixels obs > FAO, '
              f'{neg_pct:.1f}% pixels obs < FAO')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        overall_diff = np.nanmean(np.stack(mean_diffs), axis=0)
    overall_diff[~mask] = np.nan
    mean_diffs.append(overall_diff)
    labels.append('OVERALL')
    save_raster(f'{out_dir}/overall_diff.tif',
                np.where(np.isfinite(overall_diff), overall_diff, -9999.0),
                geo_info)

    all_vals = np.concatenate([d[mask & np.isfinite(d)] for d in mean_diffs])
    vlim = max(float(np.nanpercentile(np.abs(all_vals), 98)), 1e-4)

    plot_panel(
        mean_diffs, labels, mask,
        suptitle='Mean Suitability Difference (Observed minus FAO Original)\n'
                 '1999-2018 | Blue = obs higher, Red = FAO higher',
        cmap='RdBu', vmin=-vlim, vmax=vlim, vcenter=0,
        cbar_label='delta Class',
        out_path=f'{out_dir}/diff_panel.png', ncols=4
    )

    plot_map(
        overall_diff, mask,
        title='Overall Mean Suitability Difference\n(Observed minus FAO Original, 1999-2018)',
        cmap='RdBu', vmin=-vlim, vmax=vlim, vcenter=0,
        cbar_label='Mean delta Class',
        out_path=f'{out_dir}/overall_diff_map.png'
    )

    print(f'  Overall diff range: '
          f'[{np.nanmin(overall_diff):.4f}, {np.nanmax(overall_diff):.4f}]')
    print(f'  Overall: {float(np.nanmean(overall_diff[mask & np.isfinite(overall_diff)] > 0)*100):.1f}% '
          f'pixels obs > FAO')
    return mean_diffs, labels, geo_info


# ── Analysis 2: Regional mean time series comparison ──────────────────────────

def analysis_time_series(mask, pixel_area_km2):
    print('\n[Analysis 2] Regional mean time series comparison ...')
    out_dir   = f'{OUT_ROOT}/2_time_series'
    years_arr = np.array(YEARS_ALL)
    mk_results = []

    obs_mean_all, fao_mean_all = [], []
    obs_area_all, fao_area_all = [], []

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        print(f'  {label}')

        obs_mean_s, fao_mean_s = [], []
        obs_area_s, fao_area_s = [], []

        for year in YEARS_ALL:
            obs, _ = load_raster(obs_path(tag, year))
            fao, _ = load_raster(fao_path(tag, year))
            obs_mean_s.append(regional_mean_suit(obs, mask) if obs is not None else np.nan)
            fao_mean_s.append(regional_mean_suit(fao, mask) if fao is not None else np.nan)
            obs_area_s.append(regional_area_ge2_km2(obs, mask, pixel_area_km2) if obs is not None else np.nan)
            fao_area_s.append(regional_area_ge2_km2(fao, mask, pixel_area_km2) if fao is not None else np.nan)

        obs_mean_s = np.array(obs_mean_s)
        fao_mean_s = np.array(fao_mean_s)
        obs_area_s = np.array(obs_area_s)
        fao_area_s = np.array(fao_area_s)

        obs_mean_all.append(obs_mean_s)
        fao_mean_all.append(fao_mean_s)
        obs_area_all.append(obs_area_s)
        fao_area_all.append(fao_area_s)

        # Per-crop MK trends
        for scenario, ms, area in [('obs', obs_mean_s, obs_area_s),
                                    ('fao', fao_mean_s, fao_area_s)]:
            for metric, series in [('mean_suit', ms), ('area_ge2_km2', area)]:
                mk = run_mk(series)
                if mk:
                    mk_results.append({
                        'crop': label, 'scenario': scenario,
                        'metric': metric, 'tau': mk['tau'],
                        'p': mk['p'], 'slope': mk['slope'],
                        'significant': mk['significant'],
                        'ci_lo': np.nan, 'ci_hi': np.nan,
                    })

        # Per-crop 2-panel time series
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        for ax, obs_s, fao_s, ylabel, title in [
            (axes[0], obs_mean_s, fao_mean_s,
             'Mean Suitability Score (1-5)', 'Mean Suitability'),
            (axes[1], obs_area_s, fao_area_s,
             'Suitable Land Area (km²)',     'Suitable Land Area (class ≥ 2)'),
        ]:
            ax.plot(years_arr, obs_s, color='#2166AC', linewidth=2,
                    marker='o', markersize=3, label='Observed (permafrost)')
            ax.plot(years_arr, fao_s, color='#D6604D', linewidth=2,
                    marker='s', markersize=3, linestyle='--', label='FAO Original')
            diff_s = obs_s - fao_s
            ax2 = ax.twinx()
            ax2.bar(years_arr, diff_s,
                    color=np.where(diff_s >= 0, '#92C5DE', '#F4A582'),
                    alpha=0.4, width=0.8)
            ax2.axhline(0, color='grey', linewidth=0.5)
            ax2.set_ylabel('Difference (Obs minus FAO)', fontsize=9, color='grey')
            ax2.tick_params(axis='y', labelsize=8, colors='grey')
            ax.set_xlabel('Year', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(f'{label} — {title}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='upper left')
            ax.set_xticks(years_arr[::4])
            ax.set_xticklabels(years_arr[::4], rotation=45, ha='right', fontsize=8)
        plt.tight_layout()
        fig.savefig(f'{out_dir}/{tag}_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Overall aggregate
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        obs_mean_agg = np.nanmean(obs_mean_all, axis=0)
        fao_mean_agg = np.nanmean(fao_mean_all, axis=0)
        obs_area_agg = np.nanmean(obs_area_all, axis=0)
        fao_area_agg = np.nanmean(fao_area_all, axis=0)

    # Overall MK + bootstrap CI
    print('  Bootstrapping overall Sen slope CIs ...')
    mk_obs_ms = run_mk(obs_mean_agg)
    mk_fao_ms = run_mk(fao_mean_agg)
    mk_obs_ar = run_mk(obs_area_agg)
    mk_fao_ar = run_mk(fao_area_agg)

    ci_obs_ms = bootstrap_sen_ci(obs_mean_agg)
    ci_fao_ms = bootstrap_sen_ci(fao_mean_agg)
    ci_obs_ar = bootstrap_sen_ci(obs_area_agg)
    ci_fao_ar = bootstrap_sen_ci(fao_area_agg)

    for scenario, mk_ms, mk_ar, ci_ms, ci_ar in [
        ('obs', mk_obs_ms, mk_obs_ar, ci_obs_ms, ci_obs_ar),
        ('fao', mk_fao_ms, mk_fao_ar, ci_fao_ms, ci_fao_ar),
    ]:
        for metric, mk, ci in [('mean_suit', mk_ms, ci_ms),
                                ('area_ge2_km2', mk_ar, ci_ar)]:
            if mk:
                mk_results.append({
                    'crop': 'OVERALL', 'scenario': scenario,
                    'metric': metric, 'tau': mk['tau'],
                    'p': mk['p'], 'slope': mk['slope'],
                    'significant': mk['significant'],
                    'ci_lo': ci[0], 'ci_hi': ci[1],
                })

    # Overall 2-panel figure with CI bands
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, obs_s, fao_s, mk_obs, mk_fao, ci_obs, ci_fao, ylabel, title in [
        (axes[0], obs_mean_agg, fao_mean_agg, mk_obs_ms, mk_fao_ms,
         ci_obs_ms, ci_fao_ms,
         'Mean Suitability Score (1-5)', 'Overall Mean Suitability'),
        (axes[1], obs_area_agg, fao_area_agg, mk_obs_ar, mk_fao_ar,
         ci_obs_ar, ci_fao_ar,
         'Suitable Land Area (km²)', 'Overall Suitable Land Area (class ≥ 2)'),
    ]:
        ax.plot(years_arr, obs_s, color='#2166AC', linewidth=2.5,
                marker='o', markersize=4, label='Observed (permafrost)')
        ax.plot(years_arr, fao_s, color='#D6604D', linewidth=2.5,
                marker='s', markersize=4, linestyle='--', label='FAO Original')

        for mk, ci, color in [(mk_obs, ci_obs, '#2166AC'),
                              (mk_fao, ci_fao, '#D6604D')]:
            if mk:
                sig = '*' if mk['significant'] else ''
                ax.plot(years_arr, mk['sen_line'], color=color,
                        linewidth=1.5, linestyle=':',
                        label=f"{'Obs' if color == '#2166AC' else 'FAO'} slope: "
                              f"{mk['slope']:.5f}/yr (p={mk['p']:.3f}){sig}")
                if not np.isnan(ci[0]):
                    valid = np.isfinite(obs_s if color == '#2166AC' else fao_s)
                    x_idx = np.arange(valid.sum())
                    lo_line = np.full(len(years_arr), np.nan)
                    hi_line = np.full(len(years_arr), np.nan)
                    lo_line[valid] = mk['intercept'] + ci[0] * x_idx
                    hi_line[valid] = mk['intercept'] + ci[1] * x_idx
                    ax.fill_between(years_arr, lo_line, hi_line,
                                    color=color, alpha=0.10)

        ax.fill_between(years_arr, obs_s, fao_s,
                        where=obs_s >= fao_s,
                        alpha=0.15, color='#2166AC', label='Obs > FAO')
        ax.fill_between(years_arr, obs_s, fao_s,
                        where=obs_s < fao_s,
                        alpha=0.15, color='#D6604D', label='FAO > Obs')

        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        ax.set_xticks(years_arr[::4])
        ax.set_xticklabels(years_arr[::4], rotation=45, ha='right', fontsize=8)

    fig.suptitle(
        'Observed (Permafrost-Considered) vs FAO Original Methodology\n'
        'Overall Mean across all 10 crops (1979-2018)',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    fig.savefig(f'{out_dir}/overall_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save MK results with CI columns
    pd.DataFrame(mk_results).to_csv(f'{out_dir}/mk_comparison_results.csv', index=False)

    print(f'  Obs mean_suit:  slope={mk_obs_ms["slope"]:.5f} '
          f'95% CI=[{ci_obs_ms[0]:.6f}, {ci_obs_ms[1]:.6f}] '
          f'p={mk_obs_ms["p"]:.4f}' if mk_obs_ms else '  No MK for obs mean_suit')
    print(f'  FAO mean_suit:  slope={mk_fao_ms["slope"]:.5f} '
          f'95% CI=[{ci_fao_ms[0]:.6f}, {ci_fao_ms[1]:.6f}] '
          f'p={mk_fao_ms["p"]:.4f}' if mk_fao_ms else '  No MK for FAO mean_suit')
    print(f'  Obs area km²:   slope={mk_obs_ar["slope"]:.3f} '
          f'95% CI=[{ci_obs_ar[0]:.3f}, {ci_obs_ar[1]:.3f}] '
          f'p={mk_obs_ar["p"]:.4f}' if mk_obs_ar else '  No MK for obs area')
    print(f'  FAO area km²:   slope={mk_fao_ar["slope"]:.3f} '
          f'95% CI=[{ci_fao_ar[0]:.3f}, {ci_fao_ar[1]:.3f}] '
          f'p={mk_fao_ar["p"]:.4f}' if mk_fao_ar else '  No MK for FAO area')
    print('  Time series comparison saved.')


# ── Analysis 3: Class transition matrix ───────────────────────────────────────

def analysis_transition(mask):
    print('\n[Analysis 3] Class transition matrix ...')
    out_dir = f'{OUT_ROOT}/3_transition'

    obs_stacks, fao_stacks = {}, {}
    for crop in CROPS:
        tag = crop['tag']
        obs_yr, fao_yr = [], []
        for year in YEARS_POST:
            obs, _ = load_raster(obs_path(tag, year))
            fao, _ = load_raster(fao_path(tag, year))
            if obs is not None:
                obs_yr.append(apply_remap(obs, mask))
            if fao is not None:
                fao_yr.append(apply_remap(fao, mask))
        if obs_yr:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                obs_stacks[crop['label']] = np.nanmean(np.stack(obs_yr), axis=0)
                fao_stacks[crop['label']] = np.nanmean(np.stack(fao_yr), axis=0)

    all_results = []

    for label in obs_stacks:
        obs_arr = obs_stacks[label]
        fao_arr = fao_stacks[label]

        obs_cls = np.where(mask & np.isfinite(obs_arr),
                           np.clip(np.round(obs_arr).astype(int), 1, 5), -1)
        fao_cls = np.where(mask & np.isfinite(fao_arr),
                           np.clip(np.round(fao_arr).astype(int), 1, 5), -1)

        matrix = np.zeros((5, 5), dtype=int)
        valid = mask & (obs_cls > 0) & (fao_cls > 0)
        for r, c in zip(fao_cls[valid], obs_cls[valid]):
            matrix[r-1, c-1] += 1

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(matrix, cmap='Blues', aspect='auto')
        plt.colorbar(im, ax=ax, label='Pixel count')
        ax.set_xticks(range(5))
        ax.set_xticklabels([f'Class {i+1}' for i in range(5)], fontsize=9)
        ax.set_yticks(range(5))
        ax.set_yticklabels([f'Class {i+1}' for i in range(5)], fontsize=9)
        ax.set_xlabel('Observed (Permafrost) Class', fontsize=10)
        ax.set_ylabel('FAO Original Class', fontsize=10)
        for i in range(5):
            for j in range(5):
                if matrix[i, j] > 0:
                    ax.text(j, i, str(matrix[i, j]),
                            ha='center', va='center', fontsize=8,
                            color='white' if matrix[i, j] > matrix.max() * 0.5
                            else 'black')
        ax.set_title(f'{label} — Class Transition Matrix\n'
                     f'(FAO row vs Observed column, mean 1999-2018)',
                     fontsize=11, fontweight='bold')
        for i in range(5):
            ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1,
                         fill=False, edgecolor='red', linewidth=1.5))
        plt.tight_layout()
        fig.savefig(f'{out_dir}/{label.replace(" ", "_")}_transition.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

        n_total      = int(valid.sum())
        n_agree      = int(np.sum(obs_cls[valid] == fao_cls[valid]))
        n_obs_higher = int(np.sum(obs_cls[valid] > fao_cls[valid]))
        n_fao_higher = int(np.sum(obs_cls[valid] < fao_cls[valid]))
        all_results.append({
            'crop'          : label,
            'n_pixels'      : n_total,
            'pct_agree'     : round(100 * n_agree / n_total, 1) if n_total else np.nan,
            'pct_obs_higher': round(100 * n_obs_higher / n_total, 1) if n_total else np.nan,
            'pct_fao_higher': round(100 * n_fao_higher / n_total, 1) if n_total else np.nan,
        })

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        obs_overall = np.nanmean(np.stack(list(obs_stacks.values())), axis=0)
        fao_overall = np.nanmean(np.stack(list(fao_stacks.values())), axis=0)

    obs_cls = np.where(mask & np.isfinite(obs_overall),
                       np.clip(np.round(obs_overall).astype(int), 1, 5), -1)
    fao_cls = np.where(mask & np.isfinite(fao_overall),
                       np.clip(np.round(fao_overall).astype(int), 1, 5), -1)

    matrix = np.zeros((5, 5), dtype=int)
    valid = mask & (obs_cls > 0) & (fao_cls > 0)
    for r, c in zip(fao_cls[valid], obs_cls[valid]):
        matrix[r-1, c-1] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap='Blues', aspect='auto')
    plt.colorbar(im, ax=ax, label='Pixel count')
    ax.set_xticks(range(5))
    ax.set_xticklabels([f'Class {i+1}' for i in range(5)], fontsize=9)
    ax.set_yticks(range(5))
    ax.set_yticklabels([f'Class {i+1}' for i in range(5)], fontsize=9)
    ax.set_xlabel('Observed (Permafrost) Class', fontsize=10)
    ax.set_ylabel('FAO Original Class', fontsize=10)
    for i in range(5):
        for j in range(5):
            if matrix[i, j] > 0:
                ax.text(j, i, str(matrix[i, j]),
                        ha='center', va='center', fontsize=9,
                        color='white' if matrix[i, j] > matrix.max() * 0.5
                        else 'black')
    for i in range(5):
        ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1,
                     fill=False, edgecolor='red', linewidth=1.5))
    ax.set_title('OVERALL — Class Transition Matrix\n'
                 '(FAO Original row vs Observed column, mean 1999-2018)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{out_dir}/overall_transition.png', dpi=150, bbox_inches='tight')
    plt.close()

    n_total      = int(valid.sum())
    n_agree      = int(np.sum(obs_cls[valid] == fao_cls[valid]))
    n_obs_higher = int(np.sum(obs_cls[valid] > fao_cls[valid]))
    n_fao_higher = int(np.sum(obs_cls[valid] < fao_cls[valid]))
    all_results.append({
        'crop'          : 'OVERALL',
        'n_pixels'      : n_total,
        'pct_agree'     : round(100 * n_agree / n_total, 1) if n_total else np.nan,
        'pct_obs_higher': round(100 * n_obs_higher / n_total, 1) if n_total else np.nan,
        'pct_fao_higher': round(100 * n_fao_higher / n_total, 1) if n_total else np.nan,
    })

    df = pd.DataFrame(all_results)
    df.to_csv(f'{out_dir}/transition_summary.csv', index=False)
    print('\n  Transition summary:')
    print(df.to_string(index=False))


# ── Analysis 4: Permafrost drivers of methodology difference ──────────────────

def analysis_perm_drivers(mask):
    print('\n[Analysis 4] Permafrost drivers of methodology difference ...')
    out_dir  = f'{OUT_ROOT}/4_perm_drivers'
    os.makedirs(out_dir, exist_ok=True)

    from scipy.stats import spearmanr
    from matplotlib.colors import TwoSlopeNorm as TSN

    PERM_DIR  = r'./data_input/permafrost_yearly'
    PERM_VARS = [
        {'name': 'ALT',           'file': 'active_layer_depth.npy',
         'label': 'Active Layer Depth (m)',  'agg': 'max'},
        {'name': 'soil_moisture', 'file': 'avail_soil_moisture.npy',
         'label': 'Available Soil Moisture', 'agg': 'mean'},
    ]

    perm_data = {}
    for pv in PERM_VARS:
        stack, target = [], mask.shape
        for year in YEARS_ALL:
            path = f'{PERM_DIR}/{year}/{pv["file"]}'
            if not os.path.exists(path):
                stack.append(np.full(target, np.nan))
                continue
            arr = np.load(path).astype(float)
            if arr.ndim == 3:
                arr = np.nanmax(arr, axis=2) if pv['agg'] == 'max' else np.nanmean(arr, axis=2)
            if arr.shape != target:
                stack.append(np.full(target, np.nan))
                continue
            arr[~mask] = np.nan
            stack.append(arr)
        stack = np.array(stack)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mean_pre  = np.nanmean(stack[:len(YEARS_PRE)], axis=0)
            mean_post = np.nanmean(stack[len(YEARS_PRE):], axis=0)
        change = mean_post - mean_pre
        mean_post[~mask] = np.nan
        change[~mask]    = np.nan
        perm_data[pv['name']] = {
            'mean_post': mean_post, 'change': change, 'label': pv['label']
        }
        print(f'  {pv["name"]}: mean [{np.nanmin(mean_post):.3f}, {np.nanmax(mean_post):.3f}]')

    diff_dir = f'{OUT_ROOT}/1_diff_maps'
    diff_arrays = {}
    for crop in CROPS:
        path = f'{diff_dir}/{crop["tag"]}_diff.tif'
        arr, _ = load_raster(path)
        if arr is not None:
            arr[~mask] = np.nan
            diff_arrays[crop['label']] = arr
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        overall_diff = np.nanmean(np.stack(list(diff_arrays.values())), axis=0)
    overall_diff[~mask] = np.nan
    diff_arrays['OVERALL'] = overall_diff

    all_results = []

    for pv in PERM_VARS:
        pname = pv['name']
        for metric, perm_arr, metric_label in [
            ('mean',   perm_data[pname]['mean_post'], f'Mean {perm_data[pname]["label"]} (1999-2018)'),
            ('change', perm_data[pname]['change'],    f'Delta {perm_data[pname]["label"]} (post minus pre)'),
        ]:
            ncols = 4
            nrows = -(-len(diff_arrays) // ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
            axes = axes.flatten()
            for i, (crop_label, diff) in enumerate(diff_arrays.items()):
                ax = axes[i]
                valid = mask & np.isfinite(diff) & np.isfinite(perm_arr)
                x = perm_arr[valid]
                y = diff[valid]
                if len(x) < 5:
                    ax.set_title(f'{crop_label}\n(insufficient data)', fontsize=9)
                    ax.axis('off')
                    continue
                r, p = spearmanr(x, y)
                sig = '*' if p < 0.05 else ''
                ax.scatter(x, y, alpha=0.4, s=8,
                           color='#2166AC' if r > 0 else '#D6604D')
                ax.axhline(0, color='black', linewidth=0.6, linestyle='--')
                ax.axvline(float(np.nanmean(x)), color='grey', linewidth=0.6, linestyle=':')
                ax.set_xlabel(metric_label, fontsize=8)
                ax.set_ylabel('Obs minus FAO Suitability', fontsize=8)
                ax.set_title(f'{crop_label}\nr={r:.3f}, p={p:.3f}{sig}', fontsize=9, fontweight='bold')
                all_results.append({'crop': crop_label, 'perm_var': pname, 'metric': metric,
                                    'spearman_r': round(r, 4), 'p_value': round(p, 4),
                                    'significant': p < 0.05, 'n_pixels': int(valid.sum())})
            for j in range(len(diff_arrays), len(axes)):
                axes[j].axis('off')
            fig.suptitle(f'Drivers of Obs vs FAO Difference: {metric_label}\n* = p < 0.05',
                         fontsize=12, fontweight='bold')
            plt.tight_layout()
            fig.savefig(f'{out_dir}/{pname}_{metric}_driver_corr.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f'  Saved {pname} {metric} driver correlation')

    df = pd.DataFrame(all_results)
    df.to_csv(f'{out_dir}/driver_correlation_results.csv', index=False)

    crops_list = [c for c in df['crop'].unique() if c != 'OVERALL'] + ['OVERALL']
    perm_vs    = list(df['perm_var'].unique())
    for metric in ['mean', 'change']:
        df_m  = df[df['metric'] == metric]
        r_mat = pd.DataFrame(index=crops_list, columns=perm_vs, dtype=float)
        p_mat = pd.DataFrame(index=crops_list, columns=perm_vs, dtype=float)
        for _, row in df_m.iterrows():
            if row['crop'] in crops_list:
                r_mat.loc[row['crop'], row['perm_var']] = row['spearman_r']
                p_mat.loc[row['crop'], row['perm_var']] = row['p_value']
        fig, ax = plt.subplots(figsize=(len(perm_vs) * 3 + 1.5, len(crops_list) * 0.55 + 2))
        im = ax.imshow(r_mat.values.astype(float), cmap='RdBu', vmin=-0.5, vmax=0.5, aspect='auto')
        plt.colorbar(im, ax=ax, label='Spearman r', shrink=0.8)
        ax.set_xticks(range(len(perm_vs)))
        ax.set_xticklabels(['Active Layer Depth' if v == 'ALT' else 'Soil Moisture'
                            for v in perm_vs], rotation=20, ha='right', fontsize=10)
        ax.set_yticks(range(len(crops_list)))
        ax.set_yticklabels(crops_list, fontsize=10)
        if 'OVERALL' in crops_list:
            ax.axhline(crops_list.index('OVERALL') - 0.5, color='black', linewidth=1.5)
        for i, crop in enumerate(crops_list):
            for j, pv_n in enumerate(perm_vs):
                r_val = r_mat.loc[crop, pv_n]
                p_val = p_mat.loc[crop, pv_n]
                if pd.notna(r_val):
                    sig = '*' if pd.notna(p_val) and p_val < 0.05 else ''
                    ax.text(j, i, f'{r_val:.2f}{sig}', ha='center', va='center',
                            fontsize=9, color='white' if abs(r_val) > 0.35 else 'black',
                            fontweight='bold' if sig else 'normal')
        metric_lbl = 'Mean (1999-2018)' if metric == 'mean' else 'Change (post minus pre)'
        ax.set_title(f'Permafrost Drivers of Obs minus FAO Difference\n{metric_lbl}\nSpearman r | * = p < 0.05',
                     fontsize=11, fontweight='bold')
        plt.tight_layout()
        fig.savefig(f'{out_dir}/driver_heatmap_{metric}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved driver heatmap ({metric})')

    overall = diff_arrays['OVERALL'][mask]
    panels  = [
        ('ALT',           'mean',   'Mean Active Layer Depth (m)',   'Mean (1999-2018)'),
        ('soil_moisture', 'mean',   'Mean Available Soil Moisture',  'Mean (1999-2018)'),
        ('ALT',           'change', 'Active Layer Depth Change (m)', 'Change (post minus pre)'),
        ('soil_moisture', 'change', 'Soil Moisture Change',          'Change (post minus pre)'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    vlim = max(float(np.nanpercentile(np.abs(overall[np.isfinite(overall)]), 98)), 1e-4)
    for ax, (pvar, metric, xlabel, metric_label) in zip(axes, panels):
        perm_arr = perm_data[pvar]['mean_post'] if metric == 'mean' else perm_data[pvar]['change']
        x = perm_arr[mask]
        y = overall
        valid = np.isfinite(x) & np.isfinite(y)
        xv, yv = x[valid], y[valid]
        r, p = spearmanr(xv, yv)
        sig = '*' if p < 0.05 else ''
        from matplotlib.colors import TwoSlopeNorm as TSN
        norm = TSN(vmin=-vlim, vcenter=0, vmax=vlim)
        sc = ax.scatter(xv, yv, c=yv, cmap='RdBu', norm=norm, alpha=0.5, s=12, zorder=3)
        plt.colorbar(sc, ax=ax, shrink=0.75, label='Obs minus FAO')
        z = np.polyfit(xv, yv, 1)
        x_line = np.linspace(xv.min(), xv.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), color='black',
                linewidth=1.5, linestyle='--', alpha=0.7, zorder=5)
        ax.axhline(0, color='grey', linewidth=0.8, linestyle=':')
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Obs minus FAO (class units)', fontsize=10)
        ax.set_title(f'{metric_label}\nr = {r:.3f}, p = {p:.3f}{sig}', fontsize=11, fontweight='bold')
    fig.suptitle('What Drives the Obs vs FAO Difference?\nOverall mean difference vs permafrost variables\n* = p < 0.05',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{out_dir}/overall_driver_scatter_2x2.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved overall driver 2x2 scatter')

    print('\n  Significant driver correlations:')
    df_sig = df[df['significant']]
    if df_sig.empty:
        print('  None significant')
    else:
        print(df_sig[['crop', 'perm_var', 'metric', 'spearman_r', 'p_value']].to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    mask           = load_mask()
    pixel_area_km2 = build_pixel_area_km2(mask)

    # analysis_diff_maps(mask)
    analysis_time_series(mask, pixel_area_km2)
    # analysis_transition(mask)
    # analysis_perm_drivers(mask)

    print(f'\nAll FAO comparison outputs saved to: {OUT_ROOT}/')