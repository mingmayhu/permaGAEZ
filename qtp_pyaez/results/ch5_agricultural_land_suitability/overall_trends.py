"""
Chapter 5 — Agricultural Land Suitability Figures
===================================================
Produces publication-quality figures for Chapter 5 using the
OBSERVED scenario only (fixed-boundary suitability classes).

Classes 0 and 1 are combined into class 1 (not suitable).
Suitable land threshold is class >= 2.
Mean suitability ranges from 1-5.

Section 5.1 — 40-Year Suitability Trends (1979–2018):
  - Overall aggregate time series: mean suitability, suitable area (km²)
  - Stacked class distribution chart (overall, classes 2-5 only)
  - MK trend results table
  - Per-crop time series (supplementary)
  - Per-crop stacked class distribution (supplementary)

Section 5.2 — Spatial Hotspots of Suitability:
  - Overall aggregate mean suitability map (1999–2018)
  - Overall aggregate suitability change map (1999–2018 minus 1979–1998)
  - Per-crop mean suitability maps (supplementary)
  - Area expansion vs intensification analysis

Outputs written to:
  ./results/agricultural_land_suitability/outputs/
  ./results/agricultural_land_suitability/outputs/supplementary/
"""

import os
import numpy as np
from numpy.linalg import norm
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pymannkendall import original_test as mk_test
from osgeo import gdal
from matplotlib.colors import LinearSegmentedColormap

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR  = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH = r'./data_input/qilian_mask_new.tif'
OUT_ROOT  = r'./results/agricultural_land_suitability/outputs'
SUPP_DIR  = r'./results/agricultural_land_suitability/outputs/supplementary'

YEARS_ALL       = list(range(1979, 2019))
YEARS_PRE       = list(range(1979, 1999))
YEARS_POST      = list(range(1999, 2019))
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

# Classes 0 and 1 combined into class 1
CLASS_COLORS = {
    1: '#fc8d59',
    2: '#fee08b',
    3: '#d9ef8b',
    4: '#91cf60',
    5: '#1a9850',
}
CLASS_LABELS = {
    1: 'Class 1 (not suitable)',
    2: 'Class 2 (marginal)',
    3: 'Class 3 (moderate)',
    4: 'Class 4 (suitable)',
    5: 'Class 5 (very suitable)',
}

FONTSIZE_TITLE = 13
FONTSIZE_LABEL = 11
FONTSIZE_TICK  = 9
DPI = 150

os.chdir(WORK_DIR)
os.makedirs(OUT_ROOT,  exist_ok=True)
os.makedirs(SUPP_DIR,  exist_ok=True)


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

PERMAFROST_PATH = r'./data_input/permafrost_qilian.tif'

def load_mask():
    mask = load_raster(MASK_PATH).astype(bool)
    # Exclude lake pixels (nodata or 0 in the permafrost map)
    pf_arr = load_raster(PERMAFROST_PATH)
    if pf_arr is not None:
        lake_mask = ((pf_arr == 0) | ~np.isfinite(pf_arr)) & mask
        mask[lake_mask] = False
        print(f'  Excluded {lake_mask.sum()} lake/nodata pixels from mask')
    return mask

def build_pixel_area_km2(mask):
    """
    Build per-pixel area array (km²) using cosine latitude correction.
    Resolution is 0.1°. Area = (0.1 * 111.32)^2 * cos(lat_rad).
    Returns array same shape as mask; non-mask pixels are 0.
    """
    ds = gdal.Open(MASK_PATH)
    gt = ds.GetGeoTransform()
    # gt[3] = top-left latitude, gt[5] = pixel height (negative)
    nrows, ncols = mask.shape
    # latitude at centre of each row
    lats = gt[3] + gt[5] * (np.arange(nrows) + 0.5)
    deg_to_km = 111.32
    pixel_side_km = abs(gt[5]) * deg_to_km          # ~11.132 km at equator
    area_2d = np.outer(
        pixel_side_km * pixel_side_km * np.cos(np.deg2rad(lats)),
        np.ones(ncols)
    )
    area_2d[~mask] = 0.0
    return area_2d

def obs_suit_path(tag, year):
    return f'./data_output/final_classification_fixed/{tag}/{year}_suitability_class.tif'

def clean(arr, mask):
    out = arr.copy()
    out[~mask] = np.nan
    out[out < 0] = np.nan
    return out

def remap(arr_int):
    """Combine class 0 into class 1. Returns array with values in 1-5."""
    out = arr_int.copy()
    out[out == 0] = 1
    return out

def compute_metrics(arr, mask, pixel_area_km2):
    """Compute suitability metrics. Classes 0 and 1 combined into class 1.
    Mean suitability ranges 1-5. Suitable land = class >= 2 (area in km²).
    """
    arr_int = np.where(mask & np.isfinite(clean(arr, mask)), arr, 0).astype(int)
    arr_int = np.clip(arr_int, 0, 5)
    arr_int = remap(arr_int)
    return {
        'mean_suit'    : float(np.nanmean(arr_int[mask])),
        'area_ge2_km2' : float(np.sum(pixel_area_km2[arr_int >= 2])),
        'pct_1'        : float(np.mean(arr_int[mask] == 1) * 100),
        'pct_2'        : float(np.mean(arr_int[mask] == 2) * 100),
        'pct_3'        : float(np.mean(arr_int[mask] == 3) * 100),
        'pct_4'        : float(np.mean(arr_int[mask] == 4) * 100),
        'pct_5'        : float(np.mean(arr_int[mask] == 5) * 100),
    }

def run_mk(series):
    s = np.array(series, dtype=float)
    valid = np.isfinite(s)
    if valid.sum() < 4:
        return None
    mk = mk_test(s[valid])
    line = np.full(len(s), np.nan)
    line[valid] = mk.intercept + mk.slope * np.arange(valid.sum())
    return {
        'tau': round(mk.Tau, 3), 'p': round(mk.p, 4),
        'slope': round(mk.slope, 6), 'significant': mk.p < ALPHA,
        'trend': mk.trend, 'sen_line': line,
        'intercept': mk.intercept,
    }

def bootstrap_sen_ci(series, n_boot=1000, ci=95):
    """Bootstrap 95% CI on Sen's slope; returns (lo, hi)."""
    s = np.array(series, dtype=float)
    valid_idx = np.where(np.isfinite(s))[0]
    if len(valid_idx) < 4:
        return (np.nan, np.nan)
    s_valid = s[valid_idx]
    slopes = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = np.sort(rng.choice(len(s_valid), size=len(s_valid), replace=True))
        slopes.append(mk_test(s_valid[idx]).slope)
    lo = np.percentile(slopes, (100 - ci) / 2)
    hi = np.percentile(slopes, 100 - (100 - ci) / 2)
    return (lo, hi)

def plot_metric_obs(ax, years, series, mk_result, ylabel, title,
                    color='#2166AC', ci=None):
    """Plot observed-only time series with Sen's slope and optional CI band."""
    ax.plot(years, series, color=color, linewidth=2,
            marker='o', markersize=3, label='Observed')
    if mk_result:
        sig = '★' if mk_result['significant'] else ''
        ax.plot(years, mk_result['sen_line'], color='#D6604D',
                linewidth=2, linestyle='--',
                label=f"Sen's slope: {mk_result['slope']:.5f}/yr "
                      f"(p={mk_result['p']:.3f}){sig}")
        # CI band around Sen's slope line
        if ci is not None and not np.isnan(ci[0]):
            valid = np.isfinite(series)
            x_idx = np.arange(valid.sum())
            lo_line = np.full(len(series), np.nan)
            hi_line = np.full(len(series), np.nan)
            lo_line[valid] = mk_result['intercept'] + ci[0] * x_idx
            hi_line[valid] = mk_result['intercept'] + ci[1] * x_idx
            ax.fill_between(years, lo_line, hi_line,
                            color='#D6604D', alpha=0.15,
                            label='95% CI (bootstrap)')
    ax.set_xlabel('Year', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL)
    ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xticks(years[::4])
    ax.set_xticklabels(years[::4], rotation=45, ha='right',
                       fontsize=FONTSIZE_TICK)
    ax.tick_params(axis='y', labelsize=FONTSIZE_TICK)

def plot_stacked_class(ax, years, cls_data, title):
    """Stacked area chart — classes 2-5 only (class 1 = not suitable, excluded)."""
    bottom = np.zeros(len(years))
    for c in [2, 3, 4, 5]:
        vals = np.array(cls_data[f'pct_{c}'])
        vals = np.where(np.isfinite(vals), vals, 0.0)
        ax.fill_between(years, bottom, bottom + vals,
                        color=CLASS_COLORS[c], alpha=0.85,
                        label=CLASS_LABELS[c])
        bottom += vals
    ax.set_xlabel('Year', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('% of Mask Pixels (class ≥ 2)', fontsize=FONTSIZE_LABEL)
    ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.set_ylim(0, 5)
    ax.set_xticks(years[::4])
    ax.set_xticklabels(years[::4], rotation=45, ha='right',
                       fontsize=FONTSIZE_TICK)
    ax.tick_params(axis='y', labelsize=FONTSIZE_TICK)


# ── Build annual series per crop ──────────────────────────────────────────────

def build_crop_series(tag, mask, pixel_area_km2):
    """Build annual observed suitability metrics for one crop."""
    metrics = ['mean_suit', 'area_ge2_km2',
               'pct_1', 'pct_2', 'pct_3', 'pct_4', 'pct_5']
    data = {m: [] for m in metrics}

    for year in YEARS_ALL:
        arr = load_raster(obs_suit_path(tag, year))
        m   = compute_metrics(arr, mask, pixel_area_km2) if arr is not None else \
              {k: np.nan for k in metrics}
        for k in metrics:
            data[k].append(m[k])

    return {k: np.array(v) for k, v in data.items()}


# ── Section 5.1: 40-Year Suitability Trends ───────────────────────────────────

def section_trends(mask, pixel_area_km2):
    print('\n[5.1] 40-year suitability trends …')
    years_arr = np.array(YEARS_ALL)
    mk_results = []

    all_mean     = []
    all_area_ge2 = []
    all_cls      = {c: [] for c in [1, 2, 3, 4, 5]}

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        print(f'  Loading {label} …')
        data = build_crop_series(tag, mask, pixel_area_km2)

        all_mean.append(data['mean_suit'])
        all_area_ge2.append(data['area_ge2_km2'])
        for c in [1, 2, 3, 4, 5]:
            all_cls[c].append(data[f'pct_{c}'])

        mk_ms = run_mk(data['mean_suit'])
        mk_a2 = run_mk(data['area_ge2_km2'])

        for metric, mk in [('mean_suit', mk_ms), ('area_ge2_km2', mk_a2)]:
            if mk:
                mk_results.append({
                    'crop': label, 'metric': metric,
                    'tau': mk['tau'], 'p': mk['p'],
                    'slope': mk['slope'], 'significant': mk['significant'],
                    'trend': mk['trend'],
                })

        # ── Supplementary: per-crop time series ───────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(21, 5))
        plot_metric_obs(axes[0], years_arr, data['mean_suit'], mk_ms,
                        'Mean Suitability Score (1–5)', 'Mean Suitability (all pixels)')
        plot_metric_obs(axes[1], years_arr, data['area_ge2_km2'], mk_a2,
                        'Suitable Land Area (km²)', 'Suitable Land Area (class ≥ 2)')
        fig.suptitle(f'{label} — Suitability Trends (1979–2018)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(f'{SUPP_DIR}/{crop["tag"]}_trends.png',
                    dpi=DPI, bbox_inches='tight')
        plt.close()

        # ── Supplementary: per-crop stacked class distribution ────────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_stacked_class(ax, years_arr, data,
                           f'{label} — Suitability Class Distribution (1979–2018)')
        handles = [mpatches.Patch(color=CLASS_COLORS[c], label=CLASS_LABELS[c])
                   for c in [2, 3, 4, 5]]
        ax.legend(handles=handles, loc='upper left', fontsize=8,
                  bbox_to_anchor=(1.01, 1))
        plt.tight_layout()
        fig.savefig(f'{SUPP_DIR}/{crop["tag"]}_class_dist.png',
                    dpi=DPI, bbox_inches='tight')
        plt.close()

    # ── Overall aggregate ──────────────────────────────────────────────────────
    obs_mean_agg     = np.nanmean(all_mean,     axis=0)
    obs_area_ge2_agg = np.nanmean(all_area_ge2, axis=0)
    cls_agg = {f'pct_{c}': np.nanmean(all_cls[c], axis=0) for c in [1, 2, 3, 4, 5]}

    mk_ms_agg = run_mk(obs_mean_agg)
    mk_a2_agg = run_mk(obs_area_ge2_agg)

    # Bootstrap CIs on overall aggregates
    print('  Bootstrapping Sen slope CIs …')
    ci_ms_agg = bootstrap_sen_ci(obs_mean_agg)
    ci_a2_agg = bootstrap_sen_ci(obs_area_ge2_agg)

    for metric, mk in [('mean_suit', mk_ms_agg), ('area_ge2_km2', mk_a2_agg)]:
        if mk:
            mk_results.append({
                'crop': 'OVERALL', 'metric': metric,
                'tau': mk['tau'], 'p': mk['p'],
                'slope': mk['slope'], 'significant': mk['significant'],
                'trend': mk['trend'],
            })

    # ── Figure: All crops + overall on one plot ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(CROPS)))

    for ax, metric, all_data, agg_data, mk, ylabel, title in [
        (axes[0], 'mean_suit',    all_mean,     obs_mean_agg,     mk_ms_agg,
         'Mean Suitability Score (1–5)',  '(a) Mean suitability across all crops'),
        (axes[1], 'area_ge2_km2', all_area_ge2, obs_area_ge2_agg, mk_a2_agg,
         'Suitable Land Area (km²)',      '(b) Suitable land area across all crops'),
    ]:
        ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight='bold')
        ax.set_xlabel('Year', fontsize=FONTSIZE_LABEL)
        ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL)
        for i, (crop, series) in enumerate(zip(CROPS, all_data)):
            ax.plot(years_arr, series, color=colors[i], linewidth=1,
                    alpha=0.6, label=crop['label'])
        ax.plot(years_arr, agg_data, color='black', linewidth=2.5,
                label='Overall mean', zorder=5)
        if mk:
            sig = '★' if mk['significant'] else ''
            ax.plot(years_arr, mk['sen_line'], color='black', linewidth=1.5,
                    linestyle='--',
                    label=f"Sen's slope: {mk['slope']:.5f}/yr (p={mk['p']:.3f}){sig}")
        ax.set_xticks(years_arr[::4])
        ax.set_xticklabels(years_arr[::4], rotation=45, ha='right',
                           fontsize=FONTSIZE_TICK)
        ax.tick_params(axis='y', labelsize=FONTSIZE_TICK)

    axes[0].legend(fontsize=7, loc='upper left', ncol=2)
    axes[1].legend(fontsize=7, loc='upper left', ncol=2)
    fig.suptitle('Qilian Mountain Region — Suitability Trends by Crop (1979–2018)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'{OUT_ROOT}/all_crops_trends.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print('  ✓ All-crops trend figure saved')

    # ── Figure 1: Overall time series ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(21, 5))
    plot_metric_obs(axes[0], years_arr, obs_mean_agg, mk_ms_agg,
                    'Mean Suitability Score (1–5)',
                    'Overall Mean Suitability Score\n(all pixels, all crops)',
                    ci=ci_ms_agg)
    plot_metric_obs(axes[1], years_arr, obs_area_ge2_agg, mk_a2_agg,
                    'Suitable Land Area (km²)',
                    'Overall Suitable Land Area\n(class ≥ 2, all crops)',
                    ci=ci_a2_agg)
    fig.suptitle('Qilian Mountain Region — Agricultural Suitability Trends (1979–2018)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{OUT_ROOT}/overall_suitability_trends.png',
                dpi=DPI, bbox_inches='tight')
    plt.close()
    print('  ✓ Overall trend figure saved')
    print(f'  Mean suitability CI: [{ci_ms_agg[0]:.6f}, {ci_ms_agg[1]:.6f}]')
    print(f'  Suitable area CI:    [{ci_a2_agg[0]:.3f}, {ci_a2_agg[1]:.3f}] km²/yr')

    # ── Figure 2: Overall stacked class distribution ──────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_stacked_class(ax, years_arr, cls_agg,
                       'Overall Suitability Class Distribution (1979–2018)\n'
                       'Mean across all 10 crops')
    handles = [mpatches.Patch(color=CLASS_COLORS[c], label=CLASS_LABELS[c])
               for c in [2, 3, 4, 5]]
    ax.legend(handles=handles, loc='upper left', fontsize=9,
              bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    fig.savefig(f'{OUT_ROOT}/overall_class_distribution.png',
                dpi=DPI, bbox_inches='tight')
    plt.close()
    print('  ✓ Overall class distribution saved')

    pd.DataFrame(mk_results).to_csv(f'{OUT_ROOT}/suitability_mk_results.csv',
                                    index=False)
    print('  ✓ MK results saved')


# ── Section 5.2: Spatial Hotspots ────────────────────────────────────────────

def section_hotspots(mask, pixel_area_km2):
    print('\n[5.2] Spatial hotspots …')

    suit_pre_all  = []
    suit_post_all = []

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        print(f'  Loading {label} …')

        pre_stack, post_stack = [], []
        for year in YEARS_ALL:
            arr = load_raster(obs_suit_path(tag, year))
            if arr is None:
                continue
            arr = clean(arr, mask)
            arr_int = np.where(mask & np.isfinite(arr), arr, 0).astype(int)
            arr_int = np.clip(arr_int, 0, 5)
            arr_int = remap(arr_int)
            arr_flt = arr_int.astype(float)
            arr_flt[~mask] = np.nan
            if year in YEARS_PRE:
                pre_stack.append(arr_flt)
            else:
                post_stack.append(arr_flt)

        mean_pre  = np.nanmean(pre_stack,  axis=0) if pre_stack  else None
        mean_post = np.nanmean(post_stack, axis=0) if post_stack else None

        if mean_pre is not None:
            suit_pre_all.append(mean_pre)
        if mean_post is not None:
            suit_post_all.append(mean_post)

        # ── Supplementary: per-crop suitability maps ──────────────────────────
        if mean_pre is not None and mean_post is not None:
            change = mean_post - mean_pre
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            cmap_suit = plt.get_cmap('RdYlGn')

            for ax, arr, title in [
                (axes[0], mean_pre,  'Mean Suitability\n1979–1998'),
                (axes[1], mean_post, 'Mean Suitability\n1999–2018'),
            ]:
                disp = np.where(mask, arr, np.nan)
                im = ax.imshow(disp, cmap=cmap_suit, vmin=1, vmax=5)
                ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight='bold')
                ax.axis('off')
                plt.colorbar(im, ax=ax, shrink=0.75, label='Mean Class')

            disp_chg = np.where(mask, change, np.nan)
            vlim = np.nanpercentile(
                np.abs(disp_chg[mask & np.isfinite(disp_chg)]), 98)
            vlim = max(vlim, 0.1)
            im2 = axes[2].imshow(disp_chg, cmap='RdBu', vmin=-vlim, vmax=vlim)
            axes[2].set_title('ΔSuitability\n(1999–2018 minus 1979–1998)',
                               fontsize=FONTSIZE_TITLE, fontweight='bold')
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], shrink=0.75, label='Δ Class')

            fig.suptitle(f'{label} — Suitability Spatial Patterns',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()
            fig.savefig(f'{SUPP_DIR}/{tag}_spatial.png',
                        dpi=DPI, bbox_inches='tight')
            plt.close()

    # ── Overall aggregate spatial maps ────────────────────────────────────────
    agg_pre  = np.nanmean(suit_pre_all,  axis=0)
    agg_post = np.nanmean(suit_post_all, axis=0)
    agg_chg  = agg_post - agg_pre

    agg_pre[~mask]  = np.nan
    agg_post[~mask] = np.nan
    agg_chg[~mask]  = np.nan

    # ── Area expansion vs intensification analysis ────────────────────────────
    print('\n  Computing area expansion vs intensification …')

    crop_labels      = [c['label'] for c in CROPS]
    expansion_counts = []
    class_units      = []

    for crop_idx, crop in enumerate(CROPS):
        pre  = suit_pre_all[crop_idx]
        post = suit_post_all[crop_idx]

        pre_cls  = np.round(pre).astype(int)
        post_cls = np.round(post).astype(int)
        valid    = mask & np.isfinite(pre) & np.isfinite(post)

        newly_suitable   = valid & (pre_cls <= 1) & (post_cls >= 2)
        lost_suitable    = valid & (pre_cls >= 2) & (post_cls <= 1)
        already_suitable = valid & (pre_cls >= 2) & (post_cls >= 2)

        if already_suitable.sum() > 0:
            intensification = float(np.nanmean(post[already_suitable]) -
                                    np.nanmean(pre[already_suitable]))
        else:
            intensification = np.nan

        expansion_counts.append({
            'crop':                        crop['label'],
            'newly_suitable_px':           int(newly_suitable.sum()),
            'lost_suitable_px':            int(lost_suitable.sum()),
            'net_expansion_px':            int(newly_suitable.sum() - lost_suitable.sum()),
            'intensification_delta_class': round(intensification, 4) if not np.isnan(intensification) else np.nan,
        })

        expansion_units = float(np.nansum(post[newly_suitable] - pre[newly_suitable])) \
                          if newly_suitable.sum() > 0 else 0.0
        intensification_units = float(np.nansum(post[already_suitable] - pre[already_suitable])) \
                                 if already_suitable.sum() > 0 else 0.0
        lost_units = float(np.nansum(post[lost_suitable] - pre[lost_suitable])) \
                     if lost_suitable.sum() > 0 else 0.0

        total = expansion_units + intensification_units + lost_units
        class_units.append({
            'crop':                      crop['label'],
            'expansion_units':           round(expansion_units, 2),
            'intensification_units':     round(intensification_units, 2),
            'lost_units':                round(lost_units, 2),
            'total_units':               round(total, 2),
            'pct_from_expansion':        round(100 * expansion_units / total, 1) if total > 0 else np.nan,
            'pct_from_intensification':  round(100 * intensification_units / total, 1) if total > 0 else np.nan,
            'pct_from_loss':             round(100 * lost_units / total, 1) if total > 0 else np.nan,
        })

    df_exp   = pd.DataFrame(expansion_counts)
    df_units = pd.DataFrame(class_units)

    df_exp.to_csv(f'{OUT_ROOT}/area_expansion_vs_intensification.csv', index=False)
    df_units.to_csv(f'{OUT_ROOT}/expansion_vs_intensification_units.csv', index=False)
    print(df_units.to_string(index=False))

    # ── Figure: stacked % bar chart ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(crop_labels))

    ax.bar(x, df_units['pct_from_intensification'],
           color='#2166AC', alpha=0.85, label='Intensification')
    ax.bar(x, df_units['pct_from_expansion'],
           bottom=df_units['pct_from_intensification'],
           color='#92C5DE', alpha=0.85, label='Expansion')

    ax.set_xticks(x)
    ax.set_xticklabels(crop_labels, rotation=30, ha='right', fontsize=FONTSIZE_TICK)
    ax.set_ylabel('% of Total Suitability Gain', fontsize=FONTSIZE_LABEL)
    ax.set_ylim(0, 100)
    ax.axhline(50, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)
    ax.set_title('Contribution to Suitability Gain:\nExpansion vs Intensification (1979–1998 to 1999–2018)',
                 fontsize=FONTSIZE_TITLE, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{OUT_ROOT}/expansion_vs_intensification_pct.png',
                dpi=DPI, bbox_inches='tight')
    plt.close()
    print('  ✓ % contribution figure saved')

    # ── Figure: expansion vs intensification bar chart ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x     = np.arange(len(crop_labels))
    width = 0.35

    axes[0].bar(x - width/2, df_exp['newly_suitable_px'], width,
                label='Newly suitable (≤1→≥2)', color='#2166AC', alpha=0.85)
    axes[0].bar(x + width/2, df_exp['lost_suitable_px'], width,
                label='Lost suitable (≥2→≤1)', color='#D6604D', alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(crop_labels, rotation=30, ha='right', fontsize=FONTSIZE_TICK)
    axes[0].set_ylabel('Number of Pixels', fontsize=FONTSIZE_LABEL)
    axes[0].set_title('Area Expansion\n(Pixels crossing class 1→2 threshold)',
                      fontsize=FONTSIZE_TITLE, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].axhline(0, color='black', linewidth=0.8)

    bar_colors = ['#2166AC' if v >= 0 else '#D6604D'
                  for v in df_exp['intensification_delta_class']]
    axes[1].bar(x, df_exp['intensification_delta_class'],
                color=bar_colors, alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(crop_labels, rotation=30, ha='right', fontsize=FONTSIZE_TICK)
    axes[1].set_ylabel('Δ Mean Class (already-suitable pixels)', fontsize=FONTSIZE_LABEL)
    axes[1].set_title('Intensification\n(Mean class change among class ≥2 pixels\nin both periods)',
                      fontsize=FONTSIZE_TITLE, fontweight='bold')
    axes[1].axhline(0, color='black', linewidth=0.8)

    fig.suptitle('Area Expansion vs Intensification (1979–1998 to 1999–2018)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'{OUT_ROOT}/expansion_vs_intensification.png',
                dpi=DPI, bbox_inches='tight')
    plt.close()
    print('  ✓ Expansion vs intensification figure saved')

    # ── Figure 3: Overall mean suitability + change ───────────────────────────
    cmap_suit = plt.get_cmap('RdYlGn')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5),
                             gridspec_kw={'wspace': 0.15})

    for ax, arr, title in [
        (axes[0], agg_pre,  'Mean Suitability\n1979–1998'),
        (axes[1], agg_post, 'Mean Suitability\n1999–2018'),
    ]:
        disp = np.where(mask, arr, np.nan)
        im = ax.imshow(disp, cmap=cmap_suit, vmin=1, vmax=5)
        ax.set_facecolor('#cccccc')
        ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.6, label='Mean Class')

    disp_chg = np.where(mask, agg_chg, np.nan)
    neg_prop = 0.05 / (0.05 + 1.1)
    colors_list = [(0.9, 0.5, 0.5), (0.98, 0.98, 0.98), (0.129, 0.400, 0.675)]
    nodes = [0.0, neg_prop, 1.0]
    cmap_chg = LinearSegmentedColormap.from_list('custom_RdBu',
                list(zip(nodes, colors_list)))

    im2 = axes[2].imshow(disp_chg, cmap=cmap_chg, vmin=-0.05, vmax=1.1)
    axes[2].set_facecolor('#cccccc')
    plt.colorbar(im2, ax=axes[2], shrink=0.6, label='Δ Class')
    axes[2].set_title('ΔSuitability\n(1999–2018 minus 1979–1998)',
                      fontsize=FONTSIZE_TITLE, fontweight='bold')
    axes[2].axis('off')

    fig.suptitle(
        'Qilian Mountain Region — Overall Agricultural Suitability\n'
        '(Mean across all 10 crops)',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    fig.savefig(f'{OUT_ROOT}/overall_suitability_spatial.png',
                dpi=DPI, bbox_inches='tight')
    plt.close()
    print('  ✓ Overall spatial maps saved')

    # ── Figure 4: Change hotspot map — all crops in small multiples ───────────
    n_crops = len(CROPS)
    ncols   = 5
    nrows   = -(-n_crops // ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4, nrows * 3.5))
    axes = axes.flatten()

    for i, (crop, pre, post) in enumerate(
            zip(CROPS, suit_pre_all, suit_post_all)):
        chg  = post - pre
        chg[~mask] = np.nan
        disp = np.where(mask, chg, np.nan)
        vals = disp[mask & np.isfinite(disp)]
        vlim = np.nanpercentile(np.abs(vals), 98) if len(vals) > 0 else 0.5
        vlim = max(vlim, 0.1)
        im = axes[i].imshow(disp, cmap='RdBu', vmin=-vlim, vmax=vlim)
        axes[i].set_title(crop['label'], fontsize=9, fontweight='bold')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], shrink=0.8, label='Δ Class')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(
        'ΔSuitability per Crop (1999–2018 minus 1979–1998)\n'
        'Supplementary — Individual Crop Hotspots',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    fig.savefig(f'{SUPP_DIR}/all_crops_delta_suitability.png',
                dpi=DPI, bbox_inches='tight')
    plt.close()
    print('  ✓ All-crop change panel saved to supplementary')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    mask           = load_mask()
    pixel_area_km2 = build_pixel_area_km2(mask)
    section_trends(mask, pixel_area_km2)
    section_hotspots(mask, pixel_area_km2)
    print(f'\n✓ All Chapter 5 figures saved to: {OUT_ROOT}/')
    print(f'  Supplementary figures in: {SUPP_DIR}/')