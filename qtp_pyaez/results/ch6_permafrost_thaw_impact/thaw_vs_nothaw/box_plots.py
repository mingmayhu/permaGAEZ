"""
Box Plots of Environmental Characteristics by Pixel Category
=============================================================
Compares distributions of environmental variables across the five
pixel categories from the Wilcoxon sign consistency map:

  4 = Significantly positive  (thaw consistently helps)
  3 = Consistently positive   (not significant)
  2 = Mixed / no effect
  1 = Consistently negative
  0 = Significantly negative  (thaw consistently hurts)

Variables compared:
  - Elevation (m)
  - Slope (degrees)
  - Mean ALT 1999-2018 (m)
  - Change in ALT (1999-2018 minus 1979-1998)
  - Mean soil moisture 1999-2018
  - Change in soil moisture
  - Mean temperature 1999-2018
  - Change in temperature
  - Mean annual precipitation 1999-2018
  - Change in precipitation

Statistical test: Kruskal-Wallis + pairwise Mann-Whitney U (sig+ vs others)

Lake pixels excluded via permafrost_qilian.tif.

Outputs written to:
  ./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/9_boxplots/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import kruskal, mannwhitneyu
from osgeo import gdal

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR        = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH       = r'./data_input/qilian mask.tif'
PERMAFROST_PATH = r'./data_input/permafrost_qilian.tif'
ELEV_PATH       = r'./data_input/terrain/elevation.npy'
SLOPE_PATH      = r'./data_input/terrain/slope.tif'
PERM_DIR        = r'./data_input/permafrost_yearly'
CLIM_DIR        = r'./data_input/climate_yearly'
CLASS_DIR       = r'./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/5_spatial/2_sign_consistency'
OUT_ROOT        = r'./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/9_boxplots'

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

CAT_CODES  = [0, 1, 2, 3, 4]
CAT_LABELS = {4: 'Sig+', 3: 'Cons+', 2: 'Mixed', 1: 'Cons-', 0: 'Sig-'}
CAT_COLORS = {
    4: '#1a6faf', 3: '#92c5de', 2: '#f0f0f0', 1: '#f4a582', 0: '#c0392b',
}

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
    arr = load_raster(MASK_PATH)
    mask = arr.astype(bool)
    pf_arr = load_raster(PERMAFROST_PATH)
    if pf_arr is not None:
        lake_mask = ((pf_arr == 0) | ~np.isfinite(pf_arr)) & mask
        mask[lake_mask] = False
        print(f'  Excluded {lake_mask.sum()} lake/nodata pixels from mask')
    return mask

def load_perm_annual(var_file, years, mask, agg='max'):
    stack = []
    for year in years:
        path = f'{PERM_DIR}/{year}/{var_file}'
        if not os.path.exists(path):
            continue
        arr = np.load(path).astype(float)
        if arr.ndim == 3:
            arr = np.nanmax(arr, axis=2) if agg == 'max' \
                  else np.nanmean(arr, axis=2)
        arr[~mask] = np.nan
        stack.append(arr)
    return np.nanmean(stack, axis=0) if stack else np.full(mask.shape, np.nan)

def load_clim_annual(var_file, years, mask, agg='mean'):
    stack = []
    for year in years:
        path = f'{CLIM_DIR}/{year}/{var_file}'
        if not os.path.exists(path):
            continue
        arr = np.load(path).astype(float)
        if arr.ndim == 3:
            arr = np.nansum(arr, axis=2) if agg == 'sum' \
                  else np.nanmean(arr, axis=2)
        arr[~mask] = np.nan
        stack.append(arr)
    return np.nanmean(stack, axis=0) if stack else np.full(mask.shape, np.nan)

def kruskal_test(groups):
    clean = [g[np.isfinite(g)] for g in groups if len(g[np.isfinite(g)]) > 1]
    if len(clean) < 2:
        return np.nan, np.nan
    try:
        H, p = kruskal(*clean)
        return round(float(H), 3), round(float(p), 4)
    except Exception:
        return np.nan, np.nan

def mannwhitney_pairs(groups_dict):
    results = {}
    ref = groups_dict.get(4)
    if ref is None or len(ref[np.isfinite(ref)]) < 2:
        return results
    ref_clean = ref[np.isfinite(ref)]
    for code in [3, 2, 1, 0]:
        grp = groups_dict.get(code)
        key = f'sig+_vs_{CAT_LABELS[code]}'
        if grp is None or len(grp[np.isfinite(grp)]) < 2:
            results[key] = np.nan
            continue
        try:
            _, p = mannwhitneyu(ref_clean, grp[np.isfinite(grp)],
                                alternative='two-sided')
            results[key] = round(float(p), 4)
        except Exception:
            results[key] = np.nan
    return results


# ── Load environmental variables ──────────────────────────────────────────────

def load_env_vars(mask):
    print('Loading environmental variables ...')
    elev  = np.load(ELEV_PATH)
    slope = load_raster(SLOPE_PATH)
    slope[~mask] = np.nan

    alt_post = load_perm_annual('active_layer_depth.npy',  YEARS_POST, mask, 'max')
    alt_pre  = load_perm_annual('active_layer_depth.npy',  YEARS_PRE,  mask, 'max')
    sm_post  = load_perm_annual('avail_soil_moisture.npy', YEARS_POST, mask, 'mean')
    sm_pre   = load_perm_annual('avail_soil_moisture.npy', YEARS_PRE,  mask, 'mean')

    tmax_post = load_clim_annual('TempMax.npy', YEARS_POST, mask, 'mean')
    tmax_pre  = load_clim_annual('TempMax.npy', YEARS_PRE,  mask, 'mean')
    tmin_post = load_clim_annual('TempMin.npy', YEARS_POST, mask, 'mean')
    tmin_pre  = load_clim_annual('TempMin.npy', YEARS_PRE,  mask, 'mean')
    prec_post = load_clim_annual('Precip.npy',  YEARS_POST, mask, 'sum')
    prec_pre  = load_clim_annual('Precip.npy',  YEARS_PRE,  mask, 'sum')

    # Derive mean temperature from max and min
    tmean_post = (tmax_post + tmin_post) / 2
    tmean_pre  = (tmax_pre  + tmin_pre)  / 2

    return {
        'Elevation (m)'       : elev,
        'Slope (degrees)'     : slope,
        'Mean ALT (m)'        : alt_post,
        'Delta ALT (m)'       : alt_post - alt_pre,
        'Mean Soil Moisture'  : sm_post,
        'Delta Soil Moisture' : sm_post - sm_pre,
        'Mean Temperature (C)': tmean_post,
        'Delta Temperature (C)': tmean_post - tmean_pre,
        'Mean Precip (mm)'    : prec_post,
        'Delta Precip (mm)'   : prec_post - prec_pre,
    }


# ── Box plot function ──────────────────────────────────────────────────────────

def make_boxplots(cls_map, env_vars, mask, label, out_path, kruskal_path,
                  cat_codes=None, cat_labels=None, cat_colors=None):
    if cat_codes  is None: cat_codes  = CAT_CODES
    if cat_labels is None: cat_labels = CAT_LABELS
    if cat_colors is None: cat_colors = CAT_COLORS

    cat_data = {}
    counts   = {}
    for code in cat_codes:
        px = mask & (cls_map == code) & np.isfinite(cls_map)
        cat_data[code] = px
        counts[code]   = int(np.sum(px))

    print(f'  Pixel counts: ' +
          ' | '.join([f'{cat_labels[c]}: {counts[c]}' for c in cat_codes]))

    n_vars = len(env_vars)
    ncols  = 4
    nrows  = -(-n_vars // ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 4))
    axes = axes.flatten()
    kruskal_results = []

    for i, (var_name, var_arr) in enumerate(env_vars.items()):
        ax = axes[i]
        groups, plot_data, plot_labels, plot_colors = {}, [], [], []

        for code in cat_codes:
            if counts[code] == 0:
                continue
            vals = var_arr[cat_data[code]]
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            groups[code] = vals
            plot_data.append(vals)
            plot_labels.append(f'{cat_labels[code]}\n(n={len(vals)})')
            plot_colors.append(cat_colors[code])

        if not plot_data:
            ax.axis('off')
            continue

        bp = ax.boxplot(plot_data, patch_artist=True,
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2),
                        flierprops=dict(marker='o', markersize=2,
                                        alpha=0.4, linestyle='none'))
        for patch, color in zip(bp['boxes'], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

        ax.set_xticklabels(plot_labels, fontsize=8)
        ax.set_ylabel(var_name, fontsize=9)
        ax.set_title(var_name, fontsize=10, fontweight='bold')

        H, p = kruskal_test(list(groups.values()))
        sig  = '*' if (not np.isnan(p) and p < 0.05) else ''
        ax.text(0.98, 0.97, f'KW p={p:.3f}{sig}',
                transform=ax.transAxes, fontsize=7.5,
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        mw  = mannwhitney_pairs(groups)
        row = {'variable': var_name, 'kruskal_H': H, 'kruskal_p': p,
               'kruskal_sig': (not np.isnan(p)) and p < 0.05}
        for code in [3, 2, 1, 0]:
            key = f'sig+_vs_{CAT_LABELS[code]}'
            row[key] = mw.get(key, np.nan)
        kruskal_results.append(row)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    legend_elements = [Patch(facecolor=cat_colors[c], label=cat_labels[c])
                       for c in cat_codes if counts[c] > 0]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f'{label} — Environmental Characteristics by Pixel Category\n'
        f'* = Kruskal-Wallis significant at p < 0.05',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    pd.DataFrame(kruskal_results).to_csv(kruskal_path, index=False)
    print(f'  Saved: {os.path.basename(out_path)}')


# ── Main ──────────────────────────────────────────────────────────────────────

def run(mask):
    env_vars = load_env_vars(mask)

    # ── Per-crop ──────────────────────────────────────────────────────────────
    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        print(f'\n-- {label} --')
        cls_path = f'{CLASS_DIR}/{tag}_classification.tif'
        cls_map  = load_raster(cls_path)
        if cls_map is None:
            print(f'  Warning: missing classification map for {label}')
            continue
        cls_map[~mask] = np.nan
        make_boxplots(
            cls_map, env_vars, mask, label,
            out_path     = f'{OUT_ROOT}/{tag}_boxplots.png',
            kruskal_path = f'{OUT_ROOT}/{tag}_kruskal.csv',
        )

    # ── Overall aggregate ──────────────────────────────────────────────────────
    # Use the overall_classification.tif from spatial_analysis.py
    # Classification codes:
    #   1 = significantly positive   (Wilcoxon p<0.05, median>0)
    #   2 = consistently positive    (>=70% years positive, not sig)
    #   3 = mixed / no effect
    #   4 = consistently negative    (<=30% years positive, not sig)
    #   5 = significantly negative   (Wilcoxon p<0.05, median<0)
    #
    # Grouped into 3 categories:
    #   Positive = 1 + 2  (sig positive + consistently positive)
    #   Mixed    = 3
    #   Negative = 4 + 5  (consistently negative + sig negative)

    print('\n-- Overall aggregate (3-category grouped) --')

    # Try loading the overall classification raster from spatial_analysis
    overall_cls_path = ('./results/permafrost_thaw_impact/thaw_vs_nothaw/' +
                        'outputs/5_spatial/2_wilcoxon/overall_classification.tif')
    overall_cls = load_raster(overall_cls_path)

    if overall_cls is None:
        # Fall back to per-crop majority vote
        print('  overall_classification.tif not found, computing from per-crop maps ...')
        pos_stack, neg_stack = [], []
        for crop in CROPS:
            cls_map = load_raster(
                f'{CLASS_DIR}/{crop["tag"]}_classification.tif')
            if cls_map is None:
                continue
            cls_map[~mask] = np.nan
            pos_stack.append(((cls_map == 1) | (cls_map == 2)).astype(float))
            neg_stack.append(((cls_map == 4) | (cls_map == 5)).astype(float))
        if not pos_stack:
            print('  No classification maps found.')
            return
        n_pos = np.nansum(pos_stack, axis=0)
        n_neg = np.nansum(neg_stack, axis=0)
        net   = n_pos - n_neg
        overall_cls = np.full(mask.shape, np.nan)
        overall_cls[mask & (net > 0)]  = 1   # positive
        overall_cls[mask & (net == 0)] = 3   # mixed
        overall_cls[mask & (net < 0)]  = 5   # negative
    else:
        overall_cls[~mask] = np.nan
        print('  Loaded overall_classification.tif')

    # Remap to 3-group scheme: positive=1, mixed=2, negative=3
    grouped_cls = np.full(mask.shape, np.nan)
    grouped_cls[mask & ((overall_cls == 1) | (overall_cls == 2))] = 1  # Positive
    grouped_cls[mask & (overall_cls == 3)]                         = 2  # Mixed
    grouped_cls[mask & ((overall_cls == 4) | (overall_cls == 5))] = 3  # Negative

    n_pos = int(np.sum(grouped_cls[mask] == 1))
    n_mix = int(np.sum(grouped_cls[mask] == 2))
    n_neg = int(np.sum(grouped_cls[mask] == 3))
    print(f'  Positive: {n_pos} px | Mixed: {n_mix} px | Negative: {n_neg} px')

    grp_cat_codes  = [1, 2, 3]
    grp_cat_labels = {1: 'Positive\n(sig+consistent)',
                      2: 'Mixed',
                      3: 'Negative\n(sig+consistent)'}
    grp_cat_colors = {1: '#2166AC', 2: '#d9d9d9', 3: '#D6604D'}

    make_boxplots(
        grouped_cls, env_vars, mask,
        label        = 'Overall — 3-Category Grouped',
        out_path     = f'{OUT_ROOT}/overall_boxplots_3cat.png',
        kruskal_path = f'{OUT_ROOT}/overall_kruskal_3cat.csv',
        cat_codes    = grp_cat_codes,
        cat_labels   = grp_cat_labels,
        cat_colors   = grp_cat_colors,
    )
    print(f'\nAll outputs saved to: {OUT_ROOT}/')


if __name__ == '__main__':
    mask = load_mask()
    run(mask)