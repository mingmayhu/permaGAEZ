"""
Step 6 — Box Plots of Environmental Characteristics by Pixel Category
======================================================================
Compares distributions of environmental variables across the five
pixel categories from Step 2 (sign consistency & Wilcoxon):

  4 = Significantly positive  (thaw consistently helps)
  3 = Consistently positive   (not significant)
  2 = Mixed / no effect
  1 = Consistently negative
  0 = Significantly negative  (thaw consistently hurts)

Variables compared:
  - Elevation (m)
  - Slope (°)
  - Mean ALT 1999–2018 (m)
  - Change in ALT (1999–2018 minus 1979–1998)
  - Mean soil moisture 1999–2018
  - Change in soil moisture
  - Mean temperature (TempMax, TempMin) 1999–2018
  - Change in temperature
  - Mean annual precipitation 1999–2018
  - Change in precipitation

Statistical test between categories: Kruskal-Wallis + pairwise Mann-Whitney U

Outputs per crop and overall:
  - {tag}_boxplots.png        — box plots for all variables
  - {tag}_kruskal.csv         — Kruskal-Wallis test results
  - overall_boxplots.png
  - overall_kruskal.csv

Outputs written to:
  ./results_analysis/outputs/6_spatial_analysis/6_pixel_characteristics/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu
from osgeo import gdal

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR   = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH  = r'./data_input/qilian mask.tif'
ELEV_PATH  = r'./data_input/terrain/elevation.npy'
SLOPE_PATH = r'./data_input/terrain/slope.tif'
PERM_DIR   = r'./data_input/permafrost_yearly'
CLIM_DIR   = r'./data_input/climate_yearly'
CLASS_DIR  = r'./results_analysis/outputs/6_spatial_analysis/2_sign_consistency'
OUT_ROOT   = r'./results_analysis/outputs/6_spatial_analysis/6_pixel_characteristics'

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

# Pixel category codes and labels
CAT_CODES  = [0, 1, 2, 3, 4]
CAT_LABELS = {
    4: 'Sig+',
    3: 'Cons+',
    2: 'Mixed',
    1: 'Cons−',
    0: 'Sig−',
}
CAT_COLORS = {
    4: '#1a6faf',
    3: '#92c5de',
    2: '#f0f0f0',
    1: '#f4a582',
    0: '#c0392b',
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
    return load_raster(MASK_PATH).astype(bool)

def load_perm_annual(var_file, years, mask, agg='max'):
    """Load permafrost variable, aggregate across days, return mean over years."""
    stack = []
    for year in years:
        path = f'{PERM_DIR}/{year}/{var_file}'
        if not os.path.exists(path):
            continue
        arr = np.load(path).astype(float)
        if arr.ndim == 3:
            arr = np.nanmax(arr, axis=2) if agg == 'max' else np.nanmean(arr, axis=2)
        arr[~mask] = np.nan
        stack.append(arr)
    return np.nanmean(stack, axis=0) if stack else np.full(mask.shape, np.nan)

def load_clim_annual(var_file, years, mask, agg='mean'):
    """Load climate variable, aggregate across days, return mean over years."""
    stack = []
    for year in years:
        path = f'{CLIM_DIR}/{year}/{var_file}'
        if not os.path.exists(path):
            continue
        arr = np.load(path).astype(float)
        if arr.ndim == 3:
            arr = np.nansum(arr, axis=2) if agg == 'sum' else np.nanmean(arr, axis=2)
        arr[~mask] = np.nan
        stack.append(arr)
    return np.nanmean(stack, axis=0) if stack else np.full(mask.shape, np.nan)

def kruskal_test(groups):
    """Run Kruskal-Wallis test across groups. Returns (H, p)."""
    clean = [g[np.isfinite(g)] for g in groups if len(g[np.isfinite(g)]) > 1]
    if len(clean) < 2:
        return np.nan, np.nan
    try:
        H, p = kruskal(*clean)
        return round(float(H), 3), round(float(p), 4)
    except Exception:
        return np.nan, np.nan

def mannwhitney_pairs(groups_dict):
    """
    Pairwise Mann-Whitney U between significant positive (4)
    and all other categories. Returns dict of p-values.
    """
    results = {}
    ref = groups_dict.get(4)
    if ref is None or len(ref[np.isfinite(ref)]) < 2:
        return results
    ref_clean = ref[np.isfinite(ref)]
    for code in [3, 2, 1, 0]:
        grp = groups_dict.get(code)
        if grp is None or len(grp[np.isfinite(grp)]) < 2:
            results[f'sig+_vs_{CAT_LABELS[code]}'] = np.nan
            continue
        try:
            _, p = mannwhitneyu(ref_clean, grp[np.isfinite(grp)],
                                alternative='two-sided')
            results[f'sig+_vs_{CAT_LABELS[code]}'] = round(float(p), 4)
        except Exception:
            results[f'sig+_vs_{CAT_LABELS[code]}'] = np.nan
    return results


# ── Load environmental variables ──────────────────────────────────────────────

def load_env_vars(mask):
    print('Loading environmental variables …')
    elev  = np.load(ELEV_PATH)
    slope, _ = load_raster(SLOPE_PATH), None
    slope = load_raster(SLOPE_PATH)
    slope[~mask] = np.nan

    # Permafrost — mean and change
    alt_post  = load_perm_annual('active_layer_depth.npy', YEARS_POST, mask, 'max')
    alt_pre   = load_perm_annual('active_layer_depth.npy', YEARS_PRE,  mask, 'max')
    alt_chg   = alt_post - alt_pre

    sm_post   = load_perm_annual('avail_soil_moisture.npy', YEARS_POST, mask, 'mean')
    sm_pre    = load_perm_annual('avail_soil_moisture.npy', YEARS_PRE,  mask, 'mean')
    sm_chg    = sm_post - sm_pre

    # Climate — mean and change
    tmax_post = load_clim_annual('TempMax.npy', YEARS_POST, mask, 'mean')
    tmax_pre  = load_clim_annual('TempMax.npy', YEARS_PRE,  mask, 'mean')
    tmax_chg  = tmax_post - tmax_pre

    tmin_post = load_clim_annual('TempMin.npy', YEARS_POST, mask, 'mean')
    tmin_pre  = load_clim_annual('TempMin.npy', YEARS_PRE,  mask, 'mean')
    tmin_chg  = tmin_post - tmin_pre

    prec_post = load_clim_annual('Precip.npy', YEARS_POST, mask, 'sum')
    prec_pre  = load_clim_annual('Precip.npy', YEARS_PRE,  mask, 'sum')
    prec_chg  = prec_post - prec_pre

    env_vars = {
        'Elevation (m)'          : elev,
        'Slope (°)'              : slope,
        'Mean ALT (m)'           : alt_post,
        'ΔALT (m)'               : alt_chg,
        'Mean Soil Moisture'     : sm_post,
        'ΔSoil Moisture'         : sm_chg,
        'Mean TempMax (°C)'      : tmax_post,
        'ΔTempMax (°C)'          : tmax_chg,
        'Mean TempMin (°C)'      : tmin_post,
        'ΔTempMin (°C)'          : tmin_chg,
        'Mean Precip (mm)'       : prec_post,
        'ΔPrecip (mm)'           : prec_chg,
    }
    print('  ✓ Environmental variables loaded.')
    return env_vars


# ── Box plot function ──────────────────────────────────────────────────────────

def make_boxplots(cls_map, env_vars, mask, label, out_path, kruskal_path):
    """
    For a given classification map, produce box plots comparing
    environmental variable distributions across pixel categories.
    """
    # Extract pixel values per category
    cat_data = {}
    for code in CAT_CODES:
        px = mask & (cls_map == code) & np.isfinite(cls_map)
        cat_data[code] = px

    # Count pixels per category
    counts = {code: int(np.sum(cat_data[code])) for code in CAT_CODES}
    print(f'  Pixel counts: ' +
          ' | '.join([f'{CAT_LABELS[c]}: {counts[c]}' for c in CAT_CODES]))

    n_vars = len(env_vars)
    ncols  = 4
    nrows  = -(-n_vars // ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 4))
    axes = axes.flatten()

    kruskal_results = []

    for i, (var_name, var_arr) in enumerate(env_vars.items()):
        ax = axes[i]

        groups      = {}
        plot_data   = []
        plot_labels = []
        plot_colors = []

        for code in CAT_CODES:
            if counts[code] == 0:
                continue
            vals = var_arr[cat_data[code]]
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            groups[code]   = vals
            plot_data.append(vals)
            plot_labels.append(f'{CAT_LABELS[code]}\n(n={len(vals)})')
            plot_colors.append(CAT_COLORS[code])

        if not plot_data:
            ax.axis('off')
            continue

        # Box plot
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

        # Kruskal-Wallis test
        H, p = kruskal_test(list(groups.values()))
        sig  = '★' if (not np.isnan(p) and p < 0.05) else ''
        ax.text(0.98, 0.97, f'KW p={p:.3f}{sig}',
                transform=ax.transAxes, fontsize=7.5,
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Pairwise Mann-Whitney vs Sig+
        mw = mannwhitney_pairs(groups)

        row = {
            'variable'    : var_name,
            'kruskal_H'   : H,
            'kruskal_p'   : p,
            'kruskal_sig' : (not np.isnan(p)) and p < 0.05,
        }
        # Add pairwise columns — always include all four even if NaN
        for code in [3, 2, 1, 0]:
            key = f'sig+_vs_{CAT_LABELS[code]}'
            row[key] = mw.get(key, np.nan)

        kruskal_results.append(row)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=CAT_COLORS[c], label=CAT_LABELS[c])
                       for c in CAT_CODES if counts[c] > 0]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f'{label} — Environmental Characteristics by Pixel Category\n'
        f'★ = Kruskal-Wallis significant at p < 0.05',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    pd.DataFrame(kruskal_results).to_csv(kruskal_path, index=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(mask):
    env_vars = load_env_vars(mask)

    # ── Per-crop ──────────────────────────────────────────────────────────────
    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        print(f'\n── {label} ──')

        cls_path = f'{CLASS_DIR}/{tag}_classification.tif'
        cls_map  = load_raster(cls_path)
        if cls_map is None:
            print(f'  ⚠ Missing classification map for {label}')
            continue
        cls_map[~mask] = np.nan

        make_boxplots(
            cls_map, env_vars, mask, label,
            out_path     = f'{OUT_ROOT}/{tag}_boxplots.png',
            kruskal_path = f'{OUT_ROOT}/{tag}_kruskal.csv',
        )
        print(f'  ✓ {label} saved')

    # ── Overall aggregate ──────────────────────────────────────────────────────
    # For each pixel, compute net score = n_crops_sig+ - n_crops_sig-
    # Then group into net positive / neutral / net negative
    print('\n── Overall aggregate ──')

    sig_pos_stack = []
    sig_neg_stack = []

    for crop in CROPS:
        tag = crop['tag']
        cls_path = f'{CLASS_DIR}/{tag}_classification.tif'
        cls_map  = load_raster(cls_path)
        if cls_map is None:
            continue
        cls_map[~mask] = np.nan
        sig_pos_stack.append((cls_map == 4).astype(float))
        sig_neg_stack.append((cls_map == 0).astype(float))

    if sig_pos_stack:
        n_sig_pos = np.nansum(sig_pos_stack, axis=0)   # 0–10
        n_sig_neg = np.nansum(sig_neg_stack, axis=0)   # 0–10
        net_score = n_sig_pos - n_sig_neg               # -10 to +10

        # Classification based on net score
        # Net positive: ≥2 more crops benefit than hurt
        # Net negative: ≤-2 more crops hurt than benefit
        # Neutral: -1 to +1
        NET_POS  = 4
        NEUTRAL  = 2
        NET_NEG  = 0

        net_cls = np.full(mask.shape, np.nan)
        net_cls[mask & (net_score >= 2)]  = NET_POS
        net_cls[mask & (net_score <= -2)] = NET_NEG
        net_cls[mask & (net_score > -2) & (net_score < 2)] = NEUTRAL

        # Override labels and colors for this analysis
        overall_cat_codes  = [NET_POS, NEUTRAL, NET_NEG]
        overall_cat_labels = {
            NET_POS : 'Net+ (≥2)',
            NEUTRAL : 'Neutral',
            NET_NEG : 'Net− (≤-2)',
        }
        overall_cat_colors = {
            NET_POS : '#1a6faf',
            NEUTRAL : '#f0f0f0',
            NET_NEG : '#c0392b',
        }

        counts = {code: int(np.sum(mask & (net_cls == code)))
                  for code in overall_cat_codes}
        print(f'  Net score pixel counts: ' +
              ' | '.join([f'{overall_cat_labels[c]}: {counts[c]}'
                          for c in overall_cat_codes]))

        # Build box plots with three-category scheme
        n_vars = len(env_vars)
        ncols  = 4
        nrows  = -(-n_vars // ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 4.5, nrows * 4))
        axes = axes.flatten()
        kruskal_results = []

        for i, (var_name, var_arr) in enumerate(env_vars.items()):
            ax = axes[i]

            groups      = {}
            plot_data   = []
            plot_labels = []
            plot_colors = []

            for code in overall_cat_codes:
                if counts[code] == 0:
                    continue
                px   = mask & (net_cls == code)
                vals = var_arr[px]
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    continue
                groups[code]   = vals
                plot_data.append(vals)
                plot_labels.append(
                    f'{overall_cat_labels[code]}\n(n={len(vals)})')
                plot_colors.append(overall_cat_colors[code])

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

            # Kruskal-Wallis
            H, p = kruskal_test(list(groups.values()))
            sig  = '★' if (not np.isnan(p) and p < 0.05) else ''
            ax.text(0.98, 0.97, f'KW p={p:.3f}{sig}',
                    transform=ax.transAxes, fontsize=7.5,
                    ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            # Pairwise Mann-Whitney: Net+ vs Neutral and Net+ vs Net-
            row = {
                'variable'    : var_name,
                'kruskal_H'   : H,
                'kruskal_p'   : p,
                'kruskal_sig' : (not np.isnan(p)) and p < 0.05,
            }
            ref = groups.get(NET_POS)
            for code, key in [(NEUTRAL, 'net+_vs_neutral'),
                              (NET_NEG,  'net+_vs_net-')]:
                grp = groups.get(code)
                if ref is not None and grp is not None and \
                   len(ref) > 1 and len(grp) > 1:
                    try:
                        _, p_mw = mannwhitneyu(ref, grp,
                                               alternative='two-sided')
                        row[key] = round(float(p_mw), 4)
                    except Exception:
                        row[key] = np.nan
                else:
                    row[key] = np.nan

            kruskal_results.append(row)

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=overall_cat_colors[c],
                  label=overall_cat_labels[c])
            for c in overall_cat_codes if counts[c] > 0
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=3,
                   fontsize=9, bbox_to_anchor=(0.5, -0.02))

        fig.suptitle(
            'Overall — Environmental Characteristics by Net Thaw Score\n'
            'Net score = crops where thaw sig+ minus crops where thaw sig−\n'
            '★ = Kruskal-Wallis significant at p < 0.05',
            fontsize=12, fontweight='bold', y=1.01
        )
        plt.tight_layout()
        fig.savefig(f'{OUT_ROOT}/overall_boxplots.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

        pd.DataFrame(kruskal_results).to_csv(
            f'{OUT_ROOT}/overall_kruskal.csv', index=False)
        print('  ✓ Overall saved')

    print(f'\n✓ All outputs saved to: {OUT_ROOT}/')


if __name__ == '__main__':
    mask = load_mask()
    run(mask)