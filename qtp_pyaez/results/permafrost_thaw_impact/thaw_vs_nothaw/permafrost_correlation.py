"""
Permafrost Variable Correlation with delta Suitability
=======================================================
Correlates mean delta Suitability (per pixel, 1999-2018) with:
  - Active layer depth (ALT)
  - Available soil moisture

Class 0 and class 1 are combined into class 1 (remap) before all calculations.
Lake pixels excluded via permafrost_qilian.tif.

For each variable computes:
  A) Mean value across 1999-2018
  B) Change = mean(1999-2018) minus mean(1979-1998)

Analyses:
  A) Pixel-wise spatial correlation (Spearman)
  B) Temporal correlation (Spearman)
  C) Summary heatmap (mean + change side by side)
  D) Overall 2x2 scatter (mean ALT, mean SM, change ALT, change SM)

Outputs written to:
  ./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/8_permafrost_corr/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import spearmanr
from osgeo import gdal

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR        = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH       = r'./data_input/qilian mask.tif'
PERMAFROST_PATH = r'./data_input/permafrost_qilian.tif'
PERM_DIR        = r'./data_input/permafrost_yearly'
DELTA_DIR       = r'./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/5_spatial/1_mean_delta'
OUT_ROOT        = r'./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/8_permafrost_corr'

YEARS_ALL = list(range(1979, 2019))
YEARS_PRE = list(range(1979, 1999))
YEARS_CF  = list(range(1999, 2019))

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

PERM_VARS = [
    {'name': 'ALT',           'file': 'active_layer_depth.npy',  'label': 'Active Layer Depth (m)',   'agg': 'max'},
    {'name': 'soil_moisture', 'file': 'avail_soil_moisture.npy', 'label': 'Available Soil Moisture',  'agg': 'mean'},
]

PERM_VAR_LABELS = {
    'ALT'          : 'Active Layer Depth',
    'soil_moisture': 'Soil Moisture',
}
METRIC_LABELS = {
    'mean'  : 'Mean (1999-2018)',
    'change': 'Change (post minus pre)',
}

os.chdir(WORK_DIR)
os.makedirs(OUT_ROOT, exist_ok=True)
for sub in ['A_spatial', 'B_temporal']:
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

def load_mask():
    arr, _ = load_raster(MASK_PATH)
    mask = arr.astype(bool)
    pf_arr, _ = load_raster(PERMAFROST_PATH)
    if pf_arr is not None:
        lake_mask = ((pf_arr == 0) | ~np.isfinite(pf_arr)) & mask
        mask[lake_mask] = False
        print(f'  Excluded {lake_mask.sum()} lake/nodata pixels from mask')
    return mask

def apply_remap(arr, mask):
    """Apply mask and remap class 0 to 1. Returns float array."""
    arr_c = arr.copy()
    arr_c[arr_c < 0] = np.nan
    arr_c[~mask] = np.nan
    return np.where(
        np.isfinite(arr_c),
        remap(np.where(np.isfinite(arr_c), arr_c, 0).astype(int)).astype(float),
        np.nan
    )

def obs_suit_path(tag, year):
    return f'./data_output/final_classification_fixed/{tag}/{year}_suitability_class.tif'

def cf_suit_path(tag, year):
    return f'./data_output/final_classification_nothaw_fixed/{tag}/{year}_suitability_class.tif'

def load_mean_delta(tag, mask):
    """Load pre-computed mean delta raster. Falls back to recomputing if missing."""
    path = f'{DELTA_DIR}/{tag}_mean_delta_suit.tif'
    arr, _ = load_raster(path)
    if arr is not None:
        arr[~mask] = np.nan
        return arr
    # Fallback: recompute with remap
    print(f'  Warning: {path} missing, recomputing ...')
    stack = []
    for year in YEARS_CF:
        obs, _ = load_raster(obs_suit_path(tag, year))
        cf,  _ = load_raster(cf_suit_path(tag, year))
        if obs is None or cf is None:
            continue
        obs_r = apply_remap(obs, mask)
        cf_r  = apply_remap(cf,  mask)
        d = np.where(np.isfinite(obs_r) & np.isfinite(cf_r), obs_r - cf_r, np.nan)
        d[~mask] = np.nan
        stack.append(d)
    if not stack:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(np.stack(stack), axis=0)

def load_perm_stack(var_file, years, mask, agg='mean'):
    """Load permafrost variable for a list of years.
    Arrays are (rows, cols, days) — aggregate across days.
    Returns (n_years, rows, cols).
    """
    stack        = []
    target_shape = mask.shape
    for year in years:
        path = f'{PERM_DIR}/{year}/{var_file}'
        if not os.path.exists(path):
            stack.append(np.full(target_shape, np.nan))
            continue
        arr = np.load(path).astype(float)
        if arr.ndim == 3:
            arr = np.nanmax(arr, axis=2) if agg == 'max' else np.nanmean(arr, axis=2)
        if arr.shape != target_shape:
            stack.append(np.full(target_shape, np.nan))
            continue
        arr[~mask] = np.nan
        stack.append(arr)
    return np.array(stack)

def regional_mean(arr, mask):
    valid = mask & np.isfinite(arr)
    return float(np.nanmean(arr[valid])) if valid.any() else np.nan


# ── Load data ─────────────────────────────────────────────────────────────────

def load_permafrost_data(mask):
    print('Loading permafrost data ...')
    perm_data = {}
    for pv in PERM_VARS:
        stack_all  = load_perm_stack(pv['file'], YEARS_ALL, mask, agg=pv['agg'])
        stack_pre  = stack_all[:len(YEARS_PRE)]
        stack_post = stack_all[len(YEARS_PRE):]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mean_post = np.nanmean(stack_post, axis=0)
            mean_pre  = np.nanmean(stack_pre,  axis=0)

        change = mean_post - mean_pre
        mean_post[~mask] = np.nan
        change[~mask]    = np.nan

        perm_data[pv['name']] = {
            'stack_post': stack_post,
            'mean_post' : mean_post,
            'change'    : change,
            'label'     : pv['label'],
        }
        print(f'  {pv["name"]}: mean [{np.nanmin(mean_post):.3f}, {np.nanmax(mean_post):.3f}]'
              f'  change [{np.nanmin(change):.3f}, {np.nanmax(change):.3f}]')
    return perm_data


def load_delta_arrays(mask):
    print('\nLoading mean delta suitability rasters ...')
    delta_arrays = {}
    for crop in CROPS:
        delta = load_mean_delta(crop['tag'], mask)
        if delta is not None:
            delta_arrays[crop['label']] = delta

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        overall = np.nanmean(np.stack(list(delta_arrays.values())), axis=0)
    overall[~mask] = np.nan
    delta_arrays['OVERALL'] = overall
    print(f'  Loaded {len(delta_arrays)-1} crops + OVERALL')
    return delta_arrays


def build_annual_delta_series(mask):
    """Build annual regional mean delta suitability with remap applied."""
    print('\nBuilding annual delta suitability series ...')
    annual_delta = {}
    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        series = []
        for year in YEARS_CF:
            obs, _ = load_raster(obs_suit_path(tag, year))
            cf,  _ = load_raster(cf_suit_path(tag, year))
            if obs is None or cf is None:
                series.append(np.nan)
                continue
            obs_r = apply_remap(obs, mask)
            cf_r  = apply_remap(cf,  mask)
            delta = np.where(np.isfinite(obs_r) & np.isfinite(cf_r),
                             obs_r - cf_r, np.nan)
            valid = mask & np.isfinite(delta)
            series.append(float(np.nanmean(delta[valid])) if valid.any() else np.nan)
        annual_delta[label] = np.array(series)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        overall = np.nanmean(np.stack(list(annual_delta.values())), axis=0)
    annual_delta['OVERALL'] = overall
    return annual_delta


# ── Analysis A: Pixel-wise spatial correlation ─────────────────────────────────

def analysis_spatial(mask, perm_data, delta_arrays):
    print('\n[Analysis A] Pixel-wise spatial correlation ...')
    out_dir     = f'{OUT_ROOT}/A_spatial'
    all_results = []

    for pv in PERM_VARS:
        pname  = pv['name']
        plabel = perm_data[pname]['label']

        for metric, perm_arr, metric_label in [
            ('mean',   perm_data[pname]['mean_post'],
             f'Mean {plabel} (1999-2018)'),
            ('change', perm_data[pname]['change'],
             f'Delta {plabel} (1999-2018 minus 1979-1998)'),
        ]:
            ncols = 4
            nrows = -(-len(delta_arrays) // ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
            axes = axes.flatten()

            for i, (crop_label, delta) in enumerate(delta_arrays.items()):
                ax = axes[i]
                valid = mask & np.isfinite(delta) & np.isfinite(perm_arr)
                x = perm_arr[valid]
                y = delta[valid]

                if len(x) < 5:
                    ax.set_title(f'{crop_label}\n(insufficient data)', fontsize=9)
                    ax.axis('off')
                    continue

                r, p = spearmanr(x, y)
                sig  = '*' if p < 0.05 else ''

                ax.scatter(x, y, alpha=0.4, s=8,
                           color='#2166AC' if r > 0 else '#D6604D')
                ax.axhline(0, color='black', linewidth=0.6, linestyle='--')
                ax.axvline(float(np.nanmean(x)), color='grey',
                           linewidth=0.6, linestyle=':')
                ax.set_xlabel(metric_label, fontsize=8)
                ax.set_ylabel('Mean delta Suitability (class units)', fontsize=8)
                ax.set_title(f'{crop_label}\nr={r:.3f}, p={p:.3f}{sig}',
                             fontsize=9, fontweight='bold')

                all_results.append({
                    'analysis'   : 'spatial',
                    'crop'       : crop_label,
                    'perm_var'   : pname,
                    'metric'     : metric,
                    'spearman_r' : round(r, 4),
                    'p_value'    : round(p, 4),
                    'significant': p < 0.05,
                    'n_pixels'   : int(valid.sum()),
                })

            for j in range(len(delta_arrays), len(axes)):
                axes[j].axis('off')

            fig.suptitle(
                f'Pixel-wise Spatial Correlation: {metric_label}\n'
                f'vs Mean delta Suitability | * = p < 0.05 | Class 0 and 1 combined',
                fontsize=12, fontweight='bold'
            )
            plt.tight_layout()
            fig.savefig(f'{out_dir}/{pname}_{metric}_spatial_corr.png',
                        dpi=150, bbox_inches='tight')
            plt.close()
            print(f'  Saved {pname} {metric} spatial correlation')

    return all_results


# ── Analysis B: Temporal correlation ──────────────────────────────────────────

def analysis_temporal(mask, perm_data, annual_delta):
    print('\n[Analysis B] Temporal correlation ...')
    out_dir     = f'{OUT_ROOT}/B_temporal'
    years_arr   = np.array(YEARS_CF)
    all_results = []

    for pv in PERM_VARS:
        pname  = pv['name']
        plabel = perm_data[pname]['label']
        stack  = perm_data[pname]['stack_post']

        annual_perm = np.array([
            regional_mean(stack[i], mask) for i in range(stack.shape[0])
        ])

        ncols = 4
        nrows = -(-len(annual_delta) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
        axes = axes.flatten()

        for i, (crop_label, delta_series) in enumerate(annual_delta.items()):
            ax = axes[i]
            valid = np.isfinite(annual_perm) & np.isfinite(delta_series)

            if valid.sum() < 4:
                ax.set_title(f'{crop_label}\n(insufficient data)', fontsize=9)
                ax.axis('off')
                continue

            x = annual_perm[valid]
            y = delta_series[valid]
            r, p = spearmanr(x, y)
            sig  = '*' if p < 0.05 else ''

            sc = ax.scatter(x, y, c=years_arr[valid], cmap='viridis',
                            s=50, zorder=3)
            plt.colorbar(sc, ax=ax, shrink=0.7, label='Year')
            ax.axhline(0, color='black', linewidth=0.6, linestyle='--')
            ax.set_xlabel(f'Regional Mean {plabel}', fontsize=8)
            ax.set_ylabel('Regional Mean delta Suitability', fontsize=8)
            ax.set_title(f'{crop_label}\nr={r:.3f}, p={p:.3f}{sig}',
                         fontsize=9, fontweight='bold')

            all_results.append({
                'analysis'   : 'temporal',
                'crop'       : crop_label,
                'perm_var'   : pname,
                'metric'     : 'annual_mean',
                'spearman_r' : round(r, 4),
                'p_value'    : round(p, 4),
                'significant': p < 0.05,
                'n_years'    : int(valid.sum()),
            })

        for j in range(len(annual_delta), len(axes)):
            axes[j].axis('off')

        fig.suptitle(
            f'Temporal Correlation: Annual Regional Mean {plabel}\n'
            f'vs Regional Mean delta Suitability (1999-2018)'
            f' | * = p < 0.05 | Class 0 and 1 combined',
            fontsize=12, fontweight='bold'
        )
        plt.tight_layout()
        fig.savefig(f'{out_dir}/{pname}_temporal_corr.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved {pname} temporal correlation')

    return all_results


# ── Summary heatmap ────────────────────────────────────────────────────────────

def plot_summary_heatmap(df, analysis, out_dir):
    """Combined heatmap: mean and change side by side."""
    df_a = df[df['analysis'] == analysis].copy()
    if df_a.empty:
        return

    crops    = [c for c in df_a['crop'].unique() if c != 'OVERALL'] + ['OVERALL']
    perm_vs  = list(df_a['perm_var'].unique())
    metrics  = [m for m in ['mean', 'change', 'annual_mean']
                if m in df_a['metric'].unique()]

    def build_matrix(df_sub):
        r_mat = pd.DataFrame(index=crops, columns=perm_vs, dtype=float)
        p_mat = pd.DataFrame(index=crops, columns=perm_vs, dtype=float)
        for _, row in df_sub.iterrows():
            if row['crop'] in crops and row['perm_var'] in perm_vs:
                r_mat.loc[row['crop'], row['perm_var']] = row['spearman_r']
                p_mat.loc[row['crop'], row['perm_var']] = row['p_value']
        return r_mat, p_mat

    def draw_heatmap(ax, r_mat, p_mat, title, vmin=-0.5, vmax=0.5):
        r_vals = r_mat.values.astype(float)
        im = ax.imshow(r_vals, cmap='RdBu', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks(range(len(perm_vs)))
        ax.set_xticklabels([PERM_VAR_LABELS.get(v, v) for v in perm_vs],
                           rotation=20, ha='right', fontsize=10)
        ax.set_yticks(range(len(crops)))
        ax.set_yticklabels(crops, fontsize=10)
        if 'OVERALL' in crops:
            ax.axhline(crops.index('OVERALL') - 0.5,
                       color='black', linewidth=1.5)
        for i, crop in enumerate(crops):
            for j, pv in enumerate(perm_vs):
                r_val = r_mat.loc[crop, pv]
                p_val = p_mat.loc[crop, pv]
                if pd.notna(r_val):
                    sig = '*' if pd.notna(p_val) and p_val < 0.05 else ''
                    txt_col = 'white' if abs(r_val) > 0.35 else 'black'
                    ax.text(j, i, f'{r_val:.2f}{sig}',
                            ha='center', va='center', fontsize=9,
                            color=txt_col,
                            fontweight='bold' if sig else 'normal')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        return im

    if len(metrics) == 1:
        fig, ax = plt.subplots(figsize=(len(perm_vs) * 3 + 1.5, len(crops) * 0.55 + 2))
        df_m        = df_a[df_a['metric'] == metrics[0]]
        r_mat, p_mat = build_matrix(df_m)
        im = draw_heatmap(ax, r_mat, p_mat,
                          METRIC_LABELS.get(metrics[0], metrics[0]))
        plt.colorbar(im, ax=ax, label='Spearman r', shrink=0.8)
        fig.suptitle(
            f'Permafrost Correlation with delta Suitability ({analysis.capitalize()})\n'
            f'Spearman r | * = p < 0.05 | Class 0 and 1 combined',
            fontsize=12, fontweight='bold'
        )
        plt.tight_layout()
        fig.savefig(f'{out_dir}/{analysis}_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(len(perm_vs) * 6, len(crops) * 0.55 + 3))
        for ax, metric in zip(axes, ['mean', 'change']):
            df_m         = df_a[df_a['metric'] == metric]
            r_mat, p_mat = build_matrix(df_m)
            im = draw_heatmap(ax, r_mat, p_mat,
                              METRIC_LABELS.get(metric, metric))
            if ax != axes[0]:
                ax.set_yticklabels([])
        fig.subplots_adjust(right=0.88, wspace=0.05)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap='RdBu',
                                   norm=plt.Normalize(vmin=-0.5, vmax=0.5))
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label='Spearman r')
        fig.suptitle(
            'Spatial Correlation: Permafrost Variables vs Mean delta Suitability\n'
            'Spearman r | * = p < 0.05 | Class 0 and 1 combined',
            fontsize=13, fontweight='bold', y=1.01
        )
        fig.savefig(f'{out_dir}/{analysis}_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
    print(f'  Saved {analysis} heatmap')


# ── Overall 2x2 scatter ────────────────────────────────────────────────────────

def plot_overall_scatter(mask, perm_data, delta_arrays, out_dir):
    """2x2 scatter: mean ALT, mean SM, change ALT, change SM vs overall delta."""
    print('\n[Overall Scatter] 2x2 permafrost vs delta suitability ...')

    overall = delta_arrays['OVERALL'][mask]

    panels = [
        ('ALT',           'mean',   'Mean Active Layer Depth (m)',     'Mean (1999-2018)'),
        ('soil_moisture', 'mean',   'Mean Available Soil Moisture',    'Mean (1999-2018)'),
        ('ALT',           'change', 'Active Layer Depth Change (m)',   'Change (post minus pre)'),
        ('soil_moisture', 'change', 'Soil Moisture Change',            'Change (post minus pre)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    vlim = float(np.nanpercentile(np.abs(overall[np.isfinite(overall)]), 98))
    vlim = max(vlim, 1e-4)

    for ax, (pvar, metric, xlabel, metric_label) in zip(axes, panels):
        perm_arr = perm_data[pvar]['mean_post'] if metric == 'mean' \
                   else perm_data[pvar]['change']
        x = perm_arr[mask]
        y = overall

        valid = np.isfinite(x) & np.isfinite(y)
        xv, yv = x[valid], y[valid]

        r, p = spearmanr(xv, yv)
        sig  = '*' if p < 0.05 else ''

        norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
        sc = ax.scatter(xv, yv, c=yv, cmap='RdBu', norm=norm,
                        alpha=0.5, s=12, zorder=3)
        plt.colorbar(sc, ax=ax, shrink=0.75, label='Mean delta Suitability')

        z = np.polyfit(xv, yv, 1)
        x_line = np.linspace(xv.min(), xv.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line),
                color='black', linewidth=1.5, linestyle='--',
                alpha=0.7, zorder=5)

        ax.axhline(0, color='grey', linewidth=0.8, linestyle=':')
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Mean delta Suitability (class units)', fontsize=10)
        ax.set_title(f'{metric_label}\nr = {r:.3f}, p = {p:.3f}{sig}',
                     fontsize=11, fontweight='bold')

    fig.suptitle(
        'Overall Mean delta Suitability vs Permafrost Variables\n'
        'Each dot = one pixel | Spearman r | * = p < 0.05 | Class 0 and 1 combined',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    fig.savefig(f'{out_dir}/overall_permafrost_scatter_2x2.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved overall 2x2 scatter')


# ── Permafrost space bin map ───────────────────────────────────────────────────

def plot_permafrost_space_bins(mask, perm_data, delta_arrays, out_dir, n_bins=6):
    """Binned ALT x soil moisture map colored by mean delta suitability.
    X-axis: mean active layer depth (1999-2018) binned
    Y-axis: mean available soil moisture (1999-2018) binned
    Color:  mean delta Suitability — blue = thaw helps, red = thaw hurts
    Numbers in each cell = pixel count.
    Per crop and overall.
    """
    print('\n[Permafrost Space] Binned ALT x soil moisture maps ...')
    out_dir_ps = f'{out_dir}/permafrost_space'
    os.makedirs(out_dir_ps, exist_ok=True)

    alt_arr = perm_data['ALT']['mean_post']
    sm_arr  = perm_data['soil_moisture']['mean_post']

    alt_flat = alt_arr[mask]
    sm_flat  = sm_arr[mask]

    alt_edges = np.unique(np.nanpercentile(alt_flat, np.linspace(0, 100, n_bins + 1)))
    sm_edges  = np.unique(np.nanpercentile(sm_flat,  np.linspace(0, 100, n_bins + 1)))
    n_alt = len(alt_edges) - 1
    n_sm  = len(sm_edges)  - 1

    alt_labels = [f'{alt_edges[i]:.2f}-{alt_edges[i+1]:.2f}'
                  for i in range(n_alt)]
    sm_labels  = [f'{sm_edges[i]:.3f}-{sm_edges[i+1]:.3f}'
                  for i in range(n_sm)]

    def make_bin_grid(delta_flat):
        grid  = np.full((n_sm, n_alt), np.nan)
        count = np.zeros((n_sm, n_alt), dtype=int)
        ai    = np.clip(np.digitize(alt_flat, alt_edges) - 1, 0, n_alt - 1)
        si    = np.clip(np.digitize(sm_flat,  sm_edges)  - 1, 0, n_sm  - 1)
        for a, s, dv in zip(ai, si, delta_flat):
            if np.isfinite(dv):
                grid[s, a]  = 0.0 if np.isnan(grid[s, a]) else grid[s, a]
                grid[s, a] += dv
                count[s, a] += 1
        valid = count > 0
        grid[valid]  /= count[valid]
        grid[~valid]  = np.nan
        return grid, count

    def draw_bin_map(ax, grid, count, title, vlim):
        cmap = plt.cm.RdBu.copy()
        cmap.set_bad(color='#e0e0e0')
        vlim = max(vlim, 1e-6)
        norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
        im = ax.imshow(grid, cmap=cmap, norm=norm, origin='lower', aspect='auto')
        for si in range(n_sm):
            for ai in range(n_alt):
                if count[si, ai] > 0:
                    txt_col = 'white' if (np.isfinite(grid[si, ai]) and
                              abs(grid[si, ai]) > vlim * 0.6) else 'black'
                    ax.text(ai, si, str(count[si, ai]),
                            ha='center', va='center', fontsize=7, color=txt_col)
        ax.set_xticks(range(n_alt))
        ax.set_xticklabels(alt_labels, rotation=30, ha='right', fontsize=8)
        ax.set_yticks(range(n_sm))
        ax.set_yticklabels(sm_labels, fontsize=8)
        ax.set_xlabel('Mean Active Layer Depth (m)', fontsize=9)
        ax.set_ylabel('Mean Available Soil Moisture', fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        return im

    # Shared vlim across all crops
    all_vals = []
    for delta in delta_arrays.values():
        g, _ = make_bin_grid(delta[mask])
        all_vals.extend(g[np.isfinite(g)].tolist())
    vlim = max(float(np.nanpercentile(np.abs(all_vals), 95)), 1e-4)

    # Per-crop panel
    crops_only = {k: v for k, v in delta_arrays.items() if k != 'OVERALL'}
    ncols = 4
    nrows = -(-len(crops_only) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5))
    axes = axes.flatten()

    for i, (label, delta) in enumerate(crops_only.items()):
        grid, count = make_bin_grid(delta[mask])
        im = draw_bin_map(axes[i], grid, count, label, vlim)

    for j in range(len(crops_only), len(axes)):
        axes[j].axis('off')

    fig.subplots_adjust(right=0.88, hspace=0.55, wspace=0.35)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm_cb = plt.cm.ScalarMappable(cmap='RdBu',
                                   norm=plt.Normalize(vmin=-vlim, vmax=vlim))
    sm_cb.set_array([])
    fig.colorbar(sm_cb, cax=cbar_ax, label='Mean delta Suitability (class units)')
    fig.suptitle(
        'Mean delta Suitability in ALT x Soil Moisture Space\n'
        'Blue = thaw helps, Red = thaw hurts | Numbers = pixel count per bin',
        fontsize=13, fontweight='bold'
    )
    fig.savefig(f'{out_dir_ps}/per_crop_permafrost_space.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved per-crop permafrost space map')

    # Overall standalone
    grid_ov, cnt_ov = make_bin_grid(delta_arrays['OVERALL'][mask])
    vlim_ov = max(float(np.nanpercentile(
        np.abs(grid_ov[np.isfinite(grid_ov)]), 95)), 1e-4)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    im2 = draw_bin_map(ax2, grid_ov, cnt_ov, '', vlim_ov)
    plt.colorbar(im2, ax=ax2,
                 label='Mean delta Suitability (class units)', shrink=0.8)
    ax2.set_title(
        'OVERALL — Mean delta Suitability\n'
        'in Active Layer Depth x Soil Moisture Space\n'
        'Blue = thaw helps, Red = thaw hurts | Numbers = pixel count',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    fig2.savefig(f'{out_dir_ps}/overall_permafrost_space.png',
                 dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved overall permafrost space map')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    mask = load_mask()

    perm_data    = load_permafrost_data(mask)
    delta_arrays = load_delta_arrays(mask)
    annual_delta = build_annual_delta_series(mask)

    results_spatial  = analysis_spatial(mask, perm_data, delta_arrays)
    results_temporal = analysis_temporal(mask, perm_data, annual_delta)

    df = pd.DataFrame(results_spatial + results_temporal)
    df.to_csv(f'{OUT_ROOT}/permafrost_correlation_results.csv', index=False)

    plot_summary_heatmap(df, 'spatial',  OUT_ROOT)
    plot_summary_heatmap(df, 'temporal', OUT_ROOT)
    plot_overall_scatter(mask, perm_data, delta_arrays, OUT_ROOT)
    plot_permafrost_space_bins(mask, perm_data, delta_arrays, OUT_ROOT)

    print('\nSpatial correlation — significant only:')
    df_s = df[(df['analysis'] == 'spatial') & (df['significant'])]
    if df_s.empty:
        print('  None significant')
    else:
        print(df_s[['crop', 'perm_var', 'metric',
                    'spearman_r', 'p_value']].to_string(index=False))

    print('\nTemporal correlation — significant only:')
    df_t = df[(df['analysis'] == 'temporal') & (df['significant'])]
    if df_t.empty:
        print('  None significant')
    else:
        print(df_t[['crop', 'perm_var', 'spearman_r', 'p_value']].to_string(index=False))

    print(f'\nAll outputs saved to: {OUT_ROOT}/')