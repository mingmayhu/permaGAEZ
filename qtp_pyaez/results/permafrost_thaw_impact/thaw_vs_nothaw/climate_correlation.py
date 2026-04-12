"""
Climate Variable Correlation with Delta Suitability
=====================================================
Correlates climate variables with mean delta suitability (Thaw - No-Thaw).

Class 0 and class 1 are combined into class 1 (remap) before all calculations.

Climate variables:
  - Annual total precipitation (sum across days)
  - Mean daily max temperature (mean across days)
  - Mean daily min temperature (mean across days)
  - Mean daily mean temperature (mean across days)

For each variable computes:
  A) Mean value across 1999-2018
  B) Change = mean(1999-2018) minus mean(1979-1998)

Analyses:
  A) Pixel-wise spatial correlation (Spearman) between climate variable
     and mean delta Suitability across pixels
  B) Temporal correlation (Spearman) between annual regional mean climate
     variable and annual regional mean delta Suitability (1999-2018)

Outputs written to:
  ./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/6_climate_corr/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from osgeo import gdal

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR  = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH = r'./data_input/qilian mask.tif'
CLIM_DIR  = r'./data_input/climate_yearly'
DELTA_DIR = r'./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/5_spatial/1_mean_delta'
OUT_ROOT  = r'./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/6_climate_corr'

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

CLIM_VARS = [
    {'name': 'precip',    'file': 'Precip.npy',   'label': 'Annual Total Precipitation (mm)', 'agg': 'sum'},
    {'name': 'temp_max',  'file': 'TempMax.npy',   'label': 'Mean Daily Max Temperature (C)',  'agg': 'mean'},
    {'name': 'temp_min',  'file': 'TempMin.npy',   'label': 'Mean Daily Min Temperature (C)',  'agg': 'mean'},
    {'name': 'temp_mean', 'file': 'TempMean.npy',  'label': 'Mean Daily Temperature (C)',      'agg': 'mean'},
]

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
    """Load pre-computed mean delta raster from spatial analysis output."""
    path = f'{DELTA_DIR}/{tag}_mean_delta_suit.tif'
    arr, _ = load_raster(path)
    if arr is None:
        return None
    arr[~mask] = np.nan
    arr[arr < -1e10] = np.nan
    return arr

def load_clim_stack(var_file, years, mask, agg='mean'):
    """Load climate variable for a list of years.
    Arrays are (rows, cols, days) — aggregate across days.
    Returns (n_years, rows, cols).
    """
    stack        = []
    target_shape = mask.shape

    for year in years:
        path = f'{CLIM_DIR}/{year}/{var_file}'
        if not os.path.exists(path):
            print(f'  Warning: missing {path}')
            stack.append(np.full(target_shape, np.nan))
            continue

        arr = np.load(path).astype(float)
        if arr.ndim == 3:
            arr = np.nansum(arr, axis=2) if agg == 'sum' else np.nanmean(arr, axis=2)

        if arr.shape != target_shape:
            print(f'  Warning: shape mismatch year {year}: {arr.shape} vs {target_shape}')
            stack.append(np.full(target_shape, np.nan))
            continue

        arr[~mask] = np.nan
        stack.append(arr)

    return np.array(stack)

def regional_mean(arr, mask):
    valid = mask & np.isfinite(arr)
    return float(np.nanmean(arr[valid])) if valid.any() else np.nan


# ── Load data ─────────────────────────────────────────────────────────────────

def load_climate_data(mask):
    print('Loading climate data ...')
    clim_data = {}
    for cv in CLIM_VARS:
        stack_all  = load_clim_stack(cv['file'], YEARS_ALL, mask, agg=cv['agg'])
        stack_pre  = stack_all[:len(YEARS_PRE)]
        stack_post = stack_all[len(YEARS_PRE):]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mean_post = np.nanmean(stack_post, axis=0)
            mean_pre  = np.nanmean(stack_pre,  axis=0)

        change = mean_post - mean_pre
        mean_post[~mask] = np.nan
        change[~mask]    = np.nan

        clim_data[cv['name']] = {
            'stack_post': stack_post,
            'mean_post' : mean_post,
            'change'    : change,
            'label'     : cv['label'],
        }
        print(f'  {cv["name"]}: mean [{np.nanmin(mean_post):.2f}, {np.nanmax(mean_post):.2f}]'
              f'  change [{np.nanmin(change):.3f}, {np.nanmax(change):.3f}]')

    # Compute temp_mean as (temp_max + temp_min) / 2
    # TempMean.npy is zeroed out so we derive it from max and min
    if 'temp_max' in clim_data and 'temp_min' in clim_data:
        print('  Computing temp_mean from (temp_max + temp_min) / 2 ...')
        stack_post_mean = (clim_data['temp_max']['stack_post'] +
                           clim_data['temp_min']['stack_post']) / 2
        mean_post_mean  = (clim_data['temp_max']['mean_post'] +
                           clim_data['temp_min']['mean_post']) / 2
        change_mean     = (clim_data['temp_max']['change'] +
                           clim_data['temp_min']['change']) / 2
        mean_post_mean[~mask] = np.nan
        change_mean[~mask]    = np.nan

        clim_data['temp_mean'] = {
            'stack_post': stack_post_mean,
            'mean_post' : mean_post_mean,
            'change'    : change_mean,
            'label'     : 'Mean Daily Temperature (C) [derived]',
        }
        print(f'  temp_mean (derived): mean [{np.nanmin(mean_post_mean):.2f}, '
              f'{np.nanmax(mean_post_mean):.2f}]  '
              f'change [{np.nanmin(change_mean):.3f}, {np.nanmax(change_mean):.3f}]')

    return clim_data


def load_delta_arrays(mask):
    """Load pre-computed mean delta rasters. Falls back to recomputing if missing."""
    print('\nLoading mean delta suitability rasters ...')
    delta_arrays = {}
    for crop in CROPS:
        delta = load_mean_delta(crop['tag'], mask)
        if delta is not None:
            delta_arrays[crop['label']] = delta
        else:
            print(f'  Warning: {crop["label"]} delta raster not found, recomputing ...')
            stack = []
            for year in YEARS_CF:
                obs, _ = load_raster(obs_suit_path(crop['tag'], year))
                cf,  _ = load_raster(cf_suit_path(crop['tag'], year))
                if obs is None or cf is None:
                    continue
                obs_r = apply_remap(obs, mask)
                cf_r  = apply_remap(cf,  mask)
                d = np.where(np.isfinite(obs_r) & np.isfinite(cf_r),
                             obs_r - cf_r, np.nan)
                d[~mask] = np.nan
                stack.append(d)
            if stack:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    delta_arrays[crop['label']] = np.nanmean(np.stack(stack), axis=0)

    # Overall aggregate
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        overall = np.nanmean(np.stack(list(delta_arrays.values())), axis=0)
    overall[~mask] = np.nan
    delta_arrays['OVERALL'] = overall

    print(f'  Loaded {len(delta_arrays)-1} crops + OVERALL')
    return delta_arrays


def build_annual_delta_series(mask):
    """Build annual regional mean delta suitability series per crop (1999-2018).
    Uses remap so class 0 and 1 are combined.
    """
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

    # Overall — average across all crops
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        overall = np.nanmean(np.stack(list(annual_delta.values())), axis=0)
    annual_delta['OVERALL'] = overall

    print(f'  Built series for {len(annual_delta)-1} crops + OVERALL')
    return annual_delta


# ── Analysis A: Pixel-wise spatial correlation ─────────────────────────────────

def analysis_spatial(mask, clim_data, delta_arrays):
    print('\n[Analysis A] Pixel-wise spatial correlation ...')
    out_dir     = f'{OUT_ROOT}/A_spatial'
    all_results = []

    for cv in CLIM_VARS:
        cname  = cv['name']
        clabel = clim_data[cname]['label']

        for metric, clim_arr, metric_label in [
            ('mean',   clim_data[cname]['mean_post'],
             f'Mean {clabel} (1999-2018)'),
            ('change', clim_data[cname]['change'],
             f'Delta {clabel} (1999-2018 minus 1979-1998)'),
        ]:
            ncols = 4
            nrows = -(-len(delta_arrays) // ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
            axes = axes.flatten()

            for i, (crop_label, delta) in enumerate(delta_arrays.items()):
                ax = axes[i]
                valid = mask & np.isfinite(delta) & np.isfinite(clim_arr)
                x = clim_arr[valid]
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
                    'clim_var'   : cname,
                    'metric'     : metric,
                    'spearman_r' : round(r, 4),
                    'p_value'    : round(p, 4),
                    'significant': p < 0.05,
                    'n_pixels'   : int(valid.sum()),
                })

            for j in range(len(delta_arrays), len(axes)):
                axes[j].axis('off')

            fig.suptitle(
                f'Pixel-wise Spatial Correlation: {metric_label}\nvs Mean delta Suitability'
                f' | * = p < 0.05 | Class 0 and 1 combined',
                fontsize=12, fontweight='bold'
            )
            plt.tight_layout()
            fig.savefig(f'{out_dir}/{cname}_{metric}_spatial_corr.png',
                        dpi=150, bbox_inches='tight')
            plt.close()
            print(f'  Saved {cname} {metric} spatial correlation')

    return all_results


# ── Analysis B: Temporal correlation ──────────────────────────────────────────

def analysis_temporal(mask, clim_data, annual_delta):
    print('\n[Analysis B] Temporal correlation ...')
    out_dir     = f'{OUT_ROOT}/B_temporal'
    years_arr   = np.array(YEARS_CF)
    all_results = []

    for cv in CLIM_VARS:
        cname  = cv['name']
        clabel = clim_data[cname]['label']
        stack  = clim_data[cname]['stack_post']

        # Annual regional mean climate variable
        annual_clim = np.array([
            regional_mean(stack[i], mask) for i in range(stack.shape[0])
        ])

        ncols = 4
        nrows = -(-len(annual_delta) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
        axes = axes.flatten()

        for i, (crop_label, delta_series) in enumerate(annual_delta.items()):
            ax = axes[i]
            valid = np.isfinite(annual_clim) & np.isfinite(delta_series)

            if valid.sum() < 4:
                ax.set_title(f'{crop_label}\n(insufficient data)', fontsize=9)
                ax.axis('off')
                continue

            x = annual_clim[valid]
            y = delta_series[valid]
            r, p = spearmanr(x, y)
            sig  = '*' if p < 0.05 else ''

            sc = ax.scatter(x, y, c=years_arr[valid], cmap='viridis',
                            s=50, zorder=3)
            plt.colorbar(sc, ax=ax, shrink=0.7, label='Year')
            ax.axhline(0, color='black', linewidth=0.6, linestyle='--')
            ax.set_xlabel(f'Regional Mean {clabel}', fontsize=8)
            ax.set_ylabel('Regional Mean delta Suitability', fontsize=8)
            ax.set_title(f'{crop_label}\nr={r:.3f}, p={p:.3f}{sig}',
                         fontsize=9, fontweight='bold')

            all_results.append({
                'analysis'   : 'temporal',
                'crop'       : crop_label,
                'clim_var'   : cname,
                'metric'     : 'annual_mean',
                'spearman_r' : round(r, 4),
                'p_value'    : round(p, 4),
                'significant': p < 0.05,
                'n_years'    : int(valid.sum()),
            })

        for j in range(len(annual_delta), len(axes)):
            axes[j].axis('off')

        fig.suptitle(
            f'Temporal Correlation: Annual Regional Mean {clabel}\n'
            f'vs Regional Mean delta Suitability (1999-2018)'
            f' | * = p < 0.05 | Class 0 and 1 combined',
            fontsize=12, fontweight='bold'
        )
        plt.tight_layout()
        fig.savefig(f'{out_dir}/{cname}_temporal_corr.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved {cname} temporal correlation')

    return all_results


# ── Summary heatmap ────────────────────────────────────────────────────────────

# Readable labels for climate variables and metrics
CLIM_VAR_LABELS = {
    'precip'   : 'Precipitation',
    'temp_max' : 'Max Temp',
    'temp_min' : 'Min Temp',
    'temp_mean': 'Mean Temp',
}
METRIC_LABELS = {
    'mean'  : 'Mean (1999-2018)',
    'change': 'Change (post minus pre)',
}

def build_matrix(df_subset, crops, clim_vars):
    """Build r and p matrices from a filtered dataframe."""
    r_mat = pd.DataFrame(index=crops, columns=clim_vars, dtype=float)
    p_mat = pd.DataFrame(index=crops, columns=clim_vars, dtype=float)
    for _, row in df_subset.iterrows():
        if row['crop'] in crops and row['clim_var'] in clim_vars:
            r_mat.loc[row['crop'], row['clim_var']] = row['spearman_r']
            p_mat.loc[row['crop'], row['clim_var']] = row['p_value']
    return r_mat, p_mat

def draw_heatmap(ax, r_mat, p_mat, crops, clim_vars, title, vmin=-0.3, vmax=0.3):
    """Draw a single annotated heatmap panel."""
    r_vals = r_mat.values.astype(float)
    im = ax.imshow(r_vals, cmap='RdBu', vmin=vmin, vmax=vmax, aspect='auto')

    # Column labels
    ax.set_xticks(range(len(clim_vars)))
    ax.set_xticklabels([CLIM_VAR_LABELS.get(v, v) for v in clim_vars],
                       rotation=30, ha='right', fontsize=10)
    # Row labels — only on leftmost panel
    ax.set_yticks(range(len(crops)))
    ax.set_yticklabels(crops, fontsize=10)

    # Add horizontal line separating OVERALL from crops
    if 'OVERALL' in crops:
        overall_idx = list(crops).index('OVERALL')
        ax.axhline(overall_idx - 0.5, color='black', linewidth=1.5)

    # Annotate cells
    for i, crop in enumerate(crops):
        for j, cv in enumerate(clim_vars):
            r_val = r_mat.loc[crop, cv]
            p_val = p_mat.loc[crop, cv]
            if pd.notna(r_val):
                sig = '*' if pd.notna(p_val) and p_val < 0.05 else ''
                text_color = 'white' if abs(r_val) > abs(vmin) * 0.7 else 'black'
                ax.text(j, i, f'{r_val:.2f}{sig}',
                        ha='center', va='center', fontsize=9,
                        color=text_color, fontweight='bold' if sig else 'normal')

    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    return im


def plot_summary_heatmap(df, analysis, out_dir):
    """Combined heatmap: mean and change side by side for spatial analysis."""
    df_a = df[df['analysis'] == analysis].copy()
    if df_a.empty:
        return

    crops     = list(df_a['crop'].unique())
    clim_vars = list(df_a['clim_var'].unique())

    # Put OVERALL at bottom
    if 'OVERALL' in crops:
        crops = [c for c in crops if c != 'OVERALL'] + ['OVERALL']

    metrics = [m for m in ['mean', 'change', 'annual_mean'] if m in df_a['metric'].unique()]

    if len(metrics) == 1:
        # Temporal has only one metric — single panel
        fig, ax = plt.subplots(figsize=(len(clim_vars) * 2.2 + 1.5, len(crops) * 0.55 + 2))
        df_m        = df_a[df_a['metric'] == metrics[0]]
        r_mat, p_mat = build_matrix(df_m, crops, clim_vars)
        im = draw_heatmap(ax, r_mat, p_mat, crops, clim_vars,
                          METRIC_LABELS.get(metrics[0], metrics[0]))
        plt.colorbar(im, ax=ax, label='Spearman r', shrink=0.8)
        fig.suptitle(
            f'Climate Correlation with delta Suitability ({analysis.capitalize()})\n'
            f'Spearman r | * = p < 0.05 | Class 0 and 1 combined',
            fontsize=12, fontweight='bold'
        )
        plt.tight_layout()
        fig.savefig(f'{out_dir}/{analysis}_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved {analysis} heatmap')

    else:
        # Spatial has mean + change — side-by-side panels
        fig, axes = plt.subplots(1, 2, figsize=(len(clim_vars) * 4.5, len(crops) * 0.55 + 3))

        for ax, metric in zip(axes, ['mean', 'change']):
            df_m         = df_a[df_a['metric'] == metric]
            r_mat, p_mat = build_matrix(df_m, crops, clim_vars)
            im = draw_heatmap(ax, r_mat, p_mat, crops, clim_vars,
                              METRIC_LABELS.get(metric, metric))
            # Only show y labels on left panel
            if ax != axes[0]:
                ax.set_yticklabels([])

        # Shared colorbar
        fig.subplots_adjust(right=0.88, wspace=0.05)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap='RdBu',
                                   norm=plt.Normalize(vmin=-0.3, vmax=0.3))
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label='Spearman r')

        fig.suptitle(
            'Spatial Correlation: Climate Variables vs Mean delta Suitability\n'
            'Spearman r | * = p < 0.05 | Class 0 and 1 combined',
            fontsize=13, fontweight='bold', y=1.01
        )
        fig.savefig(f'{out_dir}/{analysis}_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved {analysis} combined heatmap (mean + change)')


# ── Climate space bin map ──────────────────────────────────────────────────────

def plot_climate_space_bins(mask, clim_data, delta_arrays, out_dir, n_bins=6):
    """Binned temperature x precipitation map colored by mean delta suitability.
    X-axis: mean temperature (1999-2018) binned
    Y-axis: mean precipitation (1999-2018) binned
    Color:  mean delta Suitability — blue = thaw helps, red = thaw hurts
    Numbers in each cell = pixel count.
    """
    print("\n[Climate Space] Binned temp x precip maps ...")
    out_dir_cs = f"{out_dir}/climate_space"
    os.makedirs(out_dir_cs, exist_ok=True)

    from matplotlib.colors import TwoSlopeNorm

    temp_arr   = clim_data["temp_mean"]["mean_post"]
    precip_arr = clim_data["precip"]["mean_post"]

    temp_flat   = temp_arr[mask]
    precip_flat = precip_arr[mask]

    temp_edges   = np.unique(np.nanpercentile(temp_flat,   np.linspace(0, 100, n_bins + 1)))
    precip_edges = np.unique(np.nanpercentile(precip_flat, np.linspace(0, 100, n_bins + 1)))
    n_temp   = len(temp_edges) - 1
    n_precip = len(precip_edges) - 1

    temp_labels   = [f"{temp_edges[i]:.1f}-{temp_edges[i+1]:.1f}" for i in range(n_temp)]
    precip_labels = [f"{precip_edges[i]:.0f}-{precip_edges[i+1]:.0f}" for i in range(n_precip)]

    def make_bin_grid(delta_flat):
        grid  = np.full((n_precip, n_temp), np.nan)
        count = np.zeros((n_precip, n_temp), dtype=int)
        ti_arr = np.clip(np.digitize(temp_flat,   temp_edges)   - 1, 0, n_temp   - 1)
        pi_arr = np.clip(np.digitize(precip_flat, precip_edges) - 1, 0, n_precip - 1)
        for ti, pi, dv in zip(ti_arr, pi_arr, delta_flat):
            if np.isfinite(dv):
                grid[pi, ti]  = 0.0 if np.isnan(grid[pi, ti]) else grid[pi, ti]
                grid[pi, ti] += dv
                count[pi, ti] += 1
        valid = count > 0
        grid[valid] /= count[valid]
        grid[~valid] = np.nan
        return grid, count

    def draw_bin_map(ax, grid, count, title, vlim):
        cmap = plt.cm.RdBu.copy()
        cmap.set_bad(color="#e0e0e0")
        vlim = max(vlim, 1e-6)
        norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
        im = ax.imshow(grid, cmap=cmap, norm=norm, origin="lower", aspect="auto")
        for pi in range(n_precip):
            for ti in range(n_temp):
                if count[pi, ti] > 0:
                    txt_col = "white" if (np.isfinite(grid[pi, ti]) and
                              abs(grid[pi, ti]) > vlim * 0.6) else "black"
                    ax.text(ti, pi, str(count[pi, ti]),
                            ha="center", va="center", fontsize=7, color=txt_col)
        ax.set_xticks(range(n_temp))
        ax.set_xticklabels(temp_labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(n_precip))
        ax.set_yticklabels(precip_labels, fontsize=8)
        ax.set_xlabel("Mean Temperature (C)", fontsize=9)
        ax.set_ylabel("Mean Precipitation (mm)", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        return im

    # Shared vlim across all crops
    all_vals = []
    for delta in delta_arrays.values():
        g, _ = make_bin_grid(delta[mask])
        all_vals.extend(g[np.isfinite(g)].tolist())
    vlim = max(float(np.nanpercentile(np.abs(all_vals), 95)), 1e-4)

    # Per-crop panel
    crops_only = {k: v for k, v in delta_arrays.items() if k != "OVERALL"}
    ncols = 4
    nrows = -(-len(crops_only) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5))
    axes = axes.flatten()

    for i, (label, delta) in enumerate(crops_only.items()):
        grid, count = make_bin_grid(delta[mask])
        im = draw_bin_map(axes[i], grid, count, label, vlim)

    for j in range(len(crops_only), len(axes)):
        axes[j].axis("off")

    fig.subplots_adjust(right=0.88, hspace=0.55, wspace=0.35)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap="RdBu", norm=plt.Normalize(vmin=-vlim, vmax=vlim))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Mean delta Suitability (class units)")
    fig.suptitle(
        "Mean delta Suitability in Temperature x Precipitation Space\n"
        "Blue = thaw helps, Red = thaw hurts | Numbers = pixel count per bin",
        fontsize=13, fontweight="bold"
    )
    fig.savefig(f"{out_dir_cs}/per_crop_climate_space.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved per-crop climate space map")

    # Overall standalone
    grid_ov, cnt_ov = make_bin_grid(delta_arrays["OVERALL"][mask])
    vlim_ov = max(float(np.nanpercentile(np.abs(grid_ov[np.isfinite(grid_ov)]), 95)), 1e-4)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    im2 = draw_bin_map(ax2, grid_ov, cnt_ov, "", vlim_ov)
    plt.colorbar(im2, ax=ax2, label="Mean delta Suitability (class units)", shrink=0.8)
    ax2.set_title(
        "OVERALL — Mean delta Suitability\nin Temperature x Precipitation Space\n"
        "Blue = thaw helps, Red = thaw hurts | Numbers = pixel count",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    fig2.savefig(f"{out_dir_cs}/overall_climate_space.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved overall climate space map")



# ── Overall 2x2 scatter plot ───────────────────────────────────────────────────

def plot_overall_scatter(mask, clim_data, delta_arrays, out_dir):
    """2x2 scatter plot of overall mean delta suitability vs
    mean temp, mean precip, change in temp, change in precip.
    One dot per pixel, colored by delta suitability value.
    """
    from scipy.stats import spearmanr
    from matplotlib.colors import TwoSlopeNorm
    print("\n[Overall Scatter] 2x2 climate vs delta suitability ...")

    overall = delta_arrays["OVERALL"][mask]

    panels = [
        ("temp_mean",  "mean",   "Mean Temperature (C)",           "Mean (1999-2018)"),
        ("precip",     "mean",   "Mean Precipitation (mm)",        "Mean (1999-2018)"),
        ("temp_mean",  "change", "Temperature Change (C)",         "Change (post minus pre)"),
        ("precip",     "change", "Precipitation Change (mm)",      "Change (post minus pre)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Shared color scale for dots
    vlim = float(np.nanpercentile(np.abs(overall[np.isfinite(overall)]), 98))
    vlim = max(vlim, 1e-4)

    for ax, (cvar, metric, xlabel, metric_label) in zip(axes, panels):
        clim_arr = clim_data[cvar]["mean_post"] if metric == "mean"                    else clim_data[cvar]["change"]
        x = clim_arr[mask]
        y = overall

        valid = np.isfinite(x) & np.isfinite(y)
        xv, yv = x[valid], y[valid]

        r, p = spearmanr(xv, yv)
        sig  = "*" if p < 0.05 else ""

        norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
        sc = ax.scatter(xv, yv, c=yv, cmap="RdBu", norm=norm,
                        alpha=0.5, s=12, zorder=3)
        plt.colorbar(sc, ax=ax, shrink=0.75,
                     label="Mean delta Suitability")

        # Trend line
        z = np.polyfit(xv, yv, 1)
        x_line = np.linspace(xv.min(), xv.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line),
                color="black", linewidth=1.5, linestyle="--", alpha=0.7, zorder=5)

        ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Mean delta Suitability (class units)", fontsize=10)
        ax.set_title(f"{metric_label}\nr = {r:.3f}, p = {p:.3f}{sig}",
                     fontsize=11, fontweight="bold")

    fig.suptitle(
        "Overall Mean delta Suitability vs Climate Variables\n"
        "Each dot = one pixel | Spearman r | * = p < 0.05 | Class 0 and 1 combined",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(f"{out_dir}/overall_climate_scatter_2x2.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved overall 2x2 scatter")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mask = load_mask()

    clim_data    = load_climate_data(mask)
    delta_arrays = load_delta_arrays(mask)
    annual_delta = build_annual_delta_series(mask)

    results_spatial  = analysis_spatial(mask, clim_data, delta_arrays)
    results_temporal = analysis_temporal(mask, clim_data, annual_delta)

    # Save all results
    df = pd.DataFrame(results_spatial + results_temporal)
    df.to_csv(f"{OUT_ROOT}/climate_correlation_results.csv", index=False)

    # Summary heatmaps
    plot_summary_heatmap(df, "spatial",  OUT_ROOT)
    plot_summary_heatmap(df, "temporal", OUT_ROOT)

    # Climate space bin maps
    plot_climate_space_bins(mask, clim_data, delta_arrays, OUT_ROOT)

    # Overall 2x2 scatter
    plot_overall_scatter(mask, clim_data, delta_arrays, OUT_ROOT)

    # Print significant results
    print("\nSpatial correlation — significant only:")
    df_s = df[(df["analysis"] == "spatial") & (df["significant"])]
    if df_s.empty:
        print("  None significant")
    else:
        print(df_s[["crop", "clim_var", "metric",
                    "spearman_r", "p_value"]].to_string(index=False))

    print("\nTemporal correlation — significant only:")
    df_t = df[(df["analysis"] == "temporal") & (df["significant"])]
    if df_t.empty:
        print("  None significant")
    else:
        print(df_t[["crop", "clim_var", "spearman_r", "p_value"]].to_string(index=False))

    print(f"\nAll outputs saved to: {OUT_ROOT}/")