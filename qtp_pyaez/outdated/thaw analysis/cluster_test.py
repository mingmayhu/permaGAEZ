"""
ΔYield Class Characteristics Analysis
======================================
Compares terrain and soil moisture characteristics of pixels
where permafrost thaw helps, has no effect, or hurts yield.

Classes (pooled across all 10 crops):
  Positive  : ΔYield > +1 kg/ha
  Near-zero : -1 <= ΔYield <= +1 kg/ha
  Negative  : ΔYield < -1 kg/ha

Predictors compared:
  - Elevation
  - Slope
  - ΔALT  (active layer deepening vs baseline)
  - Baseline ASM (1979-1998 mean)
  - ΔASM  (1999-2018 mean minus 1979-1998 mean)

Outputs: ./thaw_analysis_output/14_class_characteristics/
"""

# =============================================================================
# CONFIGURATION
# =============================================================================

WORK_DIR   = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH  = r'./data_input/qilian mask.tif'
ELEV_PATH  = r'./data_input/terrain/elevation.npy'
SLOPE_PATH = r'./data_input/terrain/slope.tif'

ALT_PATH_PATTERN = r'./data_input/permafrost_yearly/{year}/active_layer_depth.npy'
ASM_PATH_PATTERN = r'./data_input/permafrost_yearly/{year}/avail_soil_moisture.npy'

YEARS_BASELINE   = list(range(1979, 1999))
YEARS_COMPARISON = list(range(1999, 2019))

THRESHOLD = 1.0   # kg/ha

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

OUT_DIR = './thaw_analysis_output/14_class_characteristics'

# =============================================================================
# IMPORTS
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy import stats
from scipy.ndimage import zoom

try:
    from osgeo import gdal
except ImportError:
    import gdal

os.chdir(WORK_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# HELPERS
# =============================================================================

def load_raster(path):
    ds = gdal.Open(path)
    if ds is None:
        return None
    band   = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    arr    = ds.ReadAsArray().astype(float)
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[arr < -1e10] = np.nan
    return arr

def load_mask():
    return load_raster(MASK_PATH).astype(bool)

def match_grid(arr, target_shape):
    if arr.shape == target_shape:
        return arr
    zy = target_shape[0] / arr.shape[0]
    zx = target_shape[1] / arr.shape[1]
    return zoom(arr, (zy, zx), order=1)

def load_mean_dyield(tag, mask):
    obs_stack, cf_stack = [], []
    for year in YEARS_COMPARISON:
        obs = load_raster(
            f'./data_output/final_classification/{tag}/{year}_raw_yield.tif')
        cf  = load_raster(
            f'./data_output/final_classification_nothaw/{tag}/{year}_raw_yield.tif')
        if obs is not None:
            obs[~mask] = np.nan
            obs_stack.append(obs)
        if cf is not None:
            cf[~mask] = np.nan
            cf_stack.append(cf)
    if not obs_stack or not cf_stack:
        return None
    diff = np.nanmean(obs_stack, axis=0) - np.nanmean(cf_stack, axis=0)
    diff[~mask] = np.nan
    return diff

def load_delta_alt(mask):
    def _period(years):
        stack = []
        for year in years:
            try:
                arr = np.load(ALT_PATH_PATTERN.format(year=year)).astype(float)
            except FileNotFoundError:
                continue
            if arr.ndim == 3:
                ax_ = int(np.argmax(arr.shape))
                arr = np.moveaxis(arr, ax_, 0)
                arr = np.nanmean(arr, axis=0)
            arr = match_grid(arr, mask.shape)
            arr[~mask] = np.nan
            arr[arr <= 0] = np.nan
            stack.append(arr)
        return np.nanmean(stack, axis=0) if stack else None

    base = _period(YEARS_BASELINE)
    comp = _period(YEARS_COMPARISON)
    if base is None or comp is None:
        return None
    delta = np.where(np.isfinite(base) & np.isfinite(comp),
                     comp - base, np.nan)
    delta[~mask] = np.nan
    return delta

def load_asm(mask):
    def _period(years):
        stack = []
        for year in years:
            try:
                arr = np.load(ASM_PATH_PATTERN.format(year=year)).astype(float)
            except FileNotFoundError:
                continue
            if arr.ndim == 3:
                arr = np.nanmean(arr, axis=2)
            arr = match_grid(arr, mask.shape)
            arr[~mask] = np.nan
            arr[arr < 0] = np.nan
            stack.append(arr)
        return np.nanmean(stack, axis=0) if stack else None

    base = _period(YEARS_BASELINE)
    comp = _period(YEARS_COMPARISON)
    if base is None or comp is None:
        return None, None
    delta = np.where(np.isfinite(base) & np.isfinite(comp),
                     comp - base, np.nan)
    delta[~mask] = np.nan
    return base, delta

# =============================================================================
# BUILD POOLED DATASET
# =============================================================================

def build_dataset(mask, predictors):
    """
    Pool all crops into one dataframe.
    Each row = one pixel-crop combination with its class and predictor values.
    """
    print('\nBuilding pooled dataset...')
    rows = []

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        dy = load_mean_dyield(tag, mask)
        if dy is None:
            print(f'  Skipping {label} — no data')
            continue

        # Valid where dy and all predictors are finite
        valid = mask & np.isfinite(dy)
        for arr in predictors.values():
            valid &= np.isfinite(arr)

        dy_vals = dy[valid]

        # Assign class
        cls = np.where(dy_vals >  THRESHOLD, 'Positive',
              np.where(dy_vals < -THRESHOLD, 'Negative', 'Near-zero'))

        for j in range(valid.sum()):
            row = {'crop': label, 'dy': dy_vals[j], 'class': cls[j]}
            for pname, parr in predictors.items():
                row[pname] = parr[valid][j]
            rows.append(row)

        n_pos  = int((cls == 'Positive').sum())
        n_neg  = int((cls == 'Negative').sum())
        n_zero = int((cls == 'Near-zero').sum())
        print(f'  {label}: pos={n_pos}  zero={n_zero}  neg={n_neg}')

    df = pd.DataFrame(rows)
    print(f'\nTotal pooled samples: {len(df)}')
    print(df['class'].value_counts().to_string())
    df.to_csv(f'{OUT_DIR}/pooled_dataset.csv', index=False)
    return df

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def compute_summary(df, feat_names):
    """
    Per class: mean, median, std for each predictor.
    Plus pairwise Mann-Whitney U tests between all class pairs.
    """
    print('\nComputing summary statistics...')
    classes    = ['Negative', 'Near-zero', 'Positive']
    class_pairs = [('Negative', 'Near-zero'),
                   ('Negative', 'Positive'),
                   ('Near-zero', 'Positive')]

    # Descriptive stats
    stat_rows = []
    for cls in classes:
        sub = df[df['class'] == cls]
        for feat in feat_names:
            stat_rows.append({
                'class' : cls,
                'feature': feat,
                'n'     : len(sub),
                'mean'  : round(sub[feat].mean(), 4),
                'median': round(sub[feat].median(), 4),
                'std'   : round(sub[feat].std(), 4),
            })
    stats_df = pd.DataFrame(stat_rows)
    stats_df.to_csv(f'{OUT_DIR}/descriptive_stats.csv', index=False)

    # Pairwise Mann-Whitney U
    mw_rows = []
    for feat in feat_names:
        for c1, c2 in class_pairs:
            v1 = df[df['class'] == c1][feat].dropna().values
            v2 = df[df['class'] == c2][feat].dropna().values
            if len(v1) < 3 or len(v2) < 3:
                continue
            stat, p = stats.mannwhitneyu(v1, v2, alternative='two-sided')
            # Effect size: rank-biserial correlation
            n1, n2 = len(v1), len(v2)
            r_rb   = 1 - (2 * stat) / (n1 * n2)
            mw_rows.append({
                'feature'      : feat,
                'class_1'      : c1,
                'class_2'      : c2,
                'mannwhitney_p': round(p, 6),
                'effect_size_r': round(r_rb, 4),
                'significant'  : p < 0.05,
            })

    mw_df = pd.DataFrame(mw_rows)
    mw_df.to_csv(f'{OUT_DIR}/mannwhitney_tests.csv', index=False)

    print('\nMann-Whitney U results:')
    print(mw_df.to_string(index=False))
    return stats_df, mw_df

# =============================================================================
# PLOT 1 — Boxplots: all predictors side by side, coloured by class
# =============================================================================

def plot_boxplots(df, feat_names, mw_df):
    print('\nPlotting boxplots...')

    CLASS_ORDER  = ['Negative', 'Near-zero', 'Positive']
    CLASS_COLORS = {'Negative': '#d73027', 'Near-zero': '#fee090',
                    'Positive': '#4575b4'}

    fig, axes = plt.subplots(1, len(feat_names),
                              figsize=(4.5 * len(feat_names), 6))
    fig.suptitle(f'Predictor Characteristics by ΔYield Class\n'
                 f'(Pooled across all crops, threshold = ±{THRESHOLD} kg/ha)',
                 fontsize=13, fontweight='bold')

    for ax, feat in zip(axes, feat_names):
        groups = [df[df['class'] == c][feat].dropna().values
                  for c in CLASS_ORDER]
        bp = ax.boxplot(groups,
                        patch_artist=True,
                        labels=CLASS_ORDER,
                        medianprops=dict(color='black', lw=2),
                        flierprops=dict(marker='o', markersize=2,
                                        alpha=0.3, linestyle='none'))
        for patch, cls in zip(bp['boxes'], CLASS_ORDER):
            patch.set_facecolor(CLASS_COLORS[cls])
            patch.set_alpha(0.75)

        ax.set_title(feat, fontsize=11, fontweight='bold')
        ax.set_ylabel(feat, fontsize=9)
        ax.tick_params(axis='x', labelsize=9)

        # Annotate significant pairwise differences
        # Get y position for significance bars
        y_max = max(np.nanpercentile(g, 95) for g in groups if len(g) > 0)
        y_range = y_max - min(np.nanpercentile(g, 5)
                              for g in groups if len(g) > 0)
        sig_pairs = mw_df[(mw_df['feature'] == feat) &
                          (mw_df['significant'])]

        pair_positions = {
            ('Negative', 'Near-zero'): (1, 2, y_max + y_range * 0.05),
            ('Negative', 'Positive') : (1, 3, y_max + y_range * 0.15),
            ('Near-zero', 'Positive'): (2, 3, y_max + y_range * 0.25),
        }
        for _, row in sig_pairs.iterrows():
            pair = (row['class_1'], row['class_2'])
            if pair in pair_positions:
                x1, x2, y = pair_positions[pair]
                ax.plot([x1, x2], [y, y], color='black', lw=1)
                ax.text((x1 + x2) / 2, y, '*', ha='center',
                        va='bottom', fontsize=12)

    legend_handles = [Patch(facecolor=CLASS_COLORS[c], label=c, alpha=0.75)
                      for c in CLASS_ORDER]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=3, fontsize=11, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/predictor_boxplots.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: predictor_boxplots.png')

# =============================================================================
# PLOT 2 — Effect size heatmap: how different are the classes?
# =============================================================================

def plot_effect_sizes(mw_df, feat_names):
    print('\nPlotting effect size heatmap...')

    pairs      = ['Negative vs Near-zero',
                  'Negative vs Positive',
                  'Near-zero vs Positive']
    pair_keys  = [('Negative', 'Near-zero'),
                  ('Negative', 'Positive'),
                  ('Near-zero', 'Positive')]

    r_mat = np.full((len(pairs), len(feat_names)), np.nan)
    p_mat = np.full((len(pairs), len(feat_names)), np.nan)

    for j, feat in enumerate(feat_names):
        for i, (c1, c2) in enumerate(pair_keys):
            row = mw_df[(mw_df['feature'] == feat) &
                        (mw_df['class_1'] == c1) &
                        (mw_df['class_2'] == c2)]
            if not row.empty:
                r_mat[i, j] = row['effect_size_r'].values[0]
                p_mat[i, j] = row['mannwhitney_p'].values[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im   = ax.imshow(r_mat, cmap='RdBu', norm=norm, aspect='auto')
    ax.set_xticks(range(len(feat_names)))
    ax.set_xticklabels(feat_names, fontsize=11)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pairs, fontsize=11)
    plt.colorbar(im, ax=ax, label='Rank-biserial effect size r', shrink=0.8)
    ax.set_title('Effect Size of Class Differences by Predictor\n'
                 'Black outline = p < 0.05\n'
                 'Positive r = class_1 has higher values than class_2',
                 fontsize=11, fontweight='bold')

    from matplotlib.patches import Rectangle
    for i in range(len(pairs)):
        for j in range(len(feat_names)):
            v = r_mat[i, j]
            p = p_mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=10,
                        color='white' if abs(v) > 0.5 else 'black')
            if np.isfinite(p) and p < 0.05:
                ax.add_patch(Rectangle((j-.5, i-.5), 1, 1,
                             fill=False, edgecolor='black', lw=2))

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/effect_size_heatmap.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: effect_size_heatmap.png')

# =============================================================================
# PLOT 3 — Class proportions per crop
# =============================================================================

def plot_class_proportions(df):
    print('\nPlotting class proportions per crop...')
    CLASS_ORDER  = ['Negative', 'Near-zero', 'Positive']
    CLASS_COLORS = {'Negative': '#d73027', 'Near-zero': '#fee090',
                    'Positive': '#4575b4'}

    props = (df.groupby(['crop', 'class'])
               .size()
               .unstack(fill_value=0))
    # Ensure all classes present
    for c in CLASS_ORDER:
        if c not in props.columns:
            props[c] = 0
    props = props[CLASS_ORDER]
    props_pct = props.div(props.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(props_pct))
    for cls in CLASS_ORDER:
        ax.bar(props_pct.index, props_pct[cls],
               bottom=bottom, label=cls,
               color=CLASS_COLORS[cls], alpha=0.85, edgecolor='white')
        bottom += props_pct[cls].values

    ax.set_ylabel('% of pixels', fontsize=12)
    ax.set_title(f'ΔYield Class Proportions per Crop\n'
                 f'(threshold = ±{THRESHOLD} kg/ha)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.set_ylim(0, 100)
    plt.xticks(rotation=30, ha='right', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/class_proportions_per_crop.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    props_pct.to_csv(f'{OUT_DIR}/class_proportions.csv')
    print('  Saved: class_proportions_per_crop.png')

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print('=' * 60)
    print('ΔYield Class Characteristics Analysis')
    print('=' * 60)

    mask      = load_mask().astype(bool)
    elevation = match_grid(np.load(ELEV_PATH), mask.shape)
    slope     = match_grid(load_raster(SLOPE_PATH), mask.shape)
    delta_alt = load_delta_alt(mask)
    asm_base, delta_asm = load_asm(mask)

    predictors = {
        'Elevation'   : elevation,
        'Slope'       : slope,
        'ΔALT'        : delta_alt,
        'Baseline ASM': asm_base,
        'ΔASM'        : delta_asm,
    }

    feat_names = list(predictors.keys())

    df = build_dataset(mask, predictors)
    stats_df, mw_df = compute_summary(df, feat_names)
    plot_boxplots(df, feat_names, mw_df)
    plot_effect_sizes(mw_df, feat_names)
    plot_class_proportions(df)

    print(f'\nAll outputs written to: {OUT_DIR}/')