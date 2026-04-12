"""
Spatial Analysis — Suitability Class Data
==========================================
1. Mean ΔSuitability map — average (Thaw − No-Thaw) per pixel over 1999–2018
2. Pixel-wise Wilcoxon signed-rank test with 5-class significance map:
     - Significantly positive  (Wilcoxon p < 0.05, median > 0)
     - Consistently positive   (>= 70% non-zero years positive, not significant)
     - Mixed / no effect
     - Consistently negative   (<= 30% non-zero years positive, not significant)
     - Significantly negative  (Wilcoxon p < 0.05, median < 0)

Class 0 and class 1 are combined into class 1 (remap) before all calculations.
Wilcoxon requires >= 4 non-zero delta values per pixel.
Consistency classification requires >= 5 non-zero delta values.

Outputs written to:
  ./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/5_spatial/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from osgeo import gdal
from scipy.stats import wilcoxon

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR  = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH = r'./data_input/qilian mask.tif'
OUT_ROOT  = './results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/5_spatial'

YEARS_CF = list(range(1999, 2019))

ALPHA                   = 0.05
CONSISTENCY_THRESHOLD   = 0.70
MIN_NONZERO_WILCOXON    = 4
MIN_NONZERO_CONSISTENCY = 5

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

# Classification codes
CLASS_SIG_NEG  = 0
CLASS_CONS_NEG = 1
CLASS_MIXED    = 2
CLASS_CONS_POS = 3
CLASS_SIG_POS  = 4

CLASS_COLORS = {
    CLASS_SIG_POS  : '#1a6faf',
    CLASS_CONS_POS : '#92c5de',
    CLASS_MIXED    : '#f0f0f0',
    CLASS_CONS_NEG : '#f4a582',
    CLASS_SIG_NEG  : '#c0392b',
}
CLASS_LABELS = {
    CLASS_SIG_POS  : 'Significantly positive',
    CLASS_CONS_POS : 'Consistently positive',
    CLASS_MIXED    : 'Mixed / no effect',
    CLASS_CONS_NEG : 'Consistently negative',
    CLASS_SIG_NEG  : 'Significantly negative',
}

os.chdir(WORK_DIR)
os.makedirs(OUT_ROOT, exist_ok=True)
for sub in ['1_mean_delta', '2_wilcoxon']:
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

def make_colormap():
    import matplotlib.colors as mcolors
    colors = [CLASS_COLORS[c] for c in
              [CLASS_SIG_NEG, CLASS_CONS_NEG, CLASS_MIXED, CLASS_CONS_POS, CLASS_SIG_POS]]
    cmap   = mcolors.ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm   = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm

def plot_classification(ax, cls_arr, mask, title):
    cmap, norm = make_colormap()
    display = np.where(mask, cls_arr, np.nan)
    ax.imshow(display, cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')

def legend_patches():
    return [
        mpatches.Patch(color=CLASS_COLORS[c], label=CLASS_LABELS[c])
        for c in [CLASS_SIG_POS, CLASS_CONS_POS, CLASS_MIXED,
                  CLASS_CONS_NEG, CLASS_SIG_NEG]
    ]

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


# ── Analysis 1: Mean delta suitability map ────────────────────────────────────

def analysis_mean_delta(mask):
    print('\n[Analysis 1] Mean delta suitability maps ...')
    out_dir  = f'{OUT_ROOT}/1_mean_delta'
    geo_info = None
    mean_deltas = []
    labels      = []

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        print(f'  {label}')

        delta_stack = []
        for year in YEARS_CF:
            obs, gi = load_raster(obs_suit_path(tag, year))
            cf,  _  = load_raster(cf_suit_path(tag, year))
            if obs is None or cf is None:
                continue
            if geo_info is None:
                geo_info = gi
            obs_r = apply_remap(obs, mask)
            cf_r  = apply_remap(cf,  mask)
            delta = np.where(np.isfinite(obs_r) & np.isfinite(cf_r),
                             obs_r - cf_r, np.nan)
            delta[~mask] = np.nan
            delta_stack.append(delta)

        if not delta_stack:
            print(f'    Warning: no data for {label}')
            continue

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mean_delta = np.nanmean(np.stack(delta_stack), axis=0)
        mean_delta[~mask] = np.nan
        mean_deltas.append(mean_delta)
        labels.append(label)

        save_raster(f'{out_dir}/{tag}_mean_delta_suit.tif',
                    np.where(np.isfinite(mean_delta), mean_delta, -9999.0),
                    geo_info)

    # Overall aggregate
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        overall = np.nanmean(np.stack(mean_deltas), axis=0)
    overall[~mask] = np.nan
    mean_deltas.append(overall)
    labels.append('OVERALL')
    save_raster(f'{out_dir}/overall_mean_delta_suit.tif',
                np.where(np.isfinite(overall), overall, -9999.0),
                geo_info)

    all_vals = np.concatenate([d[mask & np.isfinite(d)] for d in mean_deltas])
    vlim = max(float(np.nanpercentile(np.abs(all_vals), 98)), 1e-6)

    plot_panel(
        mean_deltas, labels, mask,
        suptitle=f'Mean delta Suitability (Thaw minus No-Thaw), 1999-2018\n'
                 f'Colour scale: +/-{vlim:.3f} class units (98th pct)',
        cmap='RdBu', vmin=-vlim, vmax=vlim, vcenter=0,
        cbar_label='delta Class',
        out_path=f'{out_dir}/mean_delta_panel.png',
        ncols=4
    )
    plot_map(
        overall, mask,
        title='Overall Mean delta Suitability (Thaw minus No-Thaw), 1999-2018',
        cmap='RdBu', vmin=-vlim, vmax=vlim, vcenter=0,
        cbar_label='Mean delta Class',
        out_path=f'{out_dir}/overall_mean_delta_map.png'
    )

    print(f'  Overall range: [{np.nanmin(overall):.4f}, {np.nanmax(overall):.4f}]')
    print(f'  Mean delta maps saved to {out_dir}/')
    return mean_deltas, labels, geo_info


# ── Analysis 2: Pixel-wise Wilcoxon with 5-class significance map ─────────────

def run_wilcoxon_pixel(delta_stack, mask, rows, cols):
    """Run pixel-wise Wilcoxon on a (n_years, rows, cols) delta stack.
    Returns pct_positive, p_value_map, median_map, cls_map.
    """
    pct_positive = np.full((rows, cols), np.nan)
    p_value_map  = np.full((rows, cols), np.nan)
    median_map   = np.full((rows, cols), np.nan)
    cls_map      = np.full((rows, cols), np.nan)

    for r, c in np.argwhere(mask):
        series  = delta_stack[:, r, c]
        valid   = series[np.isfinite(series)]
        nonzero = valid[valid != 0]

        if len(valid) == 0:
            cls_map[r, c] = CLASS_MIXED
            continue

        if len(nonzero) < MIN_NONZERO_CONSISTENCY:
            cls_map[r, c] = CLASS_MIXED
            continue

        pct_pos_nonzero    = float(np.mean(nonzero > 0))
        pct_positive[r, c] = pct_pos_nonzero * 100
        median_map[r, c]   = float(np.median(nonzero))

        sig = False
        if len(nonzero) >= MIN_NONZERO_WILCOXON:
            try:
                _, p = wilcoxon(valid, alternative='two-sided')
                p_value_map[r, c] = p
                sig = p < ALPHA
            except Exception:
                p_value_map[r, c] = np.nan

        med = median_map[r, c]
        if sig and med > 0:
            cls_map[r, c] = CLASS_SIG_POS
        elif sig and med < 0:
            cls_map[r, c] = CLASS_SIG_NEG
        elif pct_pos_nonzero >= CONSISTENCY_THRESHOLD:
            cls_map[r, c] = CLASS_CONS_POS
        elif pct_pos_nonzero <= (1 - CONSISTENCY_THRESHOLD):
            cls_map[r, c] = CLASS_CONS_NEG
        else:
            cls_map[r, c] = CLASS_MIXED

    return pct_positive, p_value_map, median_map, cls_map


def analysis_pixel_wilcoxon(mask, geo_info):
    print('\n[Analysis 2] Pixel-wise Wilcoxon signed-rank test ...')
    out_dir          = f'{OUT_ROOT}/2_wilcoxon'
    rows, cols       = mask.shape
    all_cls_maps     = {}
    summary_records  = []
    all_delta_stacks = []   # collect all crop stacks for overall

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        print(f'  {label}')

        # Build delta stack with remap applied
        delta_stack = []
        for year in YEARS_CF:
            obs, _ = load_raster(obs_suit_path(tag, year))
            cf,  _ = load_raster(cf_suit_path(tag, year))
            if obs is None or cf is None:
                delta_stack.append(np.full((rows, cols), np.nan))
                continue
            obs_r = apply_remap(obs, mask)
            cf_r  = apply_remap(cf,  mask)
            delta = np.where(np.isfinite(obs_r) & np.isfinite(cf_r),
                             obs_r - cf_r, np.nan)
            delta[~mask] = np.nan
            delta_stack.append(delta)

        delta_stack = np.stack(delta_stack)   # (n_years, rows, cols)
        all_delta_stacks.append(delta_stack)  # collect for overall

        pct_positive, p_value_map, median_map, cls_map = run_wilcoxon_pixel(
            delta_stack, mask, rows, cols
        )

        all_cls_maps[label] = cls_map

        # Summary stats
        valid_cls = cls_map[mask & np.isfinite(cls_map)]
        n_total   = len(valid_cls)
        record    = {'crop': label, 'n_pixels': n_total}
        for code, lbl in CLASS_LABELS.items():
            pct = float(np.mean(valid_cls == code) * 100) if n_total > 0 else 0.0
            key = lbl.lower().replace(' ', '_').replace('/', '_')
            record[key] = round(pct, 2)
        summary_records.append(record)

        print(f'    Sig positive:  {record.get("significantly_positive", 0):.2f}%')
        print(f'    Cons positive: {record.get("consistently_positive", 0):.2f}%')
        print(f'    Mixed:         {record.get("mixed___no_effect", 0):.2f}%')
        print(f'    Cons negative: {record.get("consistently_negative", 0):.2f}%')
        print(f'    Sig negative:  {record.get("significantly_negative", 0):.2f}%')

        # Save rasters
        for arr, fname in [
            (pct_positive, f'{out_dir}/{tag}_pct_positive.tif'),
            (p_value_map,  f'{out_dir}/{tag}_wilcoxon_p.tif'),
            (cls_map,      f'{out_dir}/{tag}_classification.tif'),
        ]:
            save_raster(fname,
                        np.where(np.isfinite(arr), arr, -9999.0),
                        geo_info)

        # Per-crop 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Panel 1: % non-zero years positive
        disp_pct = np.where(mask & np.isfinite(pct_positive), pct_positive, np.nan)
        im0 = axes[0].imshow(disp_pct, cmap='RdBu', vmin=0, vmax=100)
        axes[0].set_title(f'% Non-Zero Years with Positive delta Suitability\n'
                          f'(pixels with >={MIN_NONZERO_CONSISTENCY} non-zero years shown)',
                          fontsize=10, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], shrink=0.75, label='%')

        # Panel 2: p-value map
        disp_p = np.where(mask & np.isfinite(p_value_map), p_value_map, np.nan)
        im1 = axes[1].imshow(disp_p, cmap='YlOrRd_r', vmin=0, vmax=0.1)
        axes[1].set_title(f'Wilcoxon p-value\n'
                          f'(pixels with >={MIN_NONZERO_WILCOXON} non-zero years shown)',
                          fontsize=10, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], shrink=0.75, label='p-value')

        # Panel 3: 5-class classification
        plot_classification(axes[2], cls_map, mask, '5-Class Significance Map')
        axes[2].legend(handles=legend_patches(), loc='lower right',
                       fontsize=8, framealpha=0.9,
                       title='Thaw Effect', title_fontsize=9)

        fig.suptitle(f'{label} — Pixel-wise Thaw Effect (1999-2018)\n'
                     f'Class 0 and 1 combined | Wilcoxon p < {ALPHA} | '
                     f'Consistency >= {int(CONSISTENCY_THRESHOLD*100)}% non-zero years',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        fig.savefig(f'{out_dir}/{tag}_wilcoxon.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'    Saved {label}')

    # ── Overall Wilcoxon ─────────────────────────────────────────────────────
    print('  OVERALL')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        overall_stack = np.nanmean(np.stack(all_delta_stacks), axis=0)

    pct_positive_ov, p_value_ov, median_ov, cls_ov = run_wilcoxon_pixel(
        overall_stack, mask, rows, cols
    )
    all_cls_maps['OVERALL'] = cls_ov

    # Summary stats for overall
    valid_cls = cls_ov[mask & np.isfinite(cls_ov)]
    n_total   = len(valid_cls)
    record    = {'crop': 'OVERALL', 'n_pixels': n_total}
    for code, lbl in CLASS_LABELS.items():
        pct = float(np.mean(valid_cls == code) * 100) if n_total > 0 else 0.0
        key = lbl.lower().replace(' ', '_').replace('/', '_')
        record[key] = round(pct, 2)
    summary_records.append(record)

    print(f'    Sig positive:  {record.get("significantly_positive", 0):.2f}%')
    print(f'    Cons positive: {record.get("consistently_positive", 0):.2f}%')
    print(f'    Mixed:         {record.get("mixed___no_effect", 0):.2f}%')
    print(f'    Cons negative: {record.get("consistently_negative", 0):.2f}%')
    print(f'    Sig negative:  {record.get("significantly_negative", 0):.2f}%')

    # Save overall rasters
    for arr, fname in [
        (pct_positive_ov, f'{out_dir}/overall_pct_positive.tif'),
        (p_value_ov,      f'{out_dir}/overall_wilcoxon_p.tif'),
        (cls_ov,          f'{out_dir}/overall_classification.tif'),
    ]:
        save_raster(fname, np.where(np.isfinite(arr), arr, -9999.0), geo_info)

    # Overall 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    disp_pct = np.where(mask & np.isfinite(pct_positive_ov), pct_positive_ov, np.nan)
    im0 = axes[0].imshow(disp_pct, cmap='RdBu', vmin=0, vmax=100)
    axes[0].set_title(f'% Non-Zero Years with Positive delta Suitability'
                      f'(pixels with >={MIN_NONZERO_CONSISTENCY} non-zero years shown)',
                      fontsize=10, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], shrink=0.75, label='%')

    disp_p = np.where(mask & np.isfinite(p_value_ov), p_value_ov, np.nan)
    im1 = axes[1].imshow(disp_p, cmap='YlOrRd_r', vmin=0, vmax=0.1)
    axes[1].set_title(f'Wilcoxon p-value'
                      f'(pixels with >={MIN_NONZERO_WILCOXON} non-zero years shown)',
                      fontsize=10, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], shrink=0.75, label='p-value')

    plot_classification(axes[2], cls_ov, mask, '5-Class Significance Map')
    axes[2].legend(handles=legend_patches(), loc='lower right',
                   fontsize=8, framealpha=0.9,
                   title='Thaw Effect', title_fontsize=9)

    fig.suptitle(f'OVERALL — Pixel-wise Thaw Effect (1999-2018)'
                 f'Mean across all 10 crops | Class 0 and 1 combined',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{out_dir}/overall_wilcoxon.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Standalone classification map only
    fig_s, ax_s = plt.subplots(figsize=(10, 6))
    plot_classification(ax_s, cls_ov, mask,
                        'Overall Pixel-wise Thaw Effect (1999-2018)\n'
                        'Mean across all 10 crops | Class 0 and 1 combined')
    ax_s.legend(handles=legend_patches(), loc='lower right',
                fontsize=9, framealpha=0.9,
                title='Thaw Effect', title_fontsize=10)
    plt.tight_layout()
    fig_s.savefig(f'{out_dir}/overall_classification_map.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved OVERALL')

    # Updated summary panel including overall
    ncols = 4
    nrows = -(-len(all_cls_maps) // ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4, nrows * 3.5))
    axes = axes.flatten()
    for i, (label, cls_map) in enumerate(all_cls_maps.items()):
        plot_classification(axes[i], cls_map, mask, label)
    for j in range(len(all_cls_maps), len(axes)):
        axes[j].axis('off')
    fig.legend(handles=legend_patches(), loc='lower center', ncol=5,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f'Pixel-wise Thaw Effect — All Crops + Overall (1999-2018)'
        f'Wilcoxon p < {ALPHA} | Consistency >= {int(CONSISTENCY_THRESHOLD*100)}% '
        f'non-zero years | Class 0 and 1 combined',
        fontsize=12, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    fig.savefig(f'{out_dir}/ALL_CROPS_OVERALL_classification.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    # Summary CSV
    df = pd.DataFrame(summary_records)
    df.to_csv(f'{out_dir}/pixel_significance_summary.csv', index=False)
    print(f'\n  Summary CSV saved')
    print(df[['crop', 'n_pixels',
              'significantly_positive', 'consistently_positive',
              'mixed___no_effect',
              'consistently_negative', 'significantly_negative']].to_string(index=False))

    return all_cls_maps, summary_records


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    mask = load_mask()
    mean_deltas, delta_labels, geo_info = analysis_mean_delta(mask)
    analysis_pixel_wilcoxon(mask, geo_info)
    print(f'\nAll spatial analyses complete. Outputs in: {OUT_ROOT}/')