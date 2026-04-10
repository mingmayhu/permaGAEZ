"""
Step 2 — Pixel-wise Sign Consistency & Wilcoxon Significance Map
=================================================================
For each pixel, across 1999–2018:
  1. % of years with positive ΔSuitability (sign consistency)
  2. Wilcoxon signed-rank test — is the distribution significantly ≠ 0?
  3. Classification map:
       - Significantly positive  (p < 0.05, median > 0)
       - Significantly negative  (p < 0.05, median < 0)
       - Consistently positive   (≥ 70% years positive, not significant)
       - Consistently negative   (≤ 30% years positive, not significant)
       - Mixed / no effect       (everything else)

Note: pixels with fewer than 4 non-zero ΔSuitability values are skipped
(insufficient data for Wilcoxon test).

Outputs per crop:
  - {tag}_pct_positive.tif/.png   — % years with positive ΔSuitability
  - {tag}_wilcoxon_p.tif          — p-value map (two-sided)
  - {tag}_classification.png      — 5-class significance map

Outputs overall:
  - ALL_CROPS_classification.png  — summary panel across all crops
  - pixel_significance_summary.csv — % of pixels in each category per crop

Outputs written to: ./results_analysis/outputs/6_spatial_analysis/2_sign_consistency/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import wilcoxon
from osgeo import gdal

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR  = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH = r'./data_input/qilian mask.tif'
OUT_ROOT  = './results_analysis/outputs/6_spatial_analysis/2_sign_consistency'

YEARS_CF = list(range(1999, 2019))

# Significance threshold
ALPHA = 0.05
# Consistency threshold (% years positive to be "consistently positive")
CONSISTENCY_THRESHOLD = 0.70

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

# Classification codes — must be sequential for BoundaryNorm
CLASS_SIG_NEG  = 0   # significantly negative
CLASS_CONS_NEG = 1   # consistently negative (not significant)
CLASS_MIXED    = 2   # mixed / no effect
CLASS_CONS_POS = 3   # consistently positive (not significant)
CLASS_SIG_POS  = 4   # significantly positive

CLASS_COLORS = {
    CLASS_SIG_POS  : '#1a6faf',   # strong blue
    CLASS_CONS_POS : '#92c5de',   # light blue
    CLASS_MIXED    : '#f0f0f0',   # light grey
    CLASS_CONS_NEG : '#f4a582',   # light red
    CLASS_SIG_NEG  : '#c0392b',   # strong red
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


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    return arr.astype(bool)

def obs_suit_path(tag, year):
    return f'./data_output/final_classification_fixed/{tag}/{year}_suitability_class.tif'

def cf_suit_path(tag, year):
    return f'./data_output/final_classification_nothaw_fixed/{tag}/{year}_suitability_class.tif'

def clean(arr, mask):
    out = arr.copy()
    out[~mask] = np.nan
    out[out < 0] = np.nan
    return out

def make_colormap():
    """Build a custom colormap for the 5-class significance map."""
    import matplotlib.colors as mcolors
    classes = [CLASS_SIG_NEG, CLASS_CONS_NEG, CLASS_MIXED,
                CLASS_CONS_POS, CLASS_SIG_POS]
    colors  = [CLASS_COLORS[c] for c in classes]
    cmap    = mcolors.ListedColormap(colors)
    bounds  = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm    = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm

def plot_classification(ax, cls_arr, mask, title):
    """Plot the 5-class significance map."""
    cmap, norm = make_colormap()
    display = np.where(mask, cls_arr, np.nan)
    im = ax.imshow(display, cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')
    return im

def add_legend(fig, ax):
    patches = [
        mpatches.Patch(color=CLASS_COLORS[c], label=CLASS_LABELS[c])
        for c in [CLASS_SIG_POS, CLASS_CONS_POS, CLASS_MIXED,
                  CLASS_CONS_NEG, CLASS_SIG_NEG]
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=8,
              framealpha=0.9, title='Thaw Effect', title_fontsize=9)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(mask):
    rows, cols = mask.shape
    summary_records  = []
    all_cls_maps     = {}
    geo_info_ref     = None

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        print(f'\n── {label} ──')

        # ── Load annual ΔSuitability stack ────────────────────────────────────
        delta_stack = []
        for year in YEARS_CF:
            obs, geo_info = load_raster(obs_suit_path(tag, year))
            cf,  _        = load_raster(cf_suit_path(tag, year))
            if obs is None or cf is None:
                continue
            obs = clean(obs, mask)
            cf  = clean(cf,  mask)
            delta = np.where(np.isfinite(obs) & np.isfinite(cf),
                             obs - cf, np.nan)
            delta[~mask] = np.nan
            delta_stack.append(delta)
            if geo_info_ref is None:
                geo_info_ref = geo_info

        if not delta_stack:
            print(f'  ⚠ No data for {label}')
            continue

        delta_stack = np.array(delta_stack)   # (n_years, rows, cols)
        n_years     = delta_stack.shape[0]

        # ── Pixel-wise computations ────────────────────────────────────────────
        pct_positive = np.full((rows, cols), np.nan)
        p_value_map  = np.full((rows, cols), np.nan)
        median_map   = np.full((rows, cols), np.nan)
        cls_map      = np.full((rows, cols), np.nan)

        for r in range(rows):
            for c in range(cols):
                if not mask[r, c]:
                    continue

                series = delta_stack[:, r, c]
                valid  = series[np.isfinite(series)]

                if len(valid) == 0:
                    cls_map[r, c] = CLASS_MIXED
                    continue

                nonzero = valid[valid != 0]

                # If fewer than 3 years show any change — true no effect
                if len(nonzero) < 3:
                    cls_map[r, c] = CLASS_MIXED
                    pct_positive[r, c] = float(np.mean(valid > 0)) * 100
                    median_map[r, c]   = float(np.median(valid))
                    continue

                # Use non-zero values only for consistency classification
                # so zero-dominated pixels aren't misclassified as negative
                pct_pos_nonzero = float(np.mean(nonzero > 0))
                med             = float(np.median(nonzero))
                pct_positive[r, c] = pct_pos_nonzero * 100
                median_map[r, c]   = med

                # Wilcoxon on full valid series (including zeros) —
                # tests whether the overall distribution differs from zero
                if len(nonzero) >= 4:
                    try:
                        _, p = wilcoxon(valid, alternative='two-sided')
                        p_value_map[r, c] = p
                        sig = p < ALPHA
                    except Exception:
                        sig = False
                        p_value_map[r, c] = np.nan
                else:
                    sig = False

                # Classify pixel based on non-zero sign consistency
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

        all_cls_maps[label] = cls_map

        # ── Summary stats ─────────────────────────────────────────────────────
        valid_cls = cls_map[mask & np.isfinite(cls_map)]
        n_total   = len(valid_cls)
        record    = {'crop': label, 'n_pixels': n_total}
        for code, lbl in CLASS_LABELS.items():
            pct = float(np.mean(valid_cls == code) * 100) if n_total > 0 else 0
            record[lbl.lower().replace(' ', '_').replace('/', '_')] = round(pct, 2)
        summary_records.append(record)

        print(f'  Sig positive:    {record.get("significantly_positive", 0):.2f}%')
        print(f'  Cons positive:   {record.get("consistently_positive", 0):.2f}%')
        print(f'  Mixed:           {record.get("mixed___no_effect", 0):.2f}%')
        print(f'  Cons negative:   {record.get("consistently_negative", 0):.2f}%')
        print(f'  Sig negative:    {record.get("significantly_negative", 0):.2f}%')

        # ── Save rasters ──────────────────────────────────────────────────────
        for arr, fname in [
            (pct_positive, f'{OUT_ROOT}/{tag}_pct_positive.tif'),
            (p_value_map,  f'{OUT_ROOT}/{tag}_wilcoxon_p.tif'),
            (cls_map,      f'{OUT_ROOT}/{tag}_classification.tif'),
        ]:
            out = np.where(np.isfinite(arr), arr, -9999.0)
            save_raster(fname, out, geo_info_ref)

        # ── Per-crop figure: 3 panels ─────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Panel 1: % non-zero years positive — only show pixels with
        # enough non-zero values to be meaningful (same threshold as classification)
        has_enough = np.full((rows, cols), False)
        for r in range(rows):
            for c in range(cols):
                if not mask[r, c]:
                    continue
                series  = delta_stack[:, r, c]
                valid   = series[np.isfinite(series)]
                nonzero = valid[valid != 0]
                has_enough[r, c] = len(nonzero) >= 3

        # Only show pct_positive where pixel has enough non-zero values
        disp_pct = np.where(mask & has_enough, pct_positive, np.nan)
        im0 = axes[0].imshow(disp_pct, cmap='RdBu', vmin=0, vmax=100)
        axes[0].set_title('% Non-Zero Years Positive ΔSuitability\n'
                          '(only pixels with ≥3 years of change shown)',
                          fontsize=11, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], shrink=0.75, label='% non-zero years')

        # Panel 2: p-value map — only show where Wilcoxon was run
        disp_p = np.where(mask & np.isfinite(p_value_map), p_value_map, np.nan)
        im1 = axes[1].imshow(disp_p, cmap='YlOrRd_r', vmin=0, vmax=0.1)
        axes[1].set_title('Wilcoxon p-value\n(darker = more significant, '
                          'grey = insufficient data)',
                          fontsize=11, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], shrink=0.75, label='p-value')

        # Panel 3: classification map
        plot_classification(axes[2], cls_map, mask,
                            '5-Class Significance Map')
        add_legend(fig, axes[2])

        fig.suptitle(f'{label} — Pixel-wise Thaw Effect Significance (1999–2018)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        fig.savefig(f'{OUT_ROOT}/{tag}_significance.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  ✓ {label} saved')

    # ── Summary panel: classification maps all crops ───────────────────────────
    n     = len(all_cls_maps)
    ncols = 5
    nrows = -(-n // ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4, nrows * 3.5))
    axes = axes.flatten()

    for i, (label, cls_map) in enumerate(all_cls_maps.items()):
        plot_classification(axes[i], cls_map, mask, label)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Add shared legend
    patches = [
        mpatches.Patch(color=CLASS_COLORS[c], label=CLASS_LABELS[c])
        for c in [CLASS_SIG_POS, CLASS_CONS_POS, CLASS_MIXED,
                  CLASS_CONS_NEG, CLASS_SIG_NEG]
    ]
    fig.legend(handles=patches, loc='lower center', ncol=5,
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        'Pixel-wise Thaw Effect Significance — All Crops (1999–2018)\n'
        f'Significance threshold: p < {ALPHA}, '
        f'Consistency threshold: ≥{int(CONSISTENCY_THRESHOLD*100)}% years',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    fig.savefig(f'{OUT_ROOT}/ALL_CROPS_classification.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Summary CSV ───────────────────────────────────────────────────────────
    df = pd.DataFrame(summary_records)
    df.to_csv(f'{OUT_ROOT}/pixel_significance_summary.csv', index=False)
    print(f'\n✓ Summary saved to: {OUT_ROOT}/pixel_significance_summary.csv')
    print(df.to_string(index=False))
    print(f'\n✓ All outputs saved to: {OUT_ROOT}/')


if __name__ == '__main__':
    mask = load_mask()
    run(mask)