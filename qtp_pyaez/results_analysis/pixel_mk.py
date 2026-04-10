"""
Step 3 — Pixel-wise 40-Year Suitability Trend (1979–2018)
==========================================================
For each pixel, runs Mann-Kendall trend test on the suitability time series
for both observed and counterfactual scenarios.

This is the spatial version of the 40-year trend analysis — instead of
a single regional mean trend, you see where trends are concentrated.

Outputs per crop:
  - {tag}_obs_trend_map.png    — observed trend significance map
  - {tag}_cf_trend_map.png     — counterfactual trend significance map
  - {tag}_trend_comparison.png — side by side observed vs CF trend maps
  - {tag}_obs_tau.tif          — Kendall's tau raster (observed)
  - {tag}_cf_tau.tif           — Kendall's tau raster (counterfactual)

Outputs overall:
  - ALL_CROPS_obs_trend.png    — summary panel observed trends all crops
  - ALL_CROPS_cf_trend.png     — summary panel CF trends all crops
  - pixelwise_trend_summary.csv

Classification per pixel:
  - Significant positive trend  (p < 0.05, tau > 0)
  - Significant negative trend  (p < 0.05, tau < 0)
  - No significant trend

Outputs written to: ./results_analysis/outputs/6_spatial_analysis/3_pixel_trends/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from pymannkendall import original_test as mk_test
from osgeo import gdal

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR  = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH = r'./data_input/qilian mask.tif'
OUT_ROOT  = './results_analysis/outputs/6_spatial_analysis/3_pixel_trends'

YEARS_ALL       = list(range(1979, 2019))
YEARS_CF        = list(range(1999, 2019))
DIVERGENCE_YEAR = 1999
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

# Classification codes
TREND_SIG_POS = 2    # significant positive trend
TREND_NONE    = 1    # no significant trend
TREND_SIG_NEG = 0    # significant negative trend

TREND_COLORS = {
    TREND_SIG_POS: '#2166AC',   # blue
    TREND_NONE   : '#f0f0f0',   # light grey
    TREND_SIG_NEG: '#c0392b',   # red
}
TREND_LABELS = {
    TREND_SIG_POS: f'Significant positive trend (p < {ALPHA})',
    TREND_NONE   : 'No significant trend',
    TREND_SIG_NEG: f'Significant negative trend (p < {ALPHA})',
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

def make_trend_cmap():
    colors = [TREND_COLORS[TREND_SIG_NEG],
              TREND_COLORS[TREND_NONE],
              TREND_COLORS[TREND_SIG_POS]]
    cmap   = mcolors.ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm   = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm

def plot_trend_map(ax, trend_arr, mask, title):
    cmap, norm = make_trend_cmap()
    display = np.where(mask, trend_arr, np.nan)
    im = ax.imshow(display, cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')
    return im

def trend_legend_patches():
    return [
        mpatches.Patch(color=TREND_COLORS[c], label=TREND_LABELS[c])
        for c in [TREND_SIG_POS, TREND_NONE, TREND_SIG_NEG]
    ]

def pixel_mk(series):
    """Run MK on a 1D series. Returns (tau, p, trend_code)."""
    valid = series[np.isfinite(series)]
    if len(valid) < 4:
        return np.nan, np.nan, TREND_NONE
    try:
        mk = mk_test(valid)
        if mk.p < ALPHA and mk.Tau > 0:
            code = TREND_SIG_POS
        elif mk.p < ALPHA and mk.Tau < 0:
            code = TREND_SIG_NEG
        else:
            code = TREND_NONE
        return mk.Tau, mk.p, code
    except Exception:
        return np.nan, np.nan, TREND_NONE


# ── Main ──────────────────────────────────────────────────────────────────────

def run(mask):
    rows, cols    = mask.shape
    summary       = []
    obs_trend_all = {}
    cf_trend_all  = {}
    geo_info_ref  = None

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        print(f'\n── {label} ──')

        # ── Load full 40-year observed stack ──────────────────────────────────
        obs_stack = []
        for year in YEARS_ALL:
            arr, geo_info = load_raster(obs_suit_path(tag, year))
            if arr is not None:
                obs_stack.append(clean(arr, mask))
                if geo_info_ref is None:
                    geo_info_ref = geo_info
            else:
                obs_stack.append(np.full((rows, cols), np.nan))

        # ── Load CF stack (1999–2018 only, pre-1999 = observed) ───────────────
        cf_stack = []
        for i, year in enumerate(YEARS_ALL):
            if year < DIVERGENCE_YEAR:
                # Pre-divergence: CF identical to observed
                cf_stack.append(obs_stack[i])
            else:
                arr, _ = load_raster(cf_suit_path(tag, year))
                if arr is not None:
                    cf_stack.append(clean(arr, mask))
                else:
                    cf_stack.append(np.full((rows, cols), np.nan))

        obs_stack = np.array(obs_stack)   # (40, rows, cols)
        cf_stack  = np.array(cf_stack)

        # ── Pixel-wise MK ─────────────────────────────────────────────────────
        obs_tau   = np.full((rows, cols), np.nan)
        obs_p     = np.full((rows, cols), np.nan)
        obs_trend = np.full((rows, cols), np.nan)
        cf_tau    = np.full((rows, cols), np.nan)
        cf_p      = np.full((rows, cols), np.nan)
        cf_trend  = np.full((rows, cols), np.nan)

        for r in range(rows):
            for c in range(cols):
                if not mask[r, c]:
                    continue
                tau, p, code = pixel_mk(obs_stack[:, r, c])
                obs_tau[r, c]   = tau
                obs_p[r, c]     = p
                obs_trend[r, c] = code

                tau, p, code = pixel_mk(cf_stack[:, r, c])
                cf_tau[r, c]   = tau
                cf_p[r, c]     = p
                cf_trend[r, c] = code

        obs_trend_all[label] = obs_trend
        cf_trend_all[label]  = cf_trend

        # ── Summary stats ─────────────────────────────────────────────────────
        def trend_pcts(trend_arr):
            valid = trend_arr[mask & np.isfinite(trend_arr)]
            n     = len(valid)
            return {
                'sig_pos': round(float(np.mean(valid == TREND_SIG_POS) * 100), 2),
                'none'   : round(float(np.mean(valid == TREND_NONE)    * 100), 2),
                'sig_neg': round(float(np.mean(valid == TREND_SIG_NEG) * 100), 2),
                'n'      : n,
            }

        obs_pcts = trend_pcts(obs_trend)
        cf_pcts  = trend_pcts(cf_trend)

        summary.append({
            'crop'                   : label,
            'obs_pct_sig_pos'        : obs_pcts['sig_pos'],
            'obs_pct_none'           : obs_pcts['none'],
            'obs_pct_sig_neg'        : obs_pcts['sig_neg'],
            'cf_pct_sig_pos'         : cf_pcts['sig_pos'],
            'cf_pct_none'            : cf_pcts['none'],
            'cf_pct_sig_neg'         : cf_pcts['sig_neg'],
            'diff_sig_pos'           : round(obs_pcts['sig_pos'] - cf_pcts['sig_pos'], 2),
        })

        print(f'  Obs: {obs_pcts["sig_pos"]:.2f}% sig↑ | '
              f'{obs_pcts["sig_neg"]:.2f}% sig↓ | '
              f'{obs_pcts["none"]:.2f}% none')
        print(f'  CF:  {cf_pcts["sig_pos"]:.2f}% sig↑ | '
              f'{cf_pcts["sig_neg"]:.2f}% sig↓ | '
              f'{cf_pcts["none"]:.2f}% none')
        print(f'  Δ sig pos: {obs_pcts["sig_pos"] - cf_pcts["sig_pos"]:+.2f}%')

        # ── Save rasters ──────────────────────────────────────────────────────
        for arr, fname in [
            (obs_tau,   f'{OUT_ROOT}/{tag}_obs_tau.tif'),
            (obs_trend, f'{OUT_ROOT}/{tag}_obs_trend.tif'),
            (cf_tau,    f'{OUT_ROOT}/{tag}_cf_tau.tif'),
            (cf_trend,  f'{OUT_ROOT}/{tag}_cf_trend.tif'),
        ]:
            out = np.where(np.isfinite(arr), arr, -9999.0)
            save_raster(fname, out, geo_info_ref)

        # ── Per-crop figure: obs vs CF trend maps + tau maps ──────────────────
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # Row 1: trend classification maps
        plot_trend_map(axes[0, 0], obs_trend, mask,
                       f'Observed Trend (1979–2018)\n'
                       f'↑{obs_pcts["sig_pos"]:.1f}% | '
                       f'↓{obs_pcts["sig_neg"]:.1f}%')
        plot_trend_map(axes[0, 1], cf_trend, mask,
                       f'No-Thaw CF Trend (1979–2018)\n'
                       f'↑{cf_pcts["sig_pos"]:.1f}% | '
                       f'↓{cf_pcts["sig_neg"]:.1f}%')

        # Row 2: Kendall's tau maps
        tau_vlim = 1.0
        for ax, arr, title in [
            (axes[1, 0], obs_tau, "Kendall's τ — Observed"),
            (axes[1, 1], cf_tau,  "Kendall's τ — No-Thaw CF"),
        ]:
            display = np.where(mask, arr, np.nan)
            im = ax.imshow(display, cmap='RdBu',
                           vmin=-tau_vlim, vmax=tau_vlim)
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, shrink=0.75, label="Kendall's τ")

        # Add legend to first row
        patches = trend_legend_patches()
        axes[0, 0].legend(handles=patches, loc='lower left',
                          fontsize=8, framealpha=0.9)

        fig.suptitle(f'{label} — Pixel-wise 40-Year Suitability Trend',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        fig.savefig(f'{OUT_ROOT}/{tag}_trend_maps.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  ✓ {label} saved')

    # ── Summary panels: all crops ─────────────────────────────────────────────
    for scenario, trend_dict, title_suffix in [
        ('obs', obs_trend_all, 'Observed (1979–2018)'),
        ('cf',  cf_trend_all,  'No-Thaw CF (1979–2018)'),
    ]:
        n     = len(trend_dict)
        ncols = 5
        nrows = -(-n // ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 4, nrows * 3.5))
        axes = axes.flatten()

        for i, (label, trend_arr) in enumerate(trend_dict.items()):
            plot_trend_map(axes[i], trend_arr, mask, label)

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        patches = trend_legend_patches()
        fig.legend(handles=patches, loc='lower center', ncol=3,
                   fontsize=9, bbox_to_anchor=(0.5, -0.02))

        fig.suptitle(
            f'Pixel-wise Suitability Trend — {title_suffix}\nAll Crops',
            fontsize=13, fontweight='bold', y=1.01
        )
        plt.tight_layout()
        fig.savefig(f'{OUT_ROOT}/ALL_CROPS_{scenario}_trend.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ── Summary CSV + bar chart ───────────────────────────────────────────────
    df = pd.DataFrame(summary)
    df.to_csv(f'{OUT_ROOT}/pixelwise_trend_summary.csv', index=False)

    # Bar chart: % sig positive pixels obs vs CF per crop
    x     = np.arange(len(df))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, obs_col, cf_col, title in [
        (axes[0], 'obs_pct_sig_pos', 'cf_pct_sig_pos',
         '% Pixels with Significant Positive Trend'),
        (axes[1], 'obs_pct_sig_neg', 'cf_pct_sig_neg',
         '% Pixels with Significant Negative Trend'),
    ]:
        ax.bar(x - width/2, df[obs_col], width,
               label='Observed', color='#2166AC', alpha=0.85)
        ax.bar(x + width/2, df[cf_col],  width,
               label='No-Thaw CF', color='#D6604D', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(df['crop'], rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('% of Pixels', fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=10)
        ax.axhline(0, color='black', linewidth=0.8)

    fig.suptitle('Pixel-wise Suitability Trend Comparison (1979–2018)\n'
                 'Observed vs. No-Thaw Counterfactual',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{OUT_ROOT}/trend_comparison_barchart.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f'\n✓ All outputs saved to: {OUT_ROOT}/')
    print('\nSummary:')
    print(df.to_string(index=False))


if __name__ == '__main__':
    mask = load_mask()
    run(mask)