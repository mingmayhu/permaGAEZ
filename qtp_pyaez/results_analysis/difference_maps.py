"""
Annual ΔYield Maps (Observed − No-Thaw Counterfactual)
=======================================================
For each crop × year (1999–2018):
  - Saves individual GeoTIFF of ΔYield
  - Saves a small-multiples PNG panel of all 20 years
  - Saves a regional-mean ΔYield time series CSV + line plot
  - Wilcoxon signed-rank test: is ΔYield consistently above zero?
  - Summary figure of Wilcoxon results across all crops
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from osgeo import gdal
from scipy.stats import wilcoxon

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR  = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH = r'./data_input/qilian mask.tif'
OUT_ROOT  = './results_analysis/outputs/1_annual_delta_output'
YEARS     = list(range(1999, 2019))

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

DIVERGING_CMAP = 'RdBu'

os.chdir(WORK_DIR)
os.makedirs(OUT_ROOT, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_raster(path):
    ds = gdal.Open(path)
    if ds is None:
        return None, None
    band     = ds.GetRasterBand(1)
    nodata   = band.GetNoDataValue()
    arr      = ds.ReadAsArray().astype(float)
    arr[arr < -1e10] = np.nan
    if nodata is not None:
        arr[arr == nodata] = np.nan
    geo      = ds.GetGeoTransform()
    proj     = ds.GetProjection()
    return arr, (geo, proj, ds.RasterXSize, ds.RasterYSize)

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

def spatial_mean(arr, mask):
    valid = mask & np.isfinite(arr) & (arr > 0)
    return float(np.nanmean(arr[valid])) if valid.any() else np.nan

def spatial_mean_delta(delta, mask):
    """Mean over all valid pixels regardless of sign — for ΔYield."""
    valid = mask & np.isfinite(delta)
    return float(np.nanmean(delta[valid])) if valid.any() else np.nan

def obs_path(tag, year):
    return f'./data_output/final_classification/{tag}/{year}_raw_yield.tif'

def cf_path(tag, year):
    return f'./data_output/final_classification_nothaw/{tag}/{year}_raw_yield.tif'


# ── Main analysis ─────────────────────────────────────────────────────────────

def run(mask):
    all_ts       = []
    wilcoxon_res = []   # collect Wilcoxon results across crops

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        print(f'\n── {label} ──')

        tif_dir = f'{OUT_ROOT}/tif/{tag}'
        os.makedirs(tif_dir, exist_ok=True)

        annual_deltas = {}
        ts_records    = []

        # ── Step 1: compute ΔYield for each year ──────────────────────────────
        for year in YEARS:
            obs, geo_info = load_raster(obs_path(tag, year))
            cf,  _        = load_raster(cf_path(tag, year))

            if obs is None:
                print(f'  ⚠ missing observed  {year}')
                continue
            if cf is None:
                print(f'  ⚠ missing counterfactual {year}')
                continue

            obs[~mask] = np.nan
            cf[~mask]  = np.nan

            delta = np.where(np.isfinite(obs) & np.isfinite(cf),
                             obs - cf, np.nan)
            delta[~mask] = np.nan
            annual_deltas[year] = delta

            # Save GeoTIFF
            out_arr = np.where(np.isfinite(delta), delta, -9999.0)
            save_raster(f'{tif_dir}/{year}_delta_yield.tif', out_arr, geo_info)

            # Note: use spatial_mean_delta here (includes negative values)
            # so the Wilcoxon test is not biased by dropping negative pixels
            ts_records.append({
                'year'        : year,
                'crop'        : label,
                'mean_delta'  : spatial_mean_delta(delta, mask),
                'pct_positive': float(np.nanmean(delta[mask & np.isfinite(delta)] > 0) * 100),
                'pct_negative': float(np.nanmean(delta[mask & np.isfinite(delta)] < 0) * 100),
            })

        if not annual_deltas:
            print(f'  ⚠ No data for {label}, skipping.')
            continue

        all_ts.extend(ts_records)

        # ── Step 2: Wilcoxon signed-rank test ─────────────────────────────────
        # Tests whether the 20-year distribution of regional mean ΔYield
        # is centered significantly above (or below) zero.
        # alternative='greater' = one-sided test: is observed > counterfactual?
        mean_deltas = np.array([r['mean_delta'] for r in ts_records])
        valid_deltas = mean_deltas[np.isfinite(mean_deltas)]

        if len(valid_deltas) >= 4:
            stat, p_two  = wilcoxon(valid_deltas, alternative='two-sided')
            _,    p_pos  = wilcoxon(valid_deltas, alternative='greater')
            _,    p_neg  = wilcoxon(valid_deltas, alternative='less')
            median_delta = float(np.median(valid_deltas))
            pct_yrs_pos  = float(np.mean(valid_deltas > 0) * 100)

            wilcoxon_res.append({
                'crop'           : label,
                'median_delta'   : round(median_delta, 4),
                'pct_years_positive': round(pct_yrs_pos, 1),
                'wilcoxon_stat'  : round(stat, 2),
                'p_two_sided'    : round(p_two, 4),
                'p_greater_zero' : round(p_pos, 4),   # obs > CF
                'p_less_zero'    : round(p_neg, 4),   # obs < CF
                'sig_positive'   : p_pos < 0.05,
                'sig_negative'   : p_neg < 0.05,
                'n_years'        : len(valid_deltas),
            })
            print(f'  Wilcoxon: median_delta={median_delta:.4f}, '
                  f'p(greater)={p_pos:.4f}, p(two-sided)={p_two:.4f}, '
                  f'{pct_yrs_pos:.0f}% of years positive')
        else:
            print(f'  ⚠ Too few valid years for Wilcoxon test.')

        # ── Step 3: consistent colour scale across all years ──────────────────
        all_vals = np.concatenate([
            d[mask & np.isfinite(d)] for d in annual_deltas.values()
        ])
        vlim = np.nanpercentile(np.abs(all_vals), 98)
        if vlim == 0 or np.isnan(vlim):
            vlim = 1.0

        # ── Step 4: small-multiples panel ─────────────────────────────────────
        years_sorted = sorted(annual_deltas.keys())
        ncols = 5
        nrows = -(-len(years_sorted) // ncols)

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 3.5, nrows * 3.2))
        axes = axes.flatten()

        for i, year in enumerate(years_sorted):
            ax = axes[i]
            im = ax.imshow(annual_deltas[year], cmap=DIVERGING_CMAP,
                           vmin=-vlim, vmax=vlim)
            ax.set_title(str(year), fontsize=10, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, shrink=0.75, label='kg/ha')

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        fig.suptitle(
            f'{label} — Annual ΔYield (Observed − No-Thaw)\n'
            f'Colour scale: ±{vlim:.1f} kg/ha (98th pct)',
            fontsize=13, fontweight='bold', y=1.01
        )
        plt.tight_layout()
        fig.savefig(f'{OUT_ROOT}/{tag}_annual_delta_panel.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  ✓ small-multiples panel saved')

        # ── Step 5: regional mean ΔYield time series with Wilcoxon annotation ──
        ts_df = pd.DataFrame(ts_records).sort_values('year')
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.bar(ts_df['year'], ts_df['mean_delta'],
                color=np.where(ts_df['mean_delta'] >= 0, '#2166AC', '#D6604D'),
                edgecolor='white', width=0.8)
        ax2.axhline(0, color='black', linewidth=0.8)

        # Annotate with Wilcoxon result if available
        if wilcoxon_res and wilcoxon_res[-1]['crop'] == label:
            wr = wilcoxon_res[-1]
            sig_str = '★ p < 0.05' if wr['sig_positive'] else 'n.s.'
            ax2.text(0.02, 0.97,
                     f"Wilcoxon (obs > CF): p = {wr['p_greater_zero']:.3f} {sig_str}\n"
                     f"Median ΔYield = {wr['median_delta']:.3f} kg/ha  |  "
                     f"{wr['pct_years_positive']:.0f}% of years positive",
                     transform=ax2.transAxes, fontsize=9, va='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax2.set_xlabel('Year', fontsize=11)
        ax2.set_ylabel('Regional Mean ΔYield (kg/ha)', fontsize=11)
        ax2.set_title(f'{label} — Annual Mean ΔYield (Observed − No-Thaw)',
                      fontsize=12, fontweight='bold')
        ax2.set_xticks(YEARS)
        ax2.set_xticklabels(YEARS, rotation=45, ha='right', fontsize=8)
        plt.tight_layout()
        fig2.savefig(f'{OUT_ROOT}/{tag}_annual_delta_timeseries.png',
                     dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  ✓ time series bar chart saved')

    # ── Save combined CSV ──────────────────────────────────────────────────────
    pd.DataFrame(all_ts).to_csv(f'{OUT_ROOT}/annual_delta_all_crops.csv', index=False)
    print(f'\n✓ CSV saved: {OUT_ROOT}/annual_delta_all_crops.csv')

    # ── Wilcoxon summary CSV ───────────────────────────────────────────────────
    if wilcoxon_res:
        wdf = pd.DataFrame(wilcoxon_res)
        wdf.to_csv(f'{OUT_ROOT}/wilcoxon_results.csv', index=False)
        print('\nWilcoxon Results:')
        print(wdf[['crop', 'median_delta', 'pct_years_positive',
                    'p_greater_zero', 'p_two_sided', 'sig_positive']].to_string(index=False))

        # ── Wilcoxon summary figure ────────────────────────────────────────────
        wdf_s = wdf.sort_values('median_delta')
        colors = ['#2166AC' if s else '#AAAAAA' for s in wdf_s['sig_positive']]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: median ΔYield
        bars = axes[0].barh(wdf_s['crop'], wdf_s['median_delta'],
                            color=colors, edgecolor='white')
        axes[0].axvline(0, color='black', linewidth=0.8)
        for bar, (_, row) in zip(bars, wdf_s.iterrows()):
            x = bar.get_width()
            axes[0].text(x + 0.01 * np.sign(x) if x != 0 else 0.01,
                         bar.get_y() + bar.get_height() / 2,
                         f'p={row["p_greater_zero"]:.3f}'
                         f'{"★" if row["sig_positive"] else ""}',
                         va='center', fontsize=9,
                         ha='left' if x >= 0 else 'right')
        axes[0].set_xlabel('Median ΔYield (kg/ha)', fontsize=11)
        axes[0].set_title('Median Annual ΔYield\n(Blue = significantly > 0)',
                          fontsize=11, fontweight='bold')

        # Right: % years positive
        axes[1].barh(wdf_s['crop'], wdf_s['pct_years_positive'],
                     color=colors, edgecolor='white')
        axes[1].axvline(50, color='black', linewidth=0.8, linestyle='--',
                        label='50% (no consistent sign)')
        axes[1].set_xlabel('% of Years with Positive ΔYield', fontsize=11)
        axes[1].set_title('% Years Observed > Counterfactual\n(Blue = significantly > 0)',
                          fontsize=11, fontweight='bold')
        axes[1].legend(fontsize=9)
        axes[1].set_xlim(0, 100)

        fig.suptitle('Is Permafrost Thaw Consistently Positive for Yield?\n'
                     'Wilcoxon Signed-Rank Test on Annual ΔYield (1999–2018)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        fig.savefig(f'{OUT_ROOT}/wilcoxon_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        print('✓ Wilcoxon summary figure saved.')

    print(f'\n✓ Done. All outputs in: {OUT_ROOT}/')


if __name__ == '__main__':
    mask = load_mask()
    run(mask)