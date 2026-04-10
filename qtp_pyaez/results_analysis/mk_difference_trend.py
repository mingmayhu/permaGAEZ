"""
Mann-Kendall Trend Test on Regional Mean ΔYield (1999–2018)
============================================================
For each crop, tests whether the annual regional mean ΔYield
(observed − no-thaw counterfactual) has a significant temporal trend.

Outputs:
  - mk_trend_results.csv      — full MK statistics per crop
  - mk_trend_summary.png      — bar chart of Kendall's tau + significance
  - {tag}_mk_timeseries.png   — per-crop time series with Sen's slope overlaid
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from osgeo import gdal
from pymannkendall import original_test as mk_test

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH = r'./data_input/qilian mask.tif'
OUT_ROOT  = './results_analysis/outputs/2_mk_trend_output'
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

def spatial_mean_delta(tag, year, mask):
    obs = load_raster(f'./data_output/final_classification/{tag}/{year}_raw_yield.tif')
    cf  = load_raster(f'./data_output/final_classification_nothaw/{tag}/{year}_raw_yield.tif')
    if obs is None or cf is None:
        return np.nan
    delta = np.where(np.isfinite(obs) & np.isfinite(cf), obs - cf, np.nan)
    valid = mask & np.isfinite(delta)
    return float(np.nanmean(delta[valid])) if valid.any() else np.nan


# ── Main ──────────────────────────────────────────────────────────────────────

def run(mask):
    results = []

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']

        # Build annual time series of regional mean ΔYield
        series = [spatial_mean_delta(tag, yr, mask) for yr in YEARS]
        series = np.array(series)

        # Drop NaN years before testing
        valid_mask = np.isfinite(series)
        valid_years  = np.array(YEARS)[valid_mask]
        valid_series = series[valid_mask]

        if len(valid_series) < 4:
            print(f'⚠ {label}: too few valid years, skipping.')
            continue

        # Mann-Kendall test
        mk = mk_test(valid_series)

        results.append({
            'crop'       : label,
            'tau'        : round(mk.Tau, 3),
            'p_value'    : round(mk.p, 4),
            'slope_sen'  : round(mk.slope, 4),   # Sen's slope (kg/ha/yr)
            'trend'      : mk.trend,              # 'increasing', 'decreasing', 'no trend'
            'significant': mk.p < 0.05,
            'n_years'    : int(valid_mask.sum()),
        })

        # ── Per-crop time series plot ──────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 4))

        # Bar chart of annual ΔYield
        colors = ['#2166AC' if v >= 0 else '#D6604D' for v in valid_series]
        ax.bar(valid_years, valid_series, color=colors, edgecolor='white', width=0.8,
               alpha=0.85, label='Annual mean ΔYield')

        # Sen's slope line through the data
        x_centered = valid_years - valid_years.mean()
        sen_line   = mk.intercept + mk.slope * np.arange(len(valid_years))
        ax.plot(valid_years, sen_line, color='black', linewidth=1.8,
                linestyle='--', label=f"Sen's slope: {mk.slope:.4f} kg/ha/yr")

        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Regional Mean ΔYield (kg/ha)', fontsize=11)

        sig_str = f'p = {mk.p:.3f} {"★ significant" if mk.p < 0.05 else "(not significant)"}'
        ax.set_title(
            f'{label} — Temporal Trend in Regional Mean ΔYield\n'
            f'τ = {mk.Tau:.3f}, {sig_str}',
            fontsize=12, fontweight='bold'
        )
        ax.legend(fontsize=10)
        ax.set_xticks(YEARS)
        ax.set_xticklabels(YEARS, rotation=45, ha='right', fontsize=8)
        plt.tight_layout()
        fig.savefig(f'{OUT_ROOT}/{tag}_mk_timeseries.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  ✓ {label}: trend={mk.trend}, τ={mk.Tau:.3f}, p={mk.p:.4f}')

    # ── Summary table ──────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv(f'{OUT_ROOT}/mk_trend_results.csv', index=False)
    print('\n', df[['crop', 'tau', 'p_value', 'slope_sen', 'trend', 'significant']].to_string(index=False))

    # ── Summary bar chart: Kendall's tau for all crops ─────────────────────────
    df_sorted = df.sort_values('tau')
    colors    = ['#2166AC' if s else '#AAAAAA' for s in df_sorted['significant']]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df_sorted['crop'], df_sorted['tau'],
                   color=colors, edgecolor='white')
    ax.axvline(0, color='black', linewidth=0.8)

    # Annotate p-values
    for bar, (_, row) in zip(bars, df_sorted.iterrows()):
        x = bar.get_width()
        ax.text(x + 0.005 * np.sign(x), bar.get_y() + bar.get_height() / 2,
                f'p={row["p_value"]:.3f}{"★" if row["significant"] else ""}',
                va='center', fontsize=9,
                ha='left' if x >= 0 else 'right')

    ax.set_xlabel("Kendall's τ", fontsize=12)
    ax.set_title(
        'Mann-Kendall Trend in Regional Mean ΔYield (1999–2018)\n'
        'Blue = significant (p < 0.05), Grey = not significant',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    fig.savefig(f'{OUT_ROOT}/mk_trend_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n✓ All outputs saved to: {OUT_ROOT}/')


if __name__ == '__main__':
    mask = load_mask()
    run(mask)