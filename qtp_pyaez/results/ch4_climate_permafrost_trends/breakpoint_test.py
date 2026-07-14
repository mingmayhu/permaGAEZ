"""
Structural Breakpoint Analysis — Climate and Permafrost Series (1979–2018)
==========================================================================
Tests whether temperature, precipitation, ALT, and ASM show a statistically
significant structural break, and whether that break falls near 1999,
validating the choice of 1999 as the no-thaw divergence year.

Method:
  Step 1 — Binary segmentation (ruptures Binseg) identifies the single most
            likely break year endogenously, without assuming 1999 in advance.
  Step 2 — Chow F-test at the identified break year tests whether the break
            is statistically significant (H0: no structural break).
  Step 3 — CUSUM test (breaks_cusumolsresid) provides an independent test
            for any parameter instability across the full series.
  Step 4 — Hansen test (breaks_hansen) additionally tests for coefficient
            instability in the OLS trend model.

Interpretation:
  If Binseg identifies a break near 1999 AND Chow p < 0.05, the choice of
  1999 as the divergence year is empirically supported by the observed data.

Data loading:
  Reads directly from the same .npy files used by chapter4_figures.py —
  no CSV needed.

Output: ./results/breakpoint/breakpoint_results.csv
        ./results/breakpoint/breakpoint_figure.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from scipy.stats import f as f_dist
import statsmodels.api as sm
import statsmodels.stats.diagnostic as smd
import ruptures as rpt
import warnings
warnings.filterwarnings('ignore')
from osgeo import gdal

# ── Configuration ──────────────────────────────────────────────────────────────
WORK_DIR        = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH       = r'./data_input/qilian_mask_new.tif'
PERM_MAP_PATH   = r'./data_input/permafrost_qilian.tif'
CLIM_DIR        = r'./data_input/climate_yearly'
PERM_DIR        = r'./data_input/permafrost_yearly'
OUT_DIR         = r'./results/breakpoint'
BOLD_PATH       = r'/Users/ming-mayhu/Library/Fonts/Helvetica LT 75 Bold.ttf'
DIVERGENCE_YEAR = 1999
START_YEAR      = 1979
END_YEAR        = 2018
ALPHA           = 0.05
MIN_SEG         = 6       # minimum segment length each side of break (years)

os.chdir(WORK_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

bold_fp = fm.FontProperties(fname=BOLD_PATH) if os.path.exists(BOLD_PATH) else None

plt.rcParams.update({
    'font.family':        'Helvetica',
    'font.size':          10,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})

YEARS = list(range(START_YEAR, END_YEAR + 1))

# ── Mask loading (same logic as chapter4_figures.py) ──────────────────────────
def load_mask():
    ds      = gdal.Open(MASK_PATH)
    mask    = ds.GetRasterBand(1).ReadAsArray().astype(bool)
    ds      = None
    ds2     = gdal.Open(PERM_MAP_PATH)
    nd      = ds2.GetRasterBand(1).GetNoDataValue()
    pf      = ds2.GetRasterBand(1).ReadAsArray().astype(float)
    ds2     = None
    if nd is not None:
        pf[pf == nd] = np.nan
    lake    = (pf == 0) | np.isnan(pf)
    return mask & ~lake

def regional_mean(arr, mask):
    valid = mask & np.isfinite(arr)
    return float(np.nanmean(arr[valid])) if valid.any() else np.nan

# ── Build annual time series from .npy files ───────────────────────────────────
def build_series(mask):
    tmean_ann, prec_ann, alt_ann, sm_ann = [], [], [], []

    for year in YEARS:
        # Temperature — mean of (TempMax + TempMin) / 2, averaged across days
        tmax_path = f'{CLIM_DIR}/{year}/TempMax.npy'
        tmin_path = f'{CLIM_DIR}/{year}/TempMin.npy'
        if os.path.exists(tmax_path) and os.path.exists(tmin_path):
            tmax = np.load(tmax_path).astype(float)
            tmin = np.load(tmin_path).astype(float)
            tmean_2d = np.nanmean((tmax + tmin) / 2, axis=2)
            tmean_ann.append(regional_mean(tmean_2d, mask))
        else:
            tmean_ann.append(np.nan)

        # Precipitation — sum across days, then regional mean
        prec_path = f'{CLIM_DIR}/{year}/Precip.npy'
        if os.path.exists(prec_path):
            prec = np.load(prec_path).astype(float)
            prec_2d = np.nansum(prec, axis=2)
            prec_ann.append(regional_mean(prec_2d, mask))
        else:
            prec_ann.append(np.nan)

        # ALT — max across days, then regional mean
        alt_path = f'{PERM_DIR}/{year}/active_layer_depth.npy'
        if os.path.exists(alt_path):
            alt = np.load(alt_path).astype(float)
            alt_2d = np.nanmax(alt, axis=2)
            alt_ann.append(regional_mean(alt_2d, mask))
        else:
            alt_ann.append(np.nan)

        # ASM — mean across days, then regional mean
        sm_path = f'{PERM_DIR}/{year}/avail_soil_moisture.npy'
        if os.path.exists(sm_path):
            sm = np.load(sm_path).astype(float)
            sm_2d = np.nanmean(sm, axis=2)
            sm_ann.append(regional_mean(sm_2d, mask))
        else:
            sm_ann.append(np.nan)

        print(f'  {year}: T={tmean_ann[-1]:.3f}  P={prec_ann[-1]:.1f}'
              f'  ALT={alt_ann[-1]:.4f}  ASM={sm_ann[-1]:.2f}')

    return (np.array(tmean_ann), np.array(prec_ann),
            np.array(alt_ann),   np.array(sm_ann))

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading mask …")
mask = load_mask()
print(f"Valid pixels: {mask.sum()}")

print("\nBuilding annual time series …")
tmean_ann, prec_ann, alt_ann, sm_ann = build_series(mask)

# Save CSV for future use
years_arr = np.array(YEARS)
csv_df = pd.DataFrame({
    'Year':                years_arr,
    'Mean_Temperature_C':  tmean_ann,
    'Total_Precipitation_mm': prec_ann,
    'Max_ALT_m':           alt_ann,
    'Mean_SoilMoisture_mm':sm_ann,
})
csv_path = os.path.join(OUT_DIR, 'climate_permafrost_timeseries.csv')
csv_df.to_csv(csv_path, index=False)
print(f"\n✓ Time series saved to {csv_path}")

VARIABLES = [
    {'key': 'tmean', 'data': tmean_ann, 'label': 'Mean Temperature',       'units': '°C',      'color': '#E15759'},
    {'key': 'prec',  'data': prec_ann,  'label': 'Total Precipitation',    'units': 'mm yr⁻¹', 'color': '#4E79A7'},
    {'key': 'alt',   'data': alt_ann,   'label': 'Active Layer Thickness', 'units': 'm',        'color': '#F28E2B'},
    {'key': 'sm',    'data': sm_ann,    'label': 'Available Soil Moisture','units': 'mm',       'color': '#59A14F'},
]

n       = len(YEARS)
years   = years_arr
t       = np.arange(n, dtype=float)
div_idx = DIVERGENCE_YEAR - START_YEAR   # = 20 for 1999

print(f"\n{n} years ({START_YEAR}–{END_YEAR}), divergence index = {div_idx}")


# ── Core analysis function ─────────────────────────────────────────────────────
def analyse(y, label):
    """
    Run full structural break analysis on annual series y (length n).
    Returns dict of results for CSV and figure.
    """
    res = {'variable': label}

    # ── Step 1: Binary segmentation — locate the single strongest break ────────
    signal     = y.reshape(-1, 1).astype(float)
    binseg     = rpt.Binseg(model='l2').fit(signal)
    raw_breaks = binseg.predict(n_bkps=1)
    # Binseg returns end-of-segment indices; the break point is raw_breaks[0]
    # which is the first index of the *second* segment
    break_idx  = raw_breaks[0]
    break_year = START_YEAR + break_idx

    res['break_index'] = break_idx
    res['break_year']  = break_year
    res['years_from_1999'] = abs(break_year - DIVERGENCE_YEAR)

    print(f"  Binseg break: {break_year} (index {break_idx}, "
          f"{res['years_from_1999']} yr from {DIVERGENCE_YEAR})")

    # ── Step 2: Chow F-test at the identified break ────────────────────────────
    # Tests H0: regression coefficients are identical in both segments
    # H1: at least one coefficient differs — i.e. a structural break exists
    if break_idx <= MIN_SEG or break_idx >= n - MIN_SEG:
        chow_F = np.nan
        chow_p = np.nan
        chow_sig = False
        print(f"  Chow test: break too close to series edge — not computed")
    else:
        X      = sm.add_constant(t)
        rss_r  = sm.OLS(y,            X           ).fit().ssr   # restricted (full series)
        rss_1  = sm.OLS(y[:break_idx], sm.add_constant(t[:break_idx])).fit().ssr
        rss_2  = sm.OLS(y[break_idx:], sm.add_constant(t[break_idx:])).fit().ssr
        k      = X.shape[1]                                      # params per segment (= 2)
        df1    = k
        df2    = n - 2 * k
        chow_F = ((rss_r - (rss_1 + rss_2)) / df1) / ((rss_1 + rss_2) / df2)
        chow_p = 1 - f_dist.cdf(chow_F, df1, df2)
        chow_sig = chow_p < ALPHA
        sig_str  = '★ significant' if chow_sig else 'not significant'
        print(f"  Chow F({df1},{df2}) = {chow_F:.3f},  p = {chow_p:.4f}  [{sig_str}]")

    res['chow_F']   = round(chow_F, 4) if not np.isnan(chow_F) else np.nan
    res['chow_p']   = round(chow_p, 4) if not np.isnan(chow_p) else np.nan
    res['chow_sig'] = chow_sig

    # ── Step 3: CUSUM test — overall parameter instability ────────────────────
    # Tests H0: OLS residuals are stable (no structural change anywhere)
    # Returns: (test statistic, p-value, critical values at 1%, 5%, 10%)
    X         = sm.add_constant(t)
    ols_resid = sm.OLS(y, X).fit().resid
    cusum_stat, cusum_p, cusum_crit = smd.breaks_cusumolsresid(ols_resid)
    cusum_sig = cusum_p < ALPHA
    sig_str   = '★ significant' if cusum_sig else 'not significant'
    print(f"  CUSUM stat = {cusum_stat:.3f},  p = {cusum_p:.4f}  [{sig_str}]")

    res['cusum_stat'] = round(float(cusum_stat), 4)
    res['cusum_p']    = round(float(cusum_p),    4)
    res['cusum_sig']  = cusum_sig

    # ── Step 4: Hansen test — coefficient stability ────────────────────────────
    # Tests H0: OLS coefficients (intercept + slope) are stable over the sample
    # Compares Hansen Lc statistic against tabulated critical values
    ols_model  = sm.OLS(y, X).fit()
    hansen_Lc, hansen_crit = smd.breaks_hansen(ols_model)
    # critical values are at nobs=2,6,15,19 (approx 10%,5%,2.5%,1%)
    # for 2 params the 5% critical value is at index 1 (crit=1.9)
    crit_5pct  = float(hansen_crit['crit'][1])
    hansen_sig = float(hansen_Lc) > crit_5pct
    sig_str    = '★ significant' if hansen_sig else 'not significant'
    print(f"  Hansen Lc = {hansen_Lc:.3f},  5% crit = {crit_5pct:.3f}  [{sig_str}]")

    res['hansen_Lc']   = round(float(hansen_Lc), 4)
    res['hansen_crit_5pct'] = crit_5pct
    res['hansen_sig']  = hansen_sig

    return res


# ── Run analysis ───────────────────────────────────────────────────────────────
all_results = []

for var in VARIABLES:
    y = var['data'].astype(float)
    print(f"\n{'─'*60}")
    print(f"{var['label']} ({var['units']})")
    print(f"{'─'*60}")
    res = analyse(y, var['label'])
    res['units'] = var['units']
    all_results.append(res)


# ── Save results CSV ───────────────────────────────────────────────────────────
rows = []
for r in all_results:
    rows.append({
        'Variable':          r['variable'],
        'Units':             r['units'],
        'Break_year':        r['break_year'],
        'Years_from_1999':   r['years_from_1999'],
        'Chow_F':            r['chow_F'],
        'Chow_p':            r['chow_p'],
        'Chow_significant':  r['chow_sig'],
        'CUSUM_stat':        r['cusum_stat'],
        'CUSUM_p':           r['cusum_p'],
        'CUSUM_significant': r['cusum_sig'],
        'Hansen_Lc':         r['hansen_Lc'],
        'Hansen_crit_5pct':  r['hansen_crit_5pct'],
        'Hansen_significant':r['hansen_sig'],
    })

out_df = pd.DataFrame(rows)
csv_out = os.path.join(OUT_DIR, 'breakpoint_results.csv')
out_df.to_csv(csv_out, index=False)
print(f"\n✓ Results table:")
print(out_df.to_string(index=False))
print(f"\n✓ Saved to {csv_out}")


# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for ax, var, res in zip(axes, VARIABLES, all_results):
    y         = var['data'].astype(float)
    color     = var['color']
    bidx      = res['break_index']
    byear     = res['break_year']
    chow_p    = res['chow_p']
    chow_sig  = res['chow_sig']
    dist      = res['years_from_1999']

    # Raw series
    ax.plot(years, y, color=color, lw=1.5, alpha=0.9, zorder=2)
    ax.scatter(years, y, color=color, s=18, zorder=3, alpha=0.75)

    # OLS trend lines per segment
    if MIN_SEG < bidx < n - MIN_SEG:
        for slc in [slice(0, bidx), slice(bidx, n)]:
            xs = t[slc]; ys = y[slc]
            fit = np.polyfit(xs, ys, 1)
            ax.plot(years[slc], np.polyval(fit, xs),
                    color='black', lw=1.4, ls='--', alpha=0.55, zorder=4)

    # Detected break line
    ax.axvline(byear, color='black', lw=1.6, ls=':', zorder=5)

    # 1999 reference line
    ax.axvline(DIVERGENCE_YEAR, color='#666666', lw=1.0, ls='--', alpha=0.6, zorder=4)

    # Annotation box
    p_str  = f'{chow_p:.3f}' if not np.isnan(chow_p) else 'n/a'
    star   = ' ★' if chow_sig else ''
    annot  = (f"Detected break: {byear}\n"
              f"Distance from 1999: {dist} yr\n"
              f"Chow p = {p_str}{star}")
    ax.text(0.03, 0.97, annot, transform=ax.transAxes,
            va='top', ha='left', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', alpha=0.85))

    # Axis labels
    ax.set_ylabel(f"{var['label']} ({var['units']})", fontsize=9,
                  fontproperties=bold_fp if bold_fp else None)
    ax.set_xlabel('Year', fontsize=9)
    ax.set_xlim(START_YEAR - 0.5, END_YEAR + 0.5)
    ax.tick_params(labelsize=8)

legend_handles = [
    Line2D([0],[0], color='black',   lw=1.6, ls=':',  label='Detected break (Binseg)'),
    Line2D([0],[0], color='#666666', lw=1.0, ls='--', label=f'Divergence year ({DIVERGENCE_YEAR})'),
    Line2D([0],[0], color='black',   lw=1.4, ls='--', alpha=0.55, label='OLS trend per segment'),
]
axes[0].legend(handles=legend_handles, fontsize=7.5, loc='upper left', framealpha=0.8)

fig.suptitle(
    f'Structural Break Detection: Climate and Permafrost Series (1979–2018)\n'
    f'Validating {DIVERGENCE_YEAR} as the no-thaw divergence year',
    fontsize=11, fontproperties=bold_fp if bold_fp else None, y=1.02
)
plt.tight_layout()
fig_out = os.path.join(OUT_DIR, 'breakpoint_figure.png')
fig.savefig(fig_out, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Figure saved to {fig_out}")

from arch.unitroot import ZivotAndrews

for name, series in [('Temperature', tmean_ann), 
                     ('Precipitation', prec_ann),
                     ('ALT', alt_ann), 
                     ('ASM', sm_ann)]:
    za = ZivotAndrews(series, trend='ct')  # ct = constant + trend
    print(f"\n{name}")
    print(f"  Break year:  {START_YEAR + za.base_change_point}")
    print(f"  Statistic:   {za.stat:.3f}")
    print(f"  p-value:     {za.pvalue:.4f}")
    print(f"  Significant: {za.pvalue < 0.05}")