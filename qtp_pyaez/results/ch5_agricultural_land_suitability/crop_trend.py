"""
Per-crop trend summary table — Chapter 5
==========================================
Reads per_crop_mean_suitability.csv and per_crop_area_suitable_km2.csv,
runs MK test + bootstrap CI on each crop series, and exports a summary table.

Output: ./results/agricultural_land_suitability/outputs/csv/crop_trend_summary.csv
"""

import os
import numpy as np
import pandas as pd
from pymannkendall import original_test as mk_test

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
CSV_DIR  = r'./results/agricultural_land_suitability/outputs/csv'
OUT_PATH = r'./results/agricultural_land_suitability/outputs/csv/crop_trend_summary.csv'

ALPHA  = 0.05
N_BOOT = 1000

os.chdir(WORK_DIR)

# ── Helpers ───────────────────────────────────────────────────────────────────
def run_mk(series):
    s     = np.array(series, dtype=float)
    valid = np.isfinite(s)
    if valid.sum() < 4:
        return None
    mk = mk_test(s[valid])
    return {
        'slope':       mk.slope,
        'tau':         mk.Tau,
        'p':           mk.p,
        'significant': mk.p < ALPHA,
        'trend':       mk.trend,
    }

def bootstrap_sen_ci(series, n_boot=N_BOOT, ci=95):
    s         = np.array(series, dtype=float)
    valid_idx = np.where(np.isfinite(s))[0]
    if len(valid_idx) < 4:
        return (np.nan, np.nan)
    s_valid = s[valid_idx]
    slopes  = []
    rng     = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = np.sort(rng.choice(len(s_valid), size=len(s_valid), replace=True))
        slopes.append(mk_test(s_valid[idx]).slope)
    lo = np.percentile(slopes, (100 - ci) / 2)
    hi = np.percentile(slopes, 100 - (100 - ci) / 2)
    return (lo, hi)

# ── Load data ─────────────────────────────────────────────────────────────────
df_mean = pd.read_csv(f'{CSV_DIR}/per_crop_mean_suitability.csv', index_col='Year')
df_area = pd.read_csv(f'{CSV_DIR}/per_crop_area_suitable_km2.csv', index_col='Year')

crops = [c for c in df_mean.columns if c != 'Overall']

# ── Compute ───────────────────────────────────────────────────────────────────
rows = []
for crop in crops + ['Overall']:
    for df, metric, units in [
        (df_mean, 'mean_suitability',    'class yr⁻¹'),
        (df_area, 'area_suitable_km2',   'km² yr⁻¹'),
    ]:
        series = df[crop].values
        mk     = run_mk(series)
        ci     = bootstrap_sen_ci(series)
        if mk is None:
            continue
        rows.append({
            'crop':          crop,
            'metric':        metric,
            'units':         units,
            'slope':         mk['slope'],
            'ci_lo':         ci[0],
            'ci_hi':         ci[1],
            'tau':           mk['tau'],
            'p':             mk['p'],
            'significant':   mk['significant'],
            'trend':         mk['trend'],
        })
    print(f'  {crop} done')

# ── Export ────────────────────────────────────────────────────────────────────
df_out = pd.DataFrame(rows)
df_out.to_csv(OUT_PATH, index=False)
print(f'\nSaved to {OUT_PATH}')
print(df_out.to_string(index=False))