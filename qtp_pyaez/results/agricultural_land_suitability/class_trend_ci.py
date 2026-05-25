"""
Per-class percentage trend CI table
=====================================
Reads the pct_class CSV (pasted inline or from file), runs MK + bootstrap CI
on each class series, and exports a summary table.

Output: ./results/agricultural_land_suitability/outputs/csv/class_trend_ci.csv
"""

import os
import numpy as np
import pandas as pd
from pymannkendall import original_test as mk_test
from io import StringIO

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
CSV_PATH = r'./results/agricultural_land_suitability/outputs/csv/overall_class_area_km2.csv'
OUT_PATH = r'./results/agricultural_land_suitability/outputs/csv/class_trend_ci.csv'

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

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, index_col='Year')

# ── Compute ───────────────────────────────────────────────────────────────────
rows = []
for col in df.columns:
    series = df[col].values
    mk     = run_mk(series)
    ci     = bootstrap_sen_ci(series)
    if mk is None:
        continue
    rows.append({
        'class':       col,
        'slope':       mk['slope'],
        'ci_lo':       ci[0],
        'ci_hi':       ci[1],
        'tau':         mk['tau'],
        'p':           mk['p'],
        'significant': mk['significant'],
        'trend':       mk['trend'],
    })
    print(f"  {col}: slope={mk['slope']:.6f} % yr⁻¹  "
          f"95% CI=[{ci[0]:.6f}, {ci[1]:.6f}]  "
          f"τ={mk['tau']:.3f}  p={mk['p']:.4f}")

# ── Export ────────────────────────────────────────────────────────────────────
df_out = pd.DataFrame(rows)
df_out.to_csv(OUT_PATH, index=False)
print(f'\nSaved to {OUT_PATH}')