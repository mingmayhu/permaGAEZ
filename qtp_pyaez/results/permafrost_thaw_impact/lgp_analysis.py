"""
LGP and fc2 Pathway Analysis — All 10 Crops
============================================
Isolates the two permafrost thaw mechanisms:
  1. LGP pathway  : SHAW ETa → LGP difference (thaw vs no-thaw)
  2. fc2 pathway  : SHAW soil moisture → moisture reduction factor difference

LGP is crop-independent and computed once.
fc2 is computed per crop (max across varieties, consistent with combine_crop_maps).

For each pixel, computes:
  - Mean ΔLGP       (thaw minus no-thaw, 1999-2018) — shared across crops
  - Mean Δfc2       (thaw minus no-thaw, 1999-2018, max across varieties)
  - Mean precip     (1999-2018 annual total) — shared across crops
  - Mean ΔSuitability (loaded from pre-computed raster, per crop)

Statistics per crop:
  - Descriptive statistics
  - Spearman correlations
  - Mean ΔLGP and Δfc2 by precipitation tercile
  - Mean ΔLGP and Δfc2 by ΔSuitability sign
  - OLS regression of ΔSuit on ΔLGP and Δfc2
  - Pathway dominance (univariate R²)

Summary table across all crops at end.

Outputs:
  ./results/permafrost_thaw_impact/pathway_analysis/all_crops_pathway_stats.txt
  ./results/permafrost_thaw_impact/pathway_analysis/{crop}_pathway_data.csv

Working directory: /Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

try:
    from osgeo import gdal
except ImportError:
    import gdal

# =============================================================================
# CONFIGURATION
# =============================================================================

WORK_DIR = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
os.chdir(WORK_DIR)

YEARS      = list(range(1999, 2019))
MASK_VALUE = 0

CROPS = [
    {
        'output_tag': 'combined_winter_barley',
        'varieties':  ['winter_barley_59', 'winter_barley_60',
                       'winter_barley_61', 'winter_barley_62'],
    },
    {
        'output_tag': 'combined_spring_barley',
        'varieties':  ['spring_barley_63', 'spring_barley_64',
                       'spring_barley_65', 'spring_barley_66'],
    },
    {
        'output_tag': 'combined_winter_wheat',
        'varieties':  ['winter_wheat_1', 'winter_wheat_2',
                       'winter_wheat_3', 'winter_wheat_4'],
    },
    {
        'output_tag': 'combined_spring_wheat',
        'varieties':  ['spring_wheat_5', 'spring_wheat_6', 'spring_wheat_7',
                       'spring_wheat_8', 'spring_wheat_9'],
    },
    {
        'output_tag': 'combined_silage_maize',
        'varieties':  ['silage_maize_53', 'silage_maize_54', 'silage_maize_55',
                       'silage_maize_56', 'silage_maize_57', 'silage_maize_58'],
    },
    {
        'output_tag': 'combined_white_potato',
        'varieties':  ['white_potato_135', 'white_potato_136', 'white_potato_137',
                       'white_potato_138', 'white_potato_139', 'white_potato_140',
                       'white_potato_141'],
    },
    {
        'output_tag': 'combined_oat',
        'varieties':  ['spring_oat_128', 'spring_oat_129', 'spring_oat_130'],
    },
    {
        'output_tag': 'combined_dry_pea',
        'varieties':  ['dry_pea_189', 'dry_pea_190', 'dry_pea_191'],
    },
    {
        'output_tag': 'combined_winter_rape',
        'varieties':  ['winter_rape_216', 'winter_rape_217',
                       'winter_rape_218', 'winter_rape_219'],
    },
    {
        'output_tag': 'combined_spring_rape',
        'varieties':  ['spring_rape_220', 'spring_rape_221',
                       'spring_rape_222', 'spring_rape_223'],
    },
]

MASK_PATH   = r'./data_input/qilian_mask_new.tif'
DELTA_SUIT_DIR = (r'./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/'
                  r'5_spatial/1_mean_delta')
PRECIP_DIR  = r'./data_input/climate_yearly'
OUT_DIR     = r'./results/permafrost_thaw_impact/pathway_analysis'
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# HELPERS
# =============================================================================

def read_tif(path):
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(f"Cannot open: {path}")
    arr = ds.ReadAsArray().astype(float)
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr

def load_mask():
    raw = gdal.Open(MASK_PATH).ReadAsArray()
    return (raw != MASK_VALUE) & (raw > 0)

def log(msg, fh=None):
    print(msg)
    if fh:
        fh.write(msg + '\n')

def standardise(x):
    return (x - np.mean(x)) / np.std(x)

def spearman(x, y):
    r, p = stats.spearmanr(x, y)
    return r, p

def pearson_r2(x, y):
    r, p = stats.pearsonr(x, y)
    return r, r**2, p

# =============================================================================
# STEP 1 — LOAD MASK
# =============================================================================

print("Loading mask …")
mask = load_mask()

# =============================================================================
# STEP 2 — COMPUTE MEAN ΔLGP ONCE (crop-independent)
# =============================================================================

print("Computing mean ΔLGP (once, shared across all crops) …")
lgp_diff_stack = []

for year in YEARS:
    thaw_path   = f'./data_output/module1/{year}/LGP New.tif'
    nothaw_path = f'./data_output/module1_nothaw/{year}/LGP New.tif'

    if not os.path.exists(thaw_path):
        print(f"  ⚠ Missing thaw LGP {year} — skipping.")
        continue
    if not os.path.exists(nothaw_path):
        print(f"  ⚠ Missing no-thaw LGP {year} — skipping.")
        continue

    diff = read_tif(thaw_path) - read_tif(nothaw_path)
    diff[~mask] = np.nan
    lgp_diff_stack.append(diff)

mean_delta_lgp = np.nanmean(np.stack(lgp_diff_stack, axis=0), axis=0)
print(f"  LGP years loaded: {len(lgp_diff_stack)}")

# =============================================================================
# STEP 3 — COMPUTE MEAN ANNUAL PRECIPITATION ONCE (crop-independent)
# =============================================================================

print("Computing mean annual precipitation (once, shared across all crops) …")
precip_stack = []

for year in YEARS:
    path = f'{PRECIP_DIR}/{year}/Precip.npy'
    if not os.path.exists(path):
        print(f"  ⚠ Missing precip {year} — skipping.")
        continue
    p = np.load(path).astype(float)           # (rows, cols, 365)
    annual = np.sum(p, axis=2)
    annual[~mask] = np.nan
    precip_stack.append(annual)

mean_precip = np.nanmean(np.stack(precip_stack, axis=0), axis=0)
print(f"  Precip years loaded: {len(precip_stack)}")

# =============================================================================
# MAIN LOOP — PER CROP
# =============================================================================

# Collector for summary table
summary_rows = []

out_path = f'{OUT_DIR}/all_crops_pathway_stats.txt'

with open(out_path, 'w') as fh:

    log("=" * 70, fh)
    log("ALL CROPS — LGP AND FC2 PATHWAY ANALYSIS (1999-2018)", fh)
    log("=" * 70, fh)
    log("", fh)

    for crop in CROPS:
        tag       = crop['output_tag']
        varieties = crop['varieties']

        log("=" * 70, fh)
        log(f"CROP: {tag}", fh)
        log("=" * 70, fh)

        # ------------------------------------------------------------------
        # Load delta suitability
        # ------------------------------------------------------------------
        ds_path = f'{DELTA_SUIT_DIR}/{tag}_mean_delta_suit.tif'
        if not os.path.exists(ds_path):
            log(f"  ⚠ Missing delta suit raster: {ds_path} — skipping crop.", fh)
            continue
        delta_suit = read_tif(ds_path)
        delta_suit[~mask] = np.nan

        # ------------------------------------------------------------------
        # Compute mean Δfc2 for this crop
        # ------------------------------------------------------------------
        log(f"  Computing mean Δfc2 for {tag} …", fh)
        fc2_diff_stack = []

        for year in YEARS:
            fc2_thaw_vars   = []
            fc2_nothaw_vars = []

            for variety in varieties:
                tp = f'./data_output/module2/{variety}/{year}/fc2_rain.tif'
                np_ = f'./data_output/module2_nothaw/{variety}/{year}/fc2_rain.tif'

                if not os.path.exists(tp):
                    continue
                if not os.path.exists(np_):
                    continue

                fc2_t = read_tif(tp)
                fc2_n = read_tif(np_)
                fc2_t[~mask] = np.nan
                fc2_n[~mask] = np.nan
                fc2_thaw_vars.append(fc2_t)
                fc2_nothaw_vars.append(fc2_n)

            if not fc2_thaw_vars:
                continue

            fc2_thaw_max   = np.nanmax(np.stack(fc2_thaw_vars,   axis=0), axis=0)
            fc2_nothaw_max = np.nanmax(np.stack(fc2_nothaw_vars, axis=0), axis=0)
            diff = fc2_thaw_max - fc2_nothaw_max
            diff[~mask] = np.nan
            fc2_diff_stack.append(diff)

        if not fc2_diff_stack:
            log(f"  ⚠ No fc2 data found for {tag} — skipping crop.", fh)
            continue

        mean_delta_fc2 = np.nanmean(np.stack(fc2_diff_stack, axis=0), axis=0)
        log(f"  fc2 years loaded: {len(fc2_diff_stack)}", fh)

        # ------------------------------------------------------------------
        # Flatten to valid pixels
        # ------------------------------------------------------------------
        valid = (
            mask &
            np.isfinite(delta_suit) &
            np.isfinite(mean_delta_lgp) &
            np.isfinite(mean_delta_fc2) &
            np.isfinite(mean_precip)
        )

        ds_flat   = delta_suit[valid]
        lgp_flat  = mean_delta_lgp[valid]
        fc2_flat  = mean_delta_fc2[valid]
        prec_flat = mean_precip[valid]
        n_pixels  = len(ds_flat)

        log(f"  Valid pixels: {n_pixels}", fh)

        # Save pixel data
        pd.DataFrame({
            'delta_suit':  ds_flat,
            'delta_lgp':   lgp_flat,
            'delta_fc2':   fc2_flat,
            'mean_precip': prec_flat,
        }).to_csv(f'{OUT_DIR}/{tag}_pathway_data.csv', index=False)

        # ------------------------------------------------------------------
        # Descriptive statistics
        # ------------------------------------------------------------------
        log("", fh)
        log("-" * 70, fh)
        log("DESCRIPTIVE STATISTICS", fh)
        log("-" * 70, fh)

        for label, arr in [
            ('Mean ΔSuitability (class units)', ds_flat),
            ('Mean ΔLGP (days)',                lgp_flat),
            ('Mean Δfc2 (0-1)',                 fc2_flat),
            ('Mean Annual Precipitation (mm)',  prec_flat),
        ]:
            log(f"\n  {label}:", fh)
            log(f"    Mean   : {np.mean(arr):.4f}", fh)
            log(f"    Median : {np.median(arr):.4f}", fh)
            log(f"    Std    : {np.std(arr):.4f}", fh)
            log(f"    Min    : {np.min(arr):.4f}", fh)
            log(f"    Max    : {np.max(arr):.4f}", fh)
            log(f"    % > 0  : {100*np.mean(arr > 0):.1f}%", fh)
            log(f"    % < 0  : {100*np.mean(arr < 0):.1f}%", fh)

        # ------------------------------------------------------------------
        # Spearman correlations
        # ------------------------------------------------------------------
        log("", fh)
        log("-" * 70, fh)
        log("SPEARMAN CORRELATIONS", fh)
        log("-" * 70, fh)

        pairs = [
            ('ΔLGP',          lgp_flat,  'ΔSuitability',  ds_flat),
            ('Δfc2',          fc2_flat,  'ΔSuitability',  ds_flat),
            ('ΔLGP',          lgp_flat,  'Precipitation', prec_flat),
            ('Δfc2',          fc2_flat,  'Precipitation', prec_flat),
            ('ΔLGP',          lgp_flat,  'Δfc2',          fc2_flat),
            ('Precipitation', prec_flat, 'ΔSuitability',  ds_flat),
        ]

        for x_lbl, x, y_lbl, y in pairs:
            r, p = spearman(x, y)
            sig = '*' if p < 0.05 else ''
            log(f"  {x_lbl:20s} vs {y_lbl:20s}  r = {r:7.4f}  p = {p:.4f} {sig}", fh)

        # ------------------------------------------------------------------
        # Precipitation terciles
        # ------------------------------------------------------------------
        log("", fh)
        log("-" * 70, fh)
        log("MEAN ΔLGP AND Δfc2 BY PRECIPITATION TERCILE", fh)
        log("-" * 70, fh)

        t33, t67 = np.percentile(prec_flat, [33, 67])
        log(f"  Tercile thresholds: low < {t33:.1f} mm, "
            f"mid < {t67:.1f} mm, high >= {t67:.1f} mm", fh)
        log("", fh)

        tercile_labels = ['Low precip', 'Mid precip', 'High precip']
        tercile_masks  = [
            prec_flat < t33,
            (prec_flat >= t33) & (prec_flat < t67),
            prec_flat >= t67,
        ]

        log(f"  {'Group':<15} {'N':>6} {'Mean ΔLGP':>12} {'Mean Δfc2':>12} "
            f"{'Mean ΔSuit':>12} {'% ΔLGP>0':>10} {'% Δfc2>0':>10}", fh)
        log(f"  {'-'*15} {'-'*6} {'-'*12} {'-'*12} "
            f"{'-'*12} {'-'*10} {'-'*10}", fh)

        for lbl, tm in zip(tercile_labels, tercile_masks):
            n     = tm.sum()
            m_lgp = np.mean(lgp_flat[tm])
            m_fc2 = np.mean(fc2_flat[tm])
            m_ds  = np.mean(ds_flat[tm])
            pct_lgp = 100 * np.mean(lgp_flat[tm] > 0)
            pct_fc2 = 100 * np.mean(fc2_flat[tm] > 0)
            log(f"  {lbl:<15} {n:>6} {m_lgp:>12.4f} {m_fc2:>12.4f} "
                f"{m_ds:>12.4f} {pct_lgp:>9.1f}% {pct_fc2:>9.1f}%", fh)

        log("", fh)
        log("  Kruskal-Wallis across precipitation terciles:", fh)
        for lbl, arr in [('ΔLGP', lgp_flat), ('Δfc2', fc2_flat), ('ΔSuit', ds_flat)]:
            h, p = stats.kruskal(*[arr[tm] for tm in tercile_masks])
            sig = '*' if p < 0.05 else ''
            log(f"    {lbl}: H = {h:.4f}, p = {p:.4f} {sig}", fh)

        # ------------------------------------------------------------------
        # ΔSuitability direction groups
        # ------------------------------------------------------------------
        log("", fh)
        log("-" * 70, fh)
        log("MEAN ΔLGP AND Δfc2 BY ΔSUITABILITY DIRECTION", fh)
        log("-" * 70, fh)

        suit_pos  = ds_flat > 0
        suit_zero = ds_flat == 0
        suit_neg  = ds_flat < 0

        log(f"  {'Group':<20} {'N':>6} {'Mean ΔLGP':>12} {'Mean Δfc2':>12} "
            f"{'Mean Precip':>13} {'% ΔLGP>0':>10} {'% Δfc2>0':>10}", fh)
        log(f"  {'-'*20} {'-'*6} {'-'*12} {'-'*12} "
            f"{'-'*13} {'-'*10} {'-'*10}", fh)

        for lbl, sm in [
            ('ΔSuit > 0 (pos)',  suit_pos),
            ('ΔSuit = 0 (none)', suit_zero),
            ('ΔSuit < 0 (neg)',  suit_neg),
        ]:
            if sm.sum() == 0:
                log(f"  {lbl:<20} {'0':>6} {'—':>12} {'—':>12} "
                    f"{'—':>13} {'—':>10} {'—':>10}", fh)
                continue
            n     = sm.sum()
            m_lgp = np.mean(lgp_flat[sm])
            m_fc2 = np.mean(fc2_flat[sm])
            m_p   = np.mean(prec_flat[sm])
            pct_lgp = 100 * np.mean(lgp_flat[sm] > 0)
            pct_fc2 = 100 * np.mean(fc2_flat[sm] > 0)
            log(f"  {lbl:<20} {n:>6} {m_lgp:>12.4f} {m_fc2:>12.4f} "
                f"{m_p:>13.1f} {pct_lgp:>9.1f}% {pct_fc2:>9.1f}%", fh)

        log("", fh)
        log("  Mann-Whitney U: positive vs negative ΔSuit pixels:", fh)
        if suit_pos.sum() > 0 and suit_neg.sum() > 0:
            for lbl, arr in [
                ('ΔLGP',          lgp_flat),
                ('Δfc2',          fc2_flat),
                ('Precipitation', prec_flat),
            ]:
                u, p = stats.mannwhitneyu(
                    arr[suit_pos], arr[suit_neg], alternative='two-sided')
                sig = '*' if p < 0.05 else ''
                log(f"    {lbl}: U = {u:.1f}, p = {p:.4f} {sig}", fh)
        else:
            log("    Insufficient data in one group.", fh)

        # ------------------------------------------------------------------
        # OLS regression
        # ------------------------------------------------------------------
        log("", fh)
        log("-" * 70, fh)
        log("OLS REGRESSION: ΔSuitability ~ ΔLGP + Δfc2", fh)
        log("-" * 70, fh)

        lgp_std = standardise(lgp_flat)
        fc2_std = standardise(fc2_flat)
        ds_std  = standardise(ds_flat)

        X    = np.column_stack([np.ones(n_pixels), lgp_std, fc2_std])
        beta, _, _, _ = np.linalg.lstsq(X, ds_std, rcond=None)

        y_pred = X @ beta
        r2_ols = 1 - np.sum((ds_std - y_pred)**2) / np.sum((ds_std - np.mean(ds_std))**2)

        log(f"  Standardised beta — ΔLGP : {beta[1]:.4f}", fh)
        log(f"  Standardised beta — Δfc2 : {beta[2]:.4f}", fh)
        log(f"  R²                       : {r2_ols:.4f}", fh)

        # ------------------------------------------------------------------
        # Pathway dominance
        # ------------------------------------------------------------------
        log("", fh)
        log("-" * 70, fh)
        log("PATHWAY DOMINANCE (univariate R²)", fh)
        log("-" * 70, fh)

        r_lgp, r2_lgp, p_lgp = pearson_r2(lgp_flat, ds_flat)
        r_fc2, r2_fc2, p_fc2 = pearson_r2(fc2_flat, ds_flat)

        sig_lgp = '*' if p_lgp < 0.05 else ''
        sig_fc2 = '*' if p_fc2 < 0.05 else ''

        log(f"  ΔLGP: r = {r_lgp:.4f}, R² = {r2_lgp:.4f}, p = {p_lgp:.4f} {sig_lgp}", fh)
        log(f"  Δfc2: r = {r_fc2:.4f}, R² = {r2_fc2:.4f}, p = {p_fc2:.4f} {sig_fc2}", fh)
        log("", fh)

        # ------------------------------------------------------------------
        # Collect summary row
        # ------------------------------------------------------------------
        prec_neg = np.mean(prec_flat[suit_neg]) if suit_neg.sum() > 0 else np.nan
        prec_pos = np.mean(prec_flat[suit_pos]) if suit_pos.sum() > 0 else np.nan
        fc2_neg  = np.mean(fc2_flat[suit_neg])  if suit_neg.sum() > 0 else np.nan
        fc2_pos  = np.mean(fc2_flat[suit_pos])  if suit_pos.sum() > 0 else np.nan

        # spearman fc2 vs suit for summary
        r_fc2_suit, p_fc2_suit = spearman(fc2_flat, ds_flat)
        r_lgp_suit, p_lgp_suit = spearman(lgp_flat, ds_flat)

        summary_rows.append({
            'Crop':               tag.replace('combined_', '').replace('_', ' ').title(),
            'N pixels':           n_pixels,
            'N pos ΔSuit':        int(suit_pos.sum()),
            'N neg ΔSuit':        int(suit_neg.sum()),
            'Mean ΔLGP':          round(float(np.mean(lgp_flat)), 3),
            'Mean Δfc2':          round(float(np.mean(fc2_flat)), 4),
            'ΔLGP r (vs ΔSuit)':  round(float(r_lgp_suit), 4),
            'ΔLGP p (vs ΔSuit)':  round(float(p_lgp_suit), 4),
            'Δfc2 r (vs ΔSuit)':  round(float(r_fc2_suit), 4),
            'Δfc2 p (vs ΔSuit)':  round(float(p_fc2_suit), 4),
            'R² (ΔLGP)':          round(float(r2_lgp), 4),
            'R² (Δfc2)':          round(float(r2_fc2), 4),
            'R² (OLS both)':      round(float(r2_ols), 4),
            'Beta ΔLGP (std)':    round(float(beta[1]), 4),
            'Beta Δfc2 (std)':    round(float(beta[2]), 4),
            'Mean Precip pos':    round(float(prec_pos), 1) if not np.isnan(prec_pos) else 'n/a',
            'Mean Precip neg':    round(float(prec_neg), 1) if not np.isnan(prec_neg) else 'n/a',
            'Mean Δfc2 pos':      round(float(fc2_pos),  4) if not np.isnan(fc2_pos)  else 'n/a',
            'Mean Δfc2 neg':      round(float(fc2_neg),  4) if not np.isnan(fc2_neg)  else 'n/a',
        })

    # ==========================================================================
    # SUMMARY TABLE
    # ==========================================================================

    log("", fh)
    log("=" * 70, fh)
    log("SUMMARY TABLE — KEY STATS ACROSS ALL CROPS", fh)
    log("=" * 70, fh)
    log("", fh)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(f'{OUT_DIR}/all_crops_pathway_summary.csv', index=False)
        log("  Full summary saved to all_crops_pathway_summary.csv", fh)
        log("", fh)

        # Print compact version to text file
        cols_to_print = [
            'Crop', 'N pixels', 'N pos ΔSuit', 'N neg ΔSuit',
            'R² (ΔLGP)', 'R² (Δfc2)', 'R² (OLS both)',
            'Beta ΔLGP (std)', 'Beta Δfc2 (std)',
            'Mean Precip pos', 'Mean Precip neg',
            'Mean Δfc2 pos', 'Mean Δfc2 neg',
        ]

        # Header
        header = f"  {'Crop':<18} {'N':>6} {'N+':>5} {'N-':>5} " \
                 f"{'R²_LGP':>8} {'R²_fc2':>8} {'R²_OLS':>8} " \
                 f"{'β_LGP':>8} {'β_fc2':>8} " \
                 f"{'P+':>7} {'P-':>7} {'fc2+':>7} {'fc2-':>7}"
        log(header, fh)
        log("  " + "-" * (len(header) - 2), fh)

        for row in summary_rows:
            crop_short = row['Crop'][:18]
            log(
                f"  {crop_short:<18} "
                f"{row['N pixels']:>6} "
                f"{row['N pos ΔSuit']:>5} "
                f"{row['N neg ΔSuit']:>5} "
                f"{row['R² (ΔLGP)']:>8.4f} "
                f"{row['R² (Δfc2)']:>8.4f} "
                f"{row['R² (OLS both)']:>8.4f} "
                f"{row['Beta ΔLGP (std)']:>8.4f} "
                f"{row['Beta Δfc2 (std)']:>8.4f} "
                f"{str(row['Mean Precip pos']):>7} "
                f"{str(row['Mean Precip neg']):>7} "
                f"{str(row['Mean Δfc2 pos']):>7} "
                f"{str(row['Mean Δfc2 neg']):>7}",
                fh
            )

        log("", fh)
        log("  Column key:", fh)
        log("    N+      = pixels with positive ΔSuitability", fh)
        log("    N-      = pixels with negative ΔSuitability", fh)
        log("    R²_LGP  = univariate R² of ΔLGP vs ΔSuit", fh)
        log("    R²_fc2  = univariate R² of Δfc2 vs ΔSuit", fh)
        log("    R²_OLS  = OLS R² with both predictors", fh)
        log("    β_LGP   = standardised OLS beta for ΔLGP", fh)
        log("    β_fc2   = standardised OLS beta for Δfc2", fh)
        log("    P+/P-   = mean precip in pos/neg ΔSuit pixels (mm)", fh)
        log("    fc2+/-  = mean Δfc2 in pos/neg ΔSuit pixels", fh)

    log("", fh)
    log("=" * 70, fh)
    log("END OF ANALYSIS", fh)
    log("=" * 70, fh)

print(f"\nStats saved to {out_path}")
print(f"Summary CSV saved to {OUT_DIR}/all_crops_pathway_summary.csv")
print("Done.")