"""
pathway_comparison_analysis.py

Two pathway analyses:

Analysis 1 — Thaw vs No-thaw (1999–2018):
    Δfc2, ΔLGP vs Δsuitability
    Max fc2 across varieties (best moisture conditions)

Analysis 2 — P-GAEZ vs FAO (1979–2018):
    Δfc2, ΔLGP, Δfc4 vs Δsuitability
    Max fc2, fc4 across varieties

For each analysis and each crop:
    - Mean Δfc2, ΔLGP, Δfc4
    - Spearman r vs Δsuitability
    - Univariate R²
    - OLS regression (standardised betas)
    - Mean values by ΔSuit sign (positive / negative)

Outputs:
    results/pathway_comparison/thaw_nothaw_pathway.csv
    results/pathway_comparison/permagaez_fao_pathway.csv
    results/pathway_comparison/pathway_summary.txt
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from osgeo import gdal

# =============================================================================
# CONFIG
# =============================================================================

WORK_DIR = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
os.chdir(WORK_DIR)

YEARS_THAW    = list(range(1999, 2019))
YEARS_FULL    = list(range(1979, 2019))

MASK_PATH     = r'./data_input/permafrost_qilian.tif'
OUT_DIR       = r'./results/pathway_comparison'
os.makedirs(OUT_DIR, exist_ok=True)

CROPS = [
    {'tag': 'combined_winter_barley',  'varieties': ['winter_barley_59',  'winter_barley_60',  'winter_barley_61',  'winter_barley_62']},
    {'tag': 'combined_spring_barley',  'varieties': ['spring_barley_63',  'spring_barley_64',  'spring_barley_65',  'spring_barley_66']},
    {'tag': 'combined_winter_wheat',   'varieties': ['winter_wheat_1',    'winter_wheat_2',    'winter_wheat_3',    'winter_wheat_4']},
    {'tag': 'combined_spring_wheat',   'varieties': ['spring_wheat_5',    'spring_wheat_6',    'spring_wheat_7',    'spring_wheat_8',    'spring_wheat_9']},
    {'tag': 'combined_silage_maize',   'varieties': ['silage_maize_53',   'silage_maize_54',   'silage_maize_55',   'silage_maize_56',   'silage_maize_57',   'silage_maize_58']},
    {'tag': 'combined_white_potato',   'varieties': ['white_potato_135',  'white_potato_136',  'white_potato_137',  'white_potato_138',  'white_potato_139',  'white_potato_140',  'white_potato_141']},
    {'tag': 'combined_oat',            'varieties': ['spring_oat_128',    'spring_oat_129',    'spring_oat_130']},
    {'tag': 'combined_dry_pea',        'varieties': ['dry_pea_189',       'dry_pea_190',       'dry_pea_191']},
    {'tag': 'combined_winter_rape',    'varieties': ['winter_rape_216',   'winter_rape_217',   'winter_rape_218',   'winter_rape_219']},
    {'tag': 'combined_spring_rape',    'varieties': ['spring_rape_220',   'spring_rape_221',   'spring_rape_222',   'spring_rape_223']},
]

# =============================================================================
# HELPERS
# =============================================================================

def read_tif(path):
    ds = gdal.Open(path)
    if ds is None:
        return None
    arr = ds.GetRasterBand(1).ReadAsArray().astype(float)
    nd  = ds.GetRasterBand(1).GetNoDataValue()
    if nd is not None:
        arr[arr == nd] = np.nan
    return arr

def load_valid_mask():
    ds     = gdal.Open(MASK_PATH)
    arr    = ds.GetRasterBand(1).ReadAsArray().astype(float)
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    mask   = arr != 0
    if nodata is not None:
        mask &= arr != nodata
    return mask

def max_across_varieties(paths):
    """Load tifs for each variety and return pixelwise max (best reduction factor)."""
    stack = []
    for p in paths:
        arr = read_tif(p)
        if arr is not None:
            stack.append(arr)
    if not stack:
        return None
    return np.nanmax(np.array(stack), axis=0)

def mean_over_years(path_fn, years):
    """Load tifs for each year via path_fn(year) and return pixelwise mean."""
    stack = []
    for y in years:
        arr = read_tif(path_fn(y))
        if arr is not None:
            stack.append(arr)
    if not stack:
        return None
    return np.nanmean(np.array(stack), axis=0)

def mean_over_years_varieties(path_fn, years, varieties):
    """
    For each year, compute max across varieties, then mean over years.
    path_fn(year, variety) -> path string
    """
    year_means = []
    for y in years:
        paths  = [path_fn(y, v) for v in varieties]
        yr_max = max_across_varieties(paths)
        if yr_max is not None:
            year_means.append(yr_max)
    if not year_means:
        return None
    return np.nanmean(np.array(year_means), axis=0)

def pathway_stats(delta_suit, valid_mask, **predictors):
    """
    Compute Spearman r, R², standardised OLS beta for each predictor vs delta_suit.
    predictors: dict of name -> 2D array
    Returns dict of results.
    """
    # Build valid pixel mask
    px = valid_mask & ~np.isnan(delta_suit)
    for arr in predictors.values():
        px &= ~np.isnan(arr)

    n = px.sum()
    if n < 10:
        return None

    y = delta_suit[px]
    results = {'n': int(n)}

    pred_flat = {}
    for name, arr in predictors.items():
        pred_flat[name] = arr[px]

    # Spearman r and R² per predictor
    for name, x in pred_flat.items():
        r, p   = stats.spearmanr(x, y)
        slope, intercept, r_lin, _, _ = stats.linregress(x, y)
        results[f'mean_{name}']  = float(np.nanmean(x))
        results[f'r_{name}']     = round(r, 4)
        results[f'p_{name}']     = round(p, 4)
        results[f'R2_{name}']    = round(r_lin**2, 4)

    # OLS with all predictors (standardised)
    X = np.column_stack([
        (pred_flat[n] - pred_flat[n].mean()) / (pred_flat[n].std() + 1e-10)
        for n in predictors
    ])
    y_std = (y - y.mean()) / (y.std() + 1e-10)
    betas, _, _, _ = np.linalg.lstsq(
        np.column_stack([np.ones(len(y_std)), X]), y_std, rcond=None
    )
    for i, name in enumerate(predictors):
        results[f'beta_{name}'] = round(float(betas[i+1]), 4)

    # Mean by ΔSuit sign
    pos = y > 0
    neg = y < 0
    for name, x in pred_flat.items():
        results[f'mean_{name}_pos'] = round(float(np.mean(x[pos])) if pos.sum() > 0 else np.nan, 4)
        results[f'mean_{name}_neg'] = round(float(np.mean(x[neg])) if neg.sum() > 0 else np.nan, 4)

    results['n_pos'] = int(pos.sum())
    results['n_neg'] = int(neg.sum())

    return results

def majority_vote_permafrost_npy(years, path_fn):
    """
    Load per-year permafrost npy (0/1) and return majority vote (0/1) per pixel.
    path_fn(year) -> path string
    """
    stack = []
    for y in years:
        p = path_fn(y)
        if not os.path.exists(p):
            continue
        arr = np.load(p).astype(float)
        stack.append(arr)
    if not stack:
        return None
    mean_arr = np.mean(np.array(stack), axis=0)
    return (mean_arr >= 0.5).astype(float)  # majority vote

def majority_vote_permafrost_tif(years, path_fn):
    """
    Load per-year permafrost tif (classes 1-4) and return majority vote
    of permafrost presence (class 1 or 2 = permafrost = 1, else 0).
    path_fn(year) -> path string
    """
    stack = []
    for y in years:
        arr = read_tif(path_fn(y))
        if arr is None:
            continue
        stack.append((arr <= 2).astype(float))  # 1 or 2 = permafrost
    if not stack:
        return None
    mean_arr = np.mean(np.array(stack), axis=0)
    return (mean_arr >= 0.5).astype(float)

def permafrost_stats(delta_suit, pf_diff, valid_mask):
    """
    Analyse delta suitability by permafrost difference category.
    pf_diff: per-pixel difference in permafrost classification
             (-1: FAO/nothaw has permafrost, model doesn't)
             ( 0: both agree)
             (+1: model has permafrost, FAO/nothaw doesn't — rare)
    Returns dict with mean delta suit per category and Kruskal-Wallis H, p.
    """
    from scipy.stats import kruskal, mannwhitneyu
    px = valid_mask & ~np.isnan(delta_suit) & ~np.isnan(pf_diff)

    results = {}
    groups  = {}
    for cat, label in [(-1, 'lost_permafrost'), (0, 'agree'), (1, 'gained_permafrost')]:
        mask_cat = px & (pf_diff == cat)
        n = mask_cat.sum()
        results[f'n_pf_{label}']    = int(n)
        results[f'mean_suit_{label}'] = round(float(np.nanmean(delta_suit[mask_cat])), 4) if n > 0 else np.nan
        if n > 0:
            groups[label] = delta_suit[mask_cat]

    # Kruskal-Wallis across all three groups if enough data
    valid_groups = [v for v in groups.values() if len(v) > 0]
    if len(valid_groups) >= 2:
        h, p = kruskal(*valid_groups)
        results['pf_kruskal_H'] = round(float(h), 4)
        results['pf_kruskal_p'] = round(float(p), 4)
    else:
        results['pf_kruskal_H'] = np.nan
        results['pf_kruskal_p'] = np.nan

    # Mann-Whitney: lost_permafrost vs agree (key comparison)
    if 'lost_permafrost' in groups and 'agree' in groups:
        u, p_mw = mannwhitneyu(groups['lost_permafrost'], groups['agree'], alternative='two-sided')
        results['pf_mw_p'] = round(float(p_mw), 4)
    else:
        results['pf_mw_p'] = np.nan

    return results

# =============================================================================
# ANALYSIS 1 — THAW VS NO-THAW (1999–2018)
# =============================================================================

print("=" * 60)
print("Analysis 1: Thaw vs No-thaw (1999–2018)")
print("=" * 60)

valid_mask   = load_valid_mask()
results_thaw = []

for crop in CROPS:
    tag       = crop['tag']
    varieties = crop['varieties']
    print(f"\n  {tag}")

    # Δfc2: mean over years of (max across varieties thaw - max across varieties nothaw)
    def fc2_thaw_path(y, v):
        return f'./data_output/module2/{v}/{y}/fc2_rain.tif'
    def fc2_nothaw_path(y, v):
        return f'./data_output/module2_nothaw/{v}/{y}/fc2_rain.tif'

    fc2_thaw   = mean_over_years_varieties(fc2_thaw_path,   YEARS_THAW, varieties)
    fc2_nothaw = mean_over_years_varieties(fc2_nothaw_path, YEARS_THAW, varieties)
    if fc2_thaw is None or fc2_nothaw is None:
        print(f"    Missing fc2 data, skipping")
        continue
    delta_fc2 = fc2_thaw - fc2_nothaw

    # ΔLGP: mean over years of (thaw LGP - nothaw LGP)
    lgp_thaw   = mean_over_years(lambda y: f'./data_output/module1/{y}/LGP New.tif',       YEARS_THAW)
    lgp_nothaw = mean_over_years(lambda y: f'./data_output/module1_nothaw/{y}/LGP New.tif', YEARS_THAW)
    if lgp_thaw is None or lgp_nothaw is None:
        print(f"    Missing LGP data, skipping")
        continue
    delta_lgp = lgp_thaw - lgp_nothaw

    # Δsuitability
    suit_path  = f'./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/5_spatial/1_mean_delta/{tag}_mean_delta_suit.tif'
    delta_suit = read_tif(suit_path)
    if delta_suit is None:
        print(f"    Missing delta suit, skipping")
        continue

    # Permafrost difference: thaw minus no-thaw (majority vote over 1999-2018)
    # No-thaw reuses 1979-1998 maps for 1999-2018
    pf_thaw   = majority_vote_permafrost_npy(
        YEARS_THAW, lambda y: f'./data_output/module1/permafrost_maps/permafrost_{y}.npy')
    pf_nothaw = majority_vote_permafrost_npy(
        YEARS_THAW, lambda y: f'./data_output/module1/permafrost_maps/permafrost_{y-20}.npy')
    pf_diff_thaw = None
    if pf_thaw is not None and pf_nothaw is not None:
        pf_diff_thaw = pf_thaw - pf_nothaw  # -1: nothaw has pf, thaw doesn't (thawed out)

    res = pathway_stats(delta_suit, valid_mask, fc2=delta_fc2, lgp=delta_lgp)
    if res is None:
        continue

    # Add permafrost stats
    if pf_diff_thaw is not None:
        pf_res = permafrost_stats(delta_suit, pf_diff_thaw, valid_mask)
        res.update(pf_res)

    res['crop'] = tag
    results_thaw.append(res)

    print(f"    n={res['n']}  r_fc2={res['r_fc2']}  r_lgp={res['r_lgp']}  "
          f"R2_fc2={res['R2_fc2']}  R2_lgp={res['R2_lgp']}  "
          f"beta_fc2={res['beta_fc2']}  beta_lgp={res['beta_lgp']}")

df_thaw = pd.DataFrame(results_thaw)

# --- Overall row: stack all crop delta suitability maps ---
all_suit_thaw  = []
all_fc2_thaw   = []
all_lgp_thaw   = []
for crop in CROPS:
    tag = crop['tag']
    suit_path = f'./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/5_spatial/1_mean_delta/{tag}_mean_delta_suit.tif'
    s = read_tif(suit_path)
    if s is None:
        continue
    # reload fc2 and lgp deltas for this crop
    def _fc2t(y, v): return f'./data_output/module2/{v}/{y}/fc2_rain.tif'
    def _fc2nt(y, v): return f'./data_output/module2_nothaw/{v}/{y}/fc2_rain.tif'
    f2t  = mean_over_years_varieties(_fc2t,  YEARS_THAW, crop['varieties'])
    f2nt = mean_over_years_varieties(_fc2nt, YEARS_THAW, crop['varieties'])
    lt   = mean_over_years(lambda y: f'./data_output/module1/{y}/LGP New.tif',        YEARS_THAW)
    lnt  = mean_over_years(lambda y: f'./data_output/module1_nothaw/{y}/LGP New.tif', YEARS_THAW)
    if any(x is None for x in [f2t, f2nt, lt, lnt]):
        continue
    all_suit_thaw.append(s)
    all_fc2_thaw.append(f2t - f2nt)
    all_lgp_thaw.append(lt - lnt)

if all_suit_thaw:
    overall_suit = np.nanmean(np.array(all_suit_thaw), axis=0)
    overall_fc2  = np.nanmean(np.array(all_fc2_thaw),  axis=0)
    overall_lgp  = np.nanmean(np.array(all_lgp_thaw),  axis=0)
    res_overall  = pathway_stats(overall_suit, valid_mask, fc2=overall_fc2, lgp=overall_lgp)
    if res_overall:
        res_overall['crop'] = 'OVERALL'
        df_thaw = pd.concat([pd.DataFrame([res_overall]), df_thaw], ignore_index=True)

cols_order = ['crop', 'n', 'n_pos', 'n_neg',
              'mean_fc2', 'mean_lgp',
              'r_fc2', 'p_fc2', 'R2_fc2', 'beta_fc2',
              'r_lgp',  'p_lgp',  'R2_lgp',  'beta_lgp',
              'mean_fc2_pos', 'mean_fc2_neg',
              'mean_lgp_pos', 'mean_lgp_neg',
              'n_pf_lost_permafrost', 'n_pf_agree', 'n_pf_gained_permafrost',
              'mean_suit_lost_permafrost', 'mean_suit_agree', 'mean_suit_gained_permafrost',
              'pf_kruskal_H', 'pf_kruskal_p', 'pf_mw_p']
df_thaw = df_thaw[[c for c in cols_order if c in df_thaw.columns]]
df_thaw.to_csv(os.path.join(OUT_DIR, 'thaw_nothaw_pathway.csv'), index=False)
print("\nSaved thaw_nothaw_pathway.csv")

# =============================================================================
# ANALYSIS 2 — P-GAEZ VS FAO (1979–2018)
# =============================================================================

print("\n" + "=" * 60)
print("Analysis 2: P-GAEZ vs FAO (1979–2018)")
print("=" * 60)

results_fao = []

for crop in CROPS:
    tag       = crop['tag']
    varieties = crop['varieties']
    print(f"\n  {tag}")

    # Δfc2
    def fc2_perma_path(y, v):
        return f'./data_output/module2/{v}/{y}/fc2_rain.tif'
    def fc2_fao_path(y, v):
        return f'./data_output/original/module2/{v}/{y}/fc2_rain.tif'

    fc2_perma = mean_over_years_varieties(fc2_perma_path, YEARS_FULL, varieties)
    fc2_fao   = mean_over_years_varieties(fc2_fao_path,   YEARS_FULL, varieties)
    if fc2_perma is None or fc2_fao is None:
        print(f"    Missing fc2 data, skipping")
        continue
    delta_fc2 = fc2_perma - fc2_fao

    # ΔLGP
    lgp_perma = mean_over_years(lambda y: f'./data_output/module1/{y}/LGP New.tif',          YEARS_FULL)
    lgp_fao   = mean_over_years(lambda y: f'./data_output/original/module1/{y}/LGP New.tif', YEARS_FULL)
    if lgp_perma is None or lgp_fao is None:
        print(f"    Missing LGP data, skipping")
        continue
    delta_lgp = lgp_perma - lgp_fao

    # Δfc4
    def fc4_perma_path(y, v):
        return f'./data_output/module4/{v}/{y}/fc4_rain.tif'
    def fc4_fao_path(y, v):
        return f'./data_output/original/module4/{v}/{y}/fc4_rain.tif'

    fc4_perma = mean_over_years_varieties(fc4_perma_path, YEARS_FULL, varieties)
    fc4_fao   = mean_over_years_varieties(fc4_fao_path,   YEARS_FULL, varieties)
    if fc4_perma is None or fc4_fao is None:
        print(f"    Missing fc4 data, skipping")
        continue
    delta_fc4 = fc4_perma - fc4_fao

    # Δsuitability
    suit_path  = f'./results/permafrost_thaw_impact/permafrost_vs_fao/outputs/1_diff_maps/{tag}_diff.tif'
    delta_suit = read_tif(suit_path)
    if delta_suit is None:
        print(f"    Missing delta suit, skipping")
        continue

    # Permafrost difference: P-GAEZ minus FAO (majority vote over 1979-2018)
    # P-GAEZ: 0/1 npy files; FAO: classes 1-4 tif (1 or 2 = permafrost)
    pf_perma = majority_vote_permafrost_npy(
        YEARS_FULL, lambda y: f'./data_output/module1/permafrost_maps/permafrost_{y}.npy')
    pf_fao   = majority_vote_permafrost_tif(
        YEARS_FULL, lambda y: f'./data_output/original/module1/{y}/permafrost.tif')
    pf_diff_fao = None
    if pf_perma is not None and pf_fao is not None:
        pf_diff_fao = pf_perma - pf_fao  # -1: FAO has pf, P-GAEZ doesn't; +1: P-GAEZ has pf, FAO doesn't

    res = pathway_stats(delta_suit, valid_mask, fc2=delta_fc2, lgp=delta_lgp, fc4=delta_fc4)
    if res is None:
        continue

    # Add permafrost stats
    if pf_diff_fao is not None:
        pf_res = permafrost_stats(delta_suit, pf_diff_fao, valid_mask)
        res.update(pf_res)

    res['crop'] = tag
    results_fao.append(res)

    print(f"    n={res['n']}  r_fc2={res['r_fc2']}  r_lgp={res['r_lgp']}  r_fc4={res['r_fc4']}  "
          f"R2_fc2={res['R2_fc2']}  R2_lgp={res['R2_lgp']}  R2_fc4={res['R2_fc4']}  "
          f"beta_fc2={res['beta_fc2']}  beta_lgp={res['beta_lgp']}  beta_fc4={res['beta_fc4']}")

df_fao = pd.DataFrame(results_fao)

# --- Overall row ---
all_suit_fao = []
all_fc2_fao  = []
all_lgp_fao  = []
all_fc4_fao  = []
for crop in CROPS:
    tag = crop['tag']
    suit_path = f'./results/permafrost_thaw_impact/permafrost_vs_fao/outputs/1_diff_maps/{tag}_diff.tif'
    s = read_tif(suit_path)
    if s is None:
        continue
    def _fc2p(y, v): return f'./data_output/module2/{v}/{y}/fc2_rain.tif'
    def _fc2f(y, v): return f'./data_output/original/module2/{v}/{y}/fc2_rain.tif'
    def _fc4p(y, v): return f'./data_output/module4/{v}/{y}/fc4_rain.tif'
    def _fc4f(y, v): return f'./data_output/original/module4/{v}/{y}/fc4_rain.tif'
    f2p  = mean_over_years_varieties(_fc2p, YEARS_FULL, crop['varieties'])
    f2f  = mean_over_years_varieties(_fc2f, YEARS_FULL, crop['varieties'])
    lp   = mean_over_years(lambda y: f'./data_output/module1/{y}/LGP New.tif',          YEARS_FULL)
    lf   = mean_over_years(lambda y: f'./data_output/original/module1/{y}/LGP New.tif', YEARS_FULL)
    f4p  = mean_over_years_varieties(_fc4p, YEARS_FULL, crop['varieties'])
    f4f  = mean_over_years_varieties(_fc4f, YEARS_FULL, crop['varieties'])
    if any(x is None for x in [f2p, f2f, lp, lf, f4p, f4f]):
        continue
    all_suit_fao.append(s)
    all_fc2_fao.append(f2p - f2f)
    all_lgp_fao.append(lp  - lf)
    all_fc4_fao.append(f4p - f4f)

if all_suit_fao:
    overall_suit = np.nanmean(np.array(all_suit_fao), axis=0)
    overall_fc2  = np.nanmean(np.array(all_fc2_fao),  axis=0)
    overall_lgp  = np.nanmean(np.array(all_lgp_fao),  axis=0)
    overall_fc4  = np.nanmean(np.array(all_fc4_fao),  axis=0)
    res_overall  = pathway_stats(overall_suit, valid_mask, fc2=overall_fc2, lgp=overall_lgp, fc4=overall_fc4)
    if res_overall:
        res_overall['crop'] = 'OVERALL'
        df_fao = pd.concat([pd.DataFrame([res_overall]), df_fao], ignore_index=True)

cols_order_fao = ['crop', 'n', 'n_pos', 'n_neg',
                  'mean_fc2', 'mean_lgp', 'mean_fc4',
                  'r_fc2', 'p_fc2', 'R2_fc2', 'beta_fc2',
                  'r_lgp',  'p_lgp',  'R2_lgp',  'beta_lgp',
                  'r_fc4',  'p_fc4',  'R2_fc4',  'beta_fc4',
                  'mean_fc2_pos', 'mean_fc2_neg',
                  'mean_lgp_pos', 'mean_lgp_neg',
                  'mean_fc4_pos', 'mean_fc4_neg',
                  'n_pf_lost_permafrost', 'n_pf_agree', 'n_pf_gained_permafrost',
                  'mean_suit_lost_permafrost', 'mean_suit_agree', 'mean_suit_gained_permafrost',
                  'pf_kruskal_H', 'pf_kruskal_p', 'pf_mw_p']
df_fao = df_fao[[c for c in cols_order_fao if c in df_fao.columns]]
df_fao.to_csv(os.path.join(OUT_DIR, 'permagaez_fao_pathway.csv'), index=False)
print("\nSaved permagaez_fao_pathway.csv")

# =============================================================================
# SUMMARY TXT
# =============================================================================

with open(os.path.join(OUT_DIR, 'pathway_summary.txt'), 'w') as f:
    f.write("PATHWAY ANALYSIS SUMMARY\n")
    f.write("=" * 60 + "\n\n")

    f.write("ANALYSIS 1: Thaw vs No-thaw (1999–2018)\n")
    f.write("-" * 60 + "\n")
    f.write(df_thaw.to_string(index=False))
    f.write("\n\n")

    f.write("ANALYSIS 2: P-GAEZ vs FAO (1979–2018)\n")
    f.write("-" * 60 + "\n")
    f.write(df_fao.to_string(index=False))
    f.write("\n")

print("\nSaved pathway_summary.txt")
print("Done.")
