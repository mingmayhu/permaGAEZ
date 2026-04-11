"""
Comprehensive analysis of permafrost vs original FAO GAEZ yield differences
1979-2018, across multiple crops.

Investigates:
1. Year-by-year pixel differences and trends
2. Whether differences are driven by permafrost classification or soil moisture/AWC
3. Spatial and temporal patterns in the drivers
"""

import numpy as np
from osgeo import gdal
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from collections import Counter

# ─────────────────────────────────────────────
# CROPS TO ANALYZE — edit this list
# ─────────────────────────────────────────────
CROPS = [
    "combined_winter_barley",
    "combined_winter_wheat", 
    "combined_dry_pea",
    "combined_silage_maize",
    # "winter_barley_59",
    # "spring_barley_59",
    # "winter_wheat_56",
    # "spring_wheat_56",
    # "white_potato_25",
    # "dry_pea_09",
    # "oat_63",
    # "winter_rape_41",
    # "spring_rape_41",
    # "silage_maize_02",
]

# ─────────────────────────────────────────────
# FIXED PATHS — adjust BASE if needed
# ─────────────────────────────────────────────
BASE          = "/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez"
PF_PERM_DIR   = f"{BASE}/data_output/module1/permafrost_maps"  # .npy: permafrost_{year}.npy
ORIG_PERM_DIR = f"{BASE}/data_output/original/module1"         # .tif: {year}/permafrost.tif
YEARS         = range(1979, 2019)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_tif(path):
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(f"Cannot open: {path}")
    arr = ds.GetRasterBand(1).ReadAsArray().astype(float)
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr

def load_npy(path):
    if not os.path.exists(path):
        return None
    return np.load(path)

def safe_load(path, fmt):
    try:
        return load_tif(path) if fmt == "tif" else load_npy(path)
    except Exception as e:
        print(f"  WARNING: could not load {path}: {e}")
        return None

def driver_label(shaw_class, af_class, pf_yield):
    """Classify what is driving a difference at a pixel."""
    if pf_yield == 0:
        if shaw_class in [1, 2] and af_class not in [1, 2]:
            return "permafrost_class"   # SHAW flags it, air frost doesn't
        elif shaw_class not in [1, 2] and af_class not in [1, 2]:
            return "soil_moisture_awc"  # neither flags permafrost -> AWC/moisture diff
        elif shaw_class in [1, 2] and af_class in [1, 2]:
            return "both_flagged"       # both flag permafrost but orig has yield
        else:
            return "other"
    else:
        return "yield_change"           # non-zero in both, different values

def mk_test(x):
    """Mann-Kendall test, returns (tau, p)."""
    n = len(x)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(x[j] - x[i])
    var_s = n * (n - 1) * (2 * n + 5) / 18
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    p   = 2 * (1 - stats.norm.cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1))
    return tau, p

# ─────────────────────────────────────────────
# MAIN LOOP OVER CROPS
# ─────────────────────────────────────────────

all_yearly = []  # collects yearly summaries across all crops for cross-crop comparison

for CROP in CROPS:
    print(f"\n{'='*60}")
    print(f"ANALYZING: {CROP}")
    print(f"{'='*60}")

    PF_YIELD_DIR   = f"{BASE}/data_output/final_classification/{CROP}"
    ORIG_YIELD_DIR = f"{BASE}/data_output/original/final_classification/{CROP}"
    OUT_DIR        = f"{BASE}/results_original_analysis/output/{CROP}"
    os.makedirs(OUT_DIR, exist_ok=True)

    records      = []  # one row per changed pixel per year
    yearly       = []  # one row per year
    last_pf_yield = None

    for year in YEARS:
        print(f"  {year}", end="", flush=True)

        pf_yield   = safe_load(f"{PF_YIELD_DIR}/{year}_final_yield_class.tif",   "tif")
        orig_yield = safe_load(f"{ORIG_YIELD_DIR}/{year}_final_yield_class.tif", "tif")

        if pf_yield is None or orig_yield is None:
            print(" [MISSING]", end="")
            yearly.append({"year": year, "n_changed": np.nan, "crop": CROP})
            continue

        last_pf_yield = pf_yield
        diff = pf_yield - orig_yield

        pf_perm   = safe_load(f"{PF_PERM_DIR}/permafrost_{year}.npy",   "npy")
        orig_perm = safe_load(f"{ORIG_PERM_DIR}/{year}/permafrost.tif", "tif")

        changed_mask = (~np.isnan(diff)) & (diff != 0)
        rows, cols   = np.where(changed_mask)

        n_pos    = int(np.sum(diff[changed_mask] > 0))
        n_neg    = int(np.sum(diff[changed_mask] < 0))
        n_zeroed = int(np.sum((diff[changed_mask] < 0) & (pf_yield[changed_mask] == 0)))

        yearly.append({
            "crop":              CROP,
            "year":              year,
            "n_changed":         len(rows),
            "n_pos":             n_pos,
            "n_neg":             n_neg,
            "n_zeroed":          n_zeroed,
            "mean_diff":         float(np.nanmean(diff)),
            "mean_diff_changed": float(np.nanmean(diff[changed_mask])) if len(rows) > 0 else np.nan,
            "sum_diff":          float(np.nansum(diff)),
            "pct_pos":           100 * n_pos / len(rows) if len(rows) > 0 else np.nan,
            "pct_neg":           100 * n_neg / len(rows) if len(rows) > 0 else np.nan,
        })

        for r, c in zip(rows, cols):
            shaw = int(pf_perm[r, c])   if pf_perm   is not None else np.nan
            af   = int(orig_perm[r, c]) if orig_perm  is not None else np.nan
            py   = pf_yield[r, c]
            oy   = orig_yield[r, c]
            drv  = driver_label(shaw, af, py)

            records.append({
                "crop":       CROP,
                "year":       year,
                "row":        r,
                "col":        c,
                "orig_yield": oy,
                "pf_yield":   py,
                "diff":       py - oy,
                "shaw_class": shaw,
                "af_class":   af,
                "driver":     drv,
            })

        print(" ✓")

    # ── per-crop dataframes ──
    df_yearly = pd.DataFrame(yearly)
    df_pixels = pd.DataFrame(records)

    df_yearly.to_csv(f"{OUT_DIR}/yearly_summary.csv",  index=False)
    df_pixels.to_csv(f"{OUT_DIR}/pixel_records.csv",   index=False)

    all_yearly.append(df_yearly)

    # ── per-crop trend analysis ──
    valid   = df_yearly.dropna(subset=["n_changed", "mean_diff"])
    years_v = valid["year"].values

    if len(valid) < 3:
        print(f"  Skipping stats/plots for {CROP} — too few valid years")
        continue

    mk_n  = mk_test(valid["n_changed"].values)
    mk_d  = mk_test(valid["mean_diff"].values)
    mk_p  = mk_test(valid["pct_pos"].values)
    lr_n  = stats.linregress(years_v, valid["n_changed"])
    lr_d  = stats.linregress(years_v, valid["mean_diff"])
    lr_pos = stats.linregress(years_v, valid["pct_pos"])

    print(f"\n  ── Trends for {CROP} ──")
    print(f"  n_changed  OLS {lr_n.slope:+.3f}/yr p={lr_n.pvalue:.4f} | MK τ={mk_n[0]:+.3f} p={mk_n[1]:.4f}")
    print(f"  mean_diff  OLS {lr_d.slope:+.5f}/yr p={lr_d.pvalue:.4f} | MK τ={mk_d[0]:+.3f} p={mk_d[1]:.4f}")
    print(f"  pct_pos    OLS {lr_pos.slope:+.3f}/yr p={lr_pos.pvalue:.4f} | MK τ={mk_p[0]:+.3f} p={mk_p[1]:.4f}")

    # ── driver summary ──
    if len(df_pixels) > 0:
        print(f"\n  ── Drivers for {CROP} ──")
        driver_counts = Counter(df_pixels["driver"])
        total_px = len(df_pixels)
        for d, cnt in driver_counts.most_common():
            print(f"  {d:<25} {cnt:>6} px ({100*cnt/total_px:.1f}%)")
        driver_yearly = df_pixels.groupby(["year", "driver"]).size().unstack(fill_value=0)
        driver_yearly.to_csv(f"{OUT_DIR}/driver_by_year.csv")
    else:
        driver_yearly = pd.DataFrame()

    # ── figures ──
    BLUE   = "#2b6cb0"
    RED    = "#c53030"
    GREEN  = "#276749"
    ORANGE = "#c05621"

    DRIVER_COLORS = {
        "soil_moisture_awc": RED,
        "permafrost_class":  ORANGE,
        "both_flagged":      "gold",
        "yield_change":      GREEN,
        "other":             "grey",
    }

    decades = {
        "1979–88": df_yearly[df_yearly.year <= 1988],
        "1989–98": df_yearly[(df_yearly.year >= 1989) & (df_yearly.year <= 1998)],
        "1999–08": df_yearly[(df_yearly.year >= 1999) & (df_yearly.year <= 2008)],
        "2009–18": df_yearly[df_yearly.year >= 2009],
    }

    fig = plt.figure(figsize=(16, 18))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.38)
    fig.suptitle(f"Permafrost vs Original FAO GAEZ — {CROP} 1979–2018",
                 fontsize=13, fontweight="bold", y=0.98)

    # 1. n_changed
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(years_v, valid["n_changed"], color=BLUE, alpha=0.75, width=0.8)
    ax1.plot(years_v, lr_n.slope * years_v + lr_n.intercept, color=RED, lw=2, ls="--",
             label=f"OLS {lr_n.slope:+.2f} px/yr  p={lr_n.pvalue:.3f}\nMK τ={mk_n[0]:+.3f}  p={mk_n[1]:.3f}")
    ax1.set_title("Changed Pixels per Year", fontweight="bold")
    ax1.set_xlabel("Year"); ax1.set_ylabel("Pixel count")
    ax1.legend(fontsize=8, framealpha=0.7)
    ax1.set_xlim(1978, 2019)

    # 2. mean_diff
    ax2 = fig.add_subplot(gs[0, 1])
    bar_colors = [GREEN if v >= 0 else RED for v in valid["mean_diff"]]
    ax2.bar(years_v, valid["mean_diff"], color=bar_colors, alpha=0.8, width=0.8)
    ax2.plot(years_v, lr_d.slope * years_v + lr_d.intercept, color="black", lw=1.5, ls="--",
             label=f"OLS {lr_d.slope:+.5f}/yr  p={lr_d.pvalue:.3f}\nMK τ={mk_d[0]:+.3f}  p={mk_d[1]:.3f}")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_title("Mean Diff (PF − Original) per Year", fontweight="bold")
    ax2.set_xlabel("Year"); ax2.set_ylabel("Mean yield class difference")
    ax2.legend(fontsize=8, framealpha=0.7)
    ax2.set_xlim(1978, 2019)

    # 3. pos vs neg diverging
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(years_v,  valid["n_pos"], color=GREEN, alpha=0.85, width=0.8, label="PF > Orig")
    ax3.bar(years_v, -valid["n_neg"], color=RED,   alpha=0.85, width=0.8, label="PF < Orig")
    ax3.axhline(0, color="black", lw=0.9)
    ax3b = ax3.twinx()
    ax3b.plot(years_v, valid["pct_pos"], color=ORANGE, lw=1.5, marker="o", ms=3)
    ax3b.plot(years_v, lr_pos.slope * years_v + lr_pos.intercept, color=ORANGE, lw=1.5, ls="--",
              label=f"% pos OLS {lr_pos.slope:+.2f}%/yr  p={lr_pos.pvalue:.3f}")
    ax3b.set_ylabel("% positive pixels", color=ORANGE)
    ax3b.tick_params(axis="y", colors=ORANGE)
    ax3b.set_ylim(0, 100)
    ax3.set_title("Positive vs Negative Changed Pixels", fontweight="bold")
    ax3.set_xlabel("Year"); ax3.set_ylabel("Pixel count")
    lines1, labs1 = ax3.get_legend_handles_labels()
    lines2, labs2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc="upper left", framealpha=0.7)
    ax3.set_xlim(1978, 2019)

    # 4. zeroed vs gained
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.fill_between(years_v, valid["n_zeroed"], alpha=0.4, color=RED,   label="Zeroed (AWC/SM diff)")
    ax4.fill_between(years_v, valid["n_pos"],    alpha=0.4, color=GREEN, label="Gained yield class")
    ax4.plot(years_v, valid["n_zeroed"], color=RED,   lw=1.5)
    ax4.plot(years_v, valid["n_pos"],    color=GREEN, lw=1.5)
    ax4.set_title("Zeroed vs Gained Pixels per Year", fontweight="bold")
    ax4.set_xlabel("Year"); ax4.set_ylabel("Pixel count")
    ax4.legend(fontsize=8, framealpha=0.7)
    ax4.set_xlim(1978, 2019)

    # 5. cumulative sum_diff
    ax5 = fig.add_subplot(gs[2, 0])
    cum_diff = valid["sum_diff"].cumsum().values
    ax5.fill_between(years_v, cum_diff, 0, where=cum_diff >= 0, color=GREEN, alpha=0.5, label="Cumulative gain")
    ax5.fill_between(years_v, cum_diff, 0, where=cum_diff < 0,  color=RED,   alpha=0.5, label="Cumulative loss")
    ax5.plot(years_v, cum_diff, color="black", lw=1.5)
    ax5.axhline(0, color="black", lw=0.8)
    ax5.set_title("Cumulative Sum of Differences (1979–2018)", fontweight="bold")
    ax5.set_xlabel("Year"); ax5.set_ylabel("Cumulative yield class diff")
    ax5.legend(fontsize=8)
    ax5.set_xlim(1978, 2019)
    ax5.annotate(f"Final: {cum_diff[-1]:.0f}",
                 xy=(years_v[-1], cum_diff[-1]),
                 xytext=(years_v[-1] - 8, cum_diff[-1] + abs(cum_diff[-1]) * 0.1 + 50),
                 fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))

    # 6. driver stacked bar
    ax6 = fig.add_subplot(gs[2, 1])
    if len(driver_yearly) > 0:
        driver_cols = [c for c in ["soil_moisture_awc", "permafrost_class",
                                    "both_flagged", "yield_change", "other"]
                       if c in driver_yearly.columns]
        bottom = np.zeros(len(driver_yearly))
        for d in driver_cols:
            ax6.bar(driver_yearly.index, driver_yearly[d], bottom=bottom,
                    label=d, color=DRIVER_COLORS.get(d, "grey"), alpha=0.85)
            bottom += driver_yearly[d].values
        ax6.legend(fontsize=7, loc="upper left", framealpha=0.7)
    ax6.set_title("Driver of Difference by Year", fontweight="bold")
    ax6.set_xlabel("Year"); ax6.set_ylabel("Pixel count")
    ax6.set_xlim(1978, 2019)

    plt.savefig(f"{OUT_DIR}/analysis_{CROP}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: {OUT_DIR}/analysis_{CROP}.png")

# ─────────────────────────────────────────────
# CROSS-CROP SUMMARY
# ─────────────────────────────────────────────

print(f"\n{'='*60}")
print("CROSS-CROP SUMMARY")
print(f"{'='*60}")

CROSS_OUT = f"{BASE}/results_original_analysis/output/all_crops"
os.makedirs(CROSS_OUT, exist_ok=True)

df_all = pd.concat(all_yearly, ignore_index=True)
df_all.to_csv(f"{CROSS_OUT}/all_crops_yearly_summary.csv", index=False)

# ── per-crop trend stats table ──
summary_rows = []
for crop, grp in df_all.groupby("crop"):
    valid = grp.dropna(subset=["n_changed", "mean_diff"])
    if len(valid) < 3:
        continue
    lr_n  = stats.linregress(valid["year"], valid["n_changed"])
    lr_d  = stats.linregress(valid["year"], valid["mean_diff"])
    lr_p  = stats.linregress(valid["year"], valid["pct_pos"])
    mk_n  = mk_test(valid["n_changed"].values)
    mk_d  = mk_test(valid["mean_diff"].values)
    summary_rows.append({
        "crop":              crop,
        "mean_n_changed":    valid["n_changed"].mean(),
        "mean_pct_pos":      valid["pct_pos"].mean(),
        "mean_diff_overall": valid["mean_diff"].mean(),
        "n_changed_slope":   lr_n.slope,
        "n_changed_p":       lr_n.pvalue,
        "mean_diff_slope":   lr_d.slope,
        "mean_diff_p":       lr_d.pvalue,
        "pct_pos_slope":     lr_p.slope,
        "pct_pos_p":         lr_p.pvalue,
        "mk_n_tau":          mk_n[0],
        "mk_n_p":            mk_n[1],
        "mk_d_tau":          mk_d[0],
        "mk_d_p":            mk_d[1],
    })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(f"{CROSS_OUT}/cross_crop_summary.csv", index=False)

print("\nCross-crop summary:")
print(df_summary.to_string(index=False, float_format="{:.4f}".format))

# ── aggregate across all crops by year ──
# sum n_changed, n_pos, n_neg, n_zeroed; average mean_diff and pct_pos
agg_yearly = (
    df_all.groupby("year")
    .agg(
        n_changed_total   = ("n_changed",  "sum"),
        n_pos_total       = ("n_pos",      "sum"),
        n_neg_total       = ("n_neg",      "sum"),
        n_zeroed_total    = ("n_zeroed",   "sum"),
        mean_diff_mean    = ("mean_diff",  "mean"),
        sum_diff_total    = ("sum_diff",   "sum"),
        pct_pos_mean      = ("pct_pos",    "mean"),
        n_crops           = ("crop",       "count"),
    )
    .reset_index()
    .dropna(subset=["n_changed_total"])
)
agg_yearly.to_csv(f"{CROSS_OUT}/aggregated_yearly.csv", index=False)

years_agg = agg_yearly["year"].values

# trends on aggregated series
lr_an  = stats.linregress(years_agg, agg_yearly["n_changed_total"])
lr_ad  = stats.linregress(years_agg, agg_yearly["mean_diff_mean"])
lr_ap  = stats.linregress(years_agg, agg_yearly["pct_pos_mean"])
mk_an  = mk_test(agg_yearly["n_changed_total"].values)
mk_ad  = mk_test(agg_yearly["mean_diff_mean"].values)
mk_ap  = mk_test(agg_yearly["pct_pos_mean"].values)

print("\n── Aggregated (all crops) trends ──")
print(f"  n_changed_total  OLS {lr_an.slope:+.2f}/yr p={lr_an.pvalue:.4f} | MK τ={mk_an[0]:+.3f} p={mk_an[1]:.4f}")
print(f"  mean_diff_mean   OLS {lr_ad.slope:+.5f}/yr p={lr_ad.pvalue:.4f} | MK τ={mk_ad[0]:+.3f} p={mk_ad[1]:.4f}")
print(f"  pct_pos_mean     OLS {lr_ap.slope:+.3f}/yr p={lr_ap.pvalue:.4f} | MK τ={mk_ap[0]:+.3f} p={mk_ap[1]:.4f}")

# ── colour palette for crops ──
CROP_COLORS = plt.cm.tab10(np.linspace(0, 1, len(CROPS)))
crop_color  = {c: CROP_COLORS[i] for i, c in enumerate(CROPS)}

BLUE   = "#2b6cb0"
RED    = "#c53030"
GREEN  = "#276749"
ORANGE = "#c05621"

# ─────────────────────────────────────────────
# ALL-CROPS FIGURE  (4 rows × 2 cols = 8 panels)
# ─────────────────────────────────────────────

fig = plt.figure(figsize=(18, 24))
gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.38)
fig.suptitle("All Crops — Permafrost vs Original FAO GAEZ 1979–2018",
             fontsize=14, fontweight="bold", y=0.99)

# ── 1. Total changed pixels per year (stacked by crop) ──
ax1 = fig.add_subplot(gs[0, 0])
bottom = np.zeros(len(agg_yearly))
for i, crop in enumerate(CROPS):
    sub = df_all[df_all["crop"] == crop].set_index("year").reindex(years_agg)["n_changed"].fillna(0).values
    ax1.bar(years_agg, sub, bottom=bottom, color=crop_color[crop], alpha=0.85,
            label=crop, width=0.8)
    bottom += sub
ax1.plot(years_agg, lr_an.slope * years_agg + lr_an.intercept, color="black", lw=2, ls="--",
         label=f"OLS {lr_an.slope:+.1f}/yr  p={lr_an.pvalue:.3f}")
ax1.set_title("Total Changed Pixels per Year (all crops)", fontweight="bold")
ax1.set_xlabel("Year"); ax1.set_ylabel("Pixel count (sum across crops)")
ax1.legend(fontsize=6, loc="upper left", ncol=2, framealpha=0.7)
ax1.set_xlim(1978, 2019)

# ── 2. Mean diff per year per crop (line per crop) ──
ax2 = fig.add_subplot(gs[0, 1])
for crop in CROPS:
    sub = df_all[df_all["crop"] == crop].dropna(subset=["mean_diff"])
    ax2.plot(sub["year"], sub["mean_diff"], color=crop_color[crop], lw=1.2,
             marker="o", ms=2, alpha=0.85, label=crop)
ax2.plot(years_agg, agg_yearly["mean_diff_mean"], color="black", lw=2.5,
         label="Mean across crops")
ax2.plot(years_agg, lr_ad.slope * years_agg + lr_ad.intercept, color="black",
         lw=1.5, ls="--", label=f"OLS {lr_ad.slope:+.5f}/yr p={lr_ad.pvalue:.3f}")
ax2.axhline(0, color="black", lw=0.8)
ax2.set_title("Mean Diff (PF − Original) per Year by Crop", fontweight="bold")
ax2.set_xlabel("Year"); ax2.set_ylabel("Mean yield class diff")
ax2.legend(fontsize=6, loc="lower right", ncol=2, framealpha=0.7)
ax2.set_xlim(1978, 2019)

# ── 3. % positive pixels per crop (line per crop) ──
ax3 = fig.add_subplot(gs[1, 0])
for crop in CROPS:
    sub = df_all[df_all["crop"] == crop].dropna(subset=["pct_pos"])
    ax3.plot(sub["year"], sub["pct_pos"], color=crop_color[crop], lw=1.2,
             marker="o", ms=2, alpha=0.85, label=crop)
ax3.plot(years_agg, agg_yearly["pct_pos_mean"], color="black", lw=2.5,
         label="Mean across crops")
ax3.plot(years_agg, lr_ap.slope * years_agg + lr_ap.intercept, color="black",
         lw=1.5, ls="--", label=f"OLS {lr_ap.slope:+.2f}%/yr p={lr_ap.pvalue:.3f}")
ax3.axhline(50, color="grey", lw=0.8, ls=":")
ax3.set_title("% Positive Changed Pixels per Year by Crop", fontweight="bold")
ax3.set_xlabel("Year"); ax3.set_ylabel("% pixels where PF > Original")
ax3.set_ylim(0, 100)
ax3.legend(fontsize=6, loc="upper left", ncol=2, framealpha=0.7)
ax3.set_xlim(1978, 2019)

# ── 4. Aggregated pos vs neg diverging bar ──
ax4 = fig.add_subplot(gs[1, 1])
ax4.bar(years_agg,  agg_yearly["n_pos_total"], color=GREEN, alpha=0.85, width=0.8, label="PF > Orig (total)")
ax4.bar(years_agg, -agg_yearly["n_neg_total"], color=RED,   alpha=0.85, width=0.8, label="PF < Orig (total)")
ax4.axhline(0, color="black", lw=0.9)
ax4b = ax4.twinx()
ax4b.plot(years_agg, agg_yearly["pct_pos_mean"], color=ORANGE, lw=1.5, marker="o", ms=3)
ax4b.plot(years_agg, lr_ap.slope * years_agg + lr_ap.intercept,
          color=ORANGE, lw=1.5, ls="--", label=f"% pos trend")
ax4b.set_ylabel("Mean % positive pixels", color=ORANGE)
ax4b.tick_params(axis="y", colors=ORANGE)
ax4b.set_ylim(0, 100)
ax4.set_title("Aggregated Positive vs Negative Pixels (all crops)", fontweight="bold")
ax4.set_xlabel("Year"); ax4.set_ylabel("Pixel count")
lines1, labs1 = ax4.get_legend_handles_labels()
lines2, labs2 = ax4b.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc="upper left", framealpha=0.7)
ax4.set_xlim(1978, 2019)

# ── 5. Cumulative sum_diff per crop ──
ax5 = fig.add_subplot(gs[2, 0])
for crop in CROPS:
    sub = df_all[df_all["crop"] == crop].dropna(subset=["sum_diff"]).sort_values("year")
    ax5.plot(sub["year"], sub["sum_diff"].cumsum(), color=crop_color[crop],
             lw=1.2, alpha=0.85, label=crop)
# aggregate cumulative
agg_cum = agg_yearly["sum_diff_total"].cumsum().values
ax5.plot(years_agg, agg_cum, color="black", lw=2.5, label="All crops combined")
ax5.axhline(0, color="black", lw=0.8)
ax5.set_title("Cumulative Sum of Differences by Crop", fontweight="bold")
ax5.set_xlabel("Year"); ax5.set_ylabel("Cumulative yield class diff")
ax5.legend(fontsize=6, loc="lower left", ncol=2, framealpha=0.7)
ax5.set_xlim(1978, 2019)

# ── 6. Crop comparison bar: mean_diff and pct_pos ──
ax6 = fig.add_subplot(gs[2, 1])
crop_labels   = df_summary["crop"].tolist()
x             = np.arange(len(crop_labels))
mean_diffs    = df_summary["mean_diff_overall"].values
pct_pos_vals  = df_summary["mean_pct_pos"].values
bar_colors    = [GREEN if v >= 0 else RED for v in mean_diffs]
bars = ax6.bar(x, mean_diffs, color=bar_colors, alpha=0.8, width=0.6)
ax6b = ax6.twinx()
ax6b.plot(x, pct_pos_vals, color=ORANGE, marker="D", ms=7, lw=2, label="Mean % positive")
ax6b.set_ylabel("Mean % positive pixels", color=ORANGE)
ax6b.tick_params(axis="y", colors=ORANGE)
ax6b.set_ylim(0, 80)
ax6.axhline(0, color="black", lw=0.8)
ax6.set_xticks(x)
ax6.set_xticklabels([c.replace("_", "\n") for c in crop_labels], fontsize=7)
ax6.set_title("Mean Overall Diff & % Positive by Crop", fontweight="bold")
ax6.set_ylabel("Mean yield class diff (PF − Orig)")
lines1, labs1 = ax6.get_legend_handles_labels()
lines2, labs2 = ax6b.get_legend_handles_labels()
ax6.legend(lines1 + lines2, labs1 + labs2, fontsize=8, framealpha=0.7)

# ── 7. Trend slopes heatmap (n_changed and pct_pos) ──
ax7 = fig.add_subplot(gs[3, 0])
slope_data = np.array([
    df_summary["n_changed_slope"].values,
    df_summary["pct_pos_slope"].values,
    df_summary["mean_diff_slope"].values * 1000,  # scale for visibility
])
im7 = ax7.imshow(slope_data, cmap="RdYlGn", aspect="auto",
                  vmin=-np.nanmax(np.abs(slope_data)), vmax=np.nanmax(np.abs(slope_data)))
ax7.set_yticks([0, 1, 2])
ax7.set_yticklabels(["n_changed slope\n(/yr)", "pct_pos slope\n(%/yr)",
                      "mean_diff slope\n(×1000/yr)"], fontsize=8)
ax7.set_xticks(np.arange(len(crop_labels)))
ax7.set_xticklabels([c.replace("_", "\n") for c in crop_labels], fontsize=7)
plt.colorbar(im7, ax=ax7, orientation="horizontal", pad=0.25, label="Slope value")
# annotate significance
pval_data = np.array([
    df_summary["n_changed_p"].values,
    df_summary["pct_pos_p"].values,
    df_summary["mean_diff_p"].values,
])
for i in range(slope_data.shape[0]):
    for j in range(slope_data.shape[1]):
        sig = "***" if pval_data[i,j] < 0.001 else ("**" if pval_data[i,j] < 0.01 else
              ("*" if pval_data[i,j] < 0.05 else ""))
        ax7.text(j, i, f"{slope_data[i,j]:.2f}{sig}", ha="center", va="center",
                 fontsize=6, color="black")
ax7.set_title("OLS Trend Slopes by Crop (* p<0.05, ** p<0.01, *** p<0.001)", fontweight="bold")

# ── 8. Aggregated zeroed vs gained ──
ax8 = fig.add_subplot(gs[3, 1])
ax8.fill_between(years_agg, agg_yearly["n_zeroed_total"], alpha=0.4, color=RED,
                 label="Total zeroed pixels")
ax8.fill_between(years_agg, agg_yearly["n_pos_total"],    alpha=0.4, color=GREEN,
                 label="Total gained pixels")
ax8.plot(years_agg, agg_yearly["n_zeroed_total"], color=RED,   lw=1.5)
ax8.plot(years_agg, agg_yearly["n_pos_total"],    color=GREEN, lw=1.5)
ax8.set_title("Total Zeroed vs Gained Pixels Across All Crops", fontweight="bold")
ax8.set_xlabel("Year"); ax8.set_ylabel("Pixel count (sum across crops)")
ax8.legend(fontsize=8, framealpha=0.7)
ax8.set_xlim(1978, 2019)

plt.savefig(f"{CROSS_OUT}/all_crops_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\nSaved: {CROSS_OUT}/all_crops_analysis.png")
print(f"Saved: {CROSS_OUT}/cross_crop_summary.csv")
print(f"Saved: {CROSS_OUT}/all_crops_yearly_summary.csv")
print(f"Saved: {CROSS_OUT}/aggregated_yearly.csv")