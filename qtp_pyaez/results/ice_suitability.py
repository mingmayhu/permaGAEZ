"""
ice_suitability_analysis.py

Two analyses, both overall (mean across crops) and per crop:

1. Mean ice content (mean across 40 years) vs delta suitability
   — does baseline ice level predict suitability change?

2. Δice vs delta suitability
   — does ice change predict suitability change?

Outputs:
    ice_diagnostics/ice_suit_correlations.csv
    ice_diagnostics/ice_vs_deltasuit_scatter.png       (overall, 2 panels)
    ice_diagnostics/ice_vs_deltasuit_per_crop.png      (per crop, 2 rows x 10 cols)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr
from osgeo import gdal

# === Config ===
ice_root     = "../data_input/ice_processed"
suit_dir     = "../results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/5_spatial/1_mean_delta"
pf_mask_path = "../data_input/permafrost_qilian.tif"
output_dir   = "../results/ice_diagnostics"
os.makedirs(output_dir, exist_ok=True)

CROPS = [
    "spring_wheat", "spring_barley", "winter_wheat", "winter_barley",
    "silage_maize", "white_potato", "oat", "dry_pea",
    "spring_rape", "winter_rape"
]

# === Load valid mask ===
ds     = gdal.Open(pf_mask_path)
pf_arr = ds.GetRasterBand(1).ReadAsArray().astype(float)
nodata = ds.GetRasterBand(1).GetNoDataValue()
ds     = None
valid_mask = pf_arr != 0
if nodata is not None:
    valid_mask &= pf_arr != nodata

# === Load ice variables ===
ice_annual = np.load(os.path.join(ice_root, "permafrost_ice_annual.npy"))  # (40, rows, cols)
ice_mean   = np.nanmean(ice_annual, axis=0)                                 # mean across 40 years
ice_delta  = np.load(os.path.join(ice_root, "permafrost_ice_delta.npy"))   # period2 - period1

# === Load delta suitability per crop ===
def load_delta_suit(crop):
    fpath = os.path.join(suit_dir, f"combined_{crop}_mean_delta_suit.tif")
    if not os.path.exists(fpath):
        print(f"Missing: {fpath}")
        return None
    ds_s = gdal.Open(fpath)
    arr  = ds_s.GetRasterBand(1).ReadAsArray().astype(float)
    nd   = ds_s.GetRasterBand(1).GetNoDataValue()
    ds_s = None
    if nd is not None:
        arr[arr == nd] = np.nan
    return arr

suit_arrays = {}
for crop in CROPS:
    arr = load_delta_suit(crop)
    if arr is not None:
        suit_arrays[crop] = arr

# Mean delta suitability across all crops
suit_mean = np.nanmean(np.array(list(suit_arrays.values())), axis=0)

# === Correlation function ===
def correlate(x_map, y_map, mask):
    px = mask & ~np.isnan(x_map) & ~np.isnan(y_map)
    if px.sum() < 10:
        return np.nan, np.nan, 0
    r, p = spearmanr(x_map[px], y_map[px])
    return r, p, int(px.sum())

# === Run correlations ===
results = []

# Overall
r1, p1, n1 = correlate(ice_mean,  suit_mean, valid_mask)
r2, p2, n2 = correlate(ice_delta, suit_mean, valid_mask)
results.append({"crop": "OVERALL", "r_ice_mean": round(r1,3), "p_ice_mean": round(p1,4),
                "r_ice_delta": round(r2,3), "p_ice_delta": round(p2,4), "n": n1})
print(f"\nOVERALL (n={n1})")
print(f"  Ice mean  vs Δsuit: r={r1:.3f}  p={p1:.4f}")
print(f"  Δice      vs Δsuit: r={r2:.3f}  p={p2:.4f}")

# Per crop
for crop in CROPS:
    if crop not in suit_arrays:
        continue
    suit = suit_arrays[crop]
    r1, p1, n1 = correlate(ice_mean,  suit, valid_mask)
    r2, p2, n2 = correlate(ice_delta, suit, valid_mask)
    results.append({"crop": crop, "r_ice_mean": round(r1,3), "p_ice_mean": round(p1,4),
                    "r_ice_delta": round(r2,3), "p_ice_delta": round(p2,4), "n": n1})
    print(f"\n{crop} (n={n1})")
    print(f"  Ice mean  vs Δsuit: r={r1:.3f}  p={p1:.4f}")
    print(f"  Δice      vs Δsuit: r={r2:.3f}  p={p2:.4f}")

# === Save CSV ===
df = pd.DataFrame(results)
df.to_csv(os.path.join(output_dir, "ice_suit_correlations.csv"), index=False)
print("\nSaved ice_suit_correlations.csv")

# === Overall scatter (2 panels) ===
px_all = valid_mask & ~np.isnan(ice_mean) & ~np.isnan(ice_delta) & ~np.isnan(suit_mean)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, x, xlabel, color, r, p in [
    (axes[0], ice_mean[px_all],  "Mean ice content (mm, 40-yr mean)", "steelblue",
     df.loc[df.crop=="OVERALL","r_ice_mean"].values[0],
     df.loc[df.crop=="OVERALL","p_ice_mean"].values[0]),
    (axes[1], ice_delta[px_all], "Δ Ice content (mm, period2-period1)", "mediumpurple",
     df.loc[df.crop=="OVERALL","r_ice_delta"].values[0],
     df.loc[df.crop=="OVERALL","p_ice_delta"].values[0]),
]:
    ax.scatter(x, suit_mean[px_all], alpha=0.3, s=8, color=color)
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.axvline(0, color='grey', linewidth=0.5)
    m, b, _, _, _ = __import__('scipy').stats.linregress(x, suit_mean[px_all])
    xr = np.linspace(x.min(), x.max(), 100)
    ax.plot(xr, m*xr+b, 'k-', linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Δ Suitability (mean across crops)")
    ax.set_title(f"Spearman r={r:.3f}, p={p:.4f}\nn={px_all.sum()}")

plt.suptitle("Ice Content vs Delta Suitability (Overall)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ice_vs_deltasuit_scatter.png"), dpi=150)
plt.close()
print("Saved ice_vs_deltasuit_scatter.png")

# === Per crop scatter (2 rows x 10 cols) ===
n_crops = len(suit_arrays)
fig, axes = plt.subplots(2, n_crops, figsize=(3*n_crops, 8))

for ci, crop in enumerate(suit_arrays.keys()):
    suit = suit_arrays[crop]
    px_c = valid_mask & ~np.isnan(ice_mean) & ~np.isnan(ice_delta) & ~np.isnan(suit)

    row = df[df.crop == crop]

    for ri, (x, xlabel, color, r_col, p_col) in enumerate([
        (ice_mean,  "Ice mean (mm)",  "steelblue",    "r_ice_mean",  "p_ice_mean"),
        (ice_delta, "Δ Ice (mm)",     "mediumpurple", "r_ice_delta", "p_ice_delta"),
    ]):
        ax = axes[ri, ci]
        ax.scatter(x[px_c], suit[px_c], alpha=0.2, s=4, color=color)
        ax.axhline(0, color='grey', linewidth=0.4)
        r_val = row[r_col].values[0]
        p_val = row[p_col].values[0]
        ax.set_title(f"{crop}\nr={r_val:.3f} p={p_val:.3f}", fontsize=7)
        if ci == 0:
            ax.set_ylabel("Δ Suitability" if ri == 0 else "Δ Suitability")
        ax.set_xlabel(xlabel, fontsize=7)
        ax.tick_params(labelsize=6)

plt.suptitle("Ice Content vs Delta Suitability — Per Crop", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ice_vs_deltasuit_per_crop.png"), dpi=150)
plt.close()
print("Saved ice_vs_deltasuit_per_crop.png")

print("\nDone.")

