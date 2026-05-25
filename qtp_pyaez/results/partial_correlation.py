"""
partial_correlation_ice_asm_suitability.py

Two partial correlations across all valid pixels (overall, not per crop):
1. ΔALT vs Δsuit | Δice  — confirms ALT finding robust after controlling for ice
2. ΔASM vs Δsuit | ΔALT  — isolates moisture's independent contribution beyond thaw depth

Plus baselines:
    ΔALT vs Δsuit
    ΔASM vs Δsuit

Delta suitability = mean delta across all 10 crops per pixel.

Partial correlation via OLS residuals on ranked data (partial Spearman).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import rankdata, spearmanr
from osgeo import gdal

# === Config ===
ice_root     = "../data_input/ice_processed"
pf_root      = "../data_input/permafrost_yearly"
suit_root    = "../results"   # adjust to where your delta suitability rasters live
pf_mask_path = "../data_input/permafrost_qilian.tif"
output_dir   = "../results/ice_diagnostics"
os.makedirs(output_dir, exist_ok=True)

years = list(range(1979, 2019))
idx1  = [i for i, y in enumerate(years) if 1979 <= y <= 1998]
idx2  = [i for i, y in enumerate(years) if 1999 <= y <= 2018]

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

# === Load Δice ===
ice_delta = np.load(os.path.join(ice_root, "permafrost_ice_delta.npy"))

# === Compute ΔASM and ΔALT ===
asm_stack = []
alt_stack = []
for year in years:
    asm = np.load(os.path.join(pf_root, str(year), "avail_soil_moisture.npy"))
    alt = np.load(os.path.join(pf_root, str(year), "active_layer_depth.npy"))
    asm_stack.append(np.nanmean(asm, axis=2))
    alt_stack.append(np.nanmax(alt, axis=2))   # annual max ALT

asm_stack = np.array(asm_stack)
alt_stack = np.array(alt_stack)
asm_delta = np.nanmean(asm_stack[idx2], axis=0) - np.nanmean(asm_stack[idx1], axis=0)
alt_delta = np.nanmean(alt_stack[idx2], axis=0) - np.nanmean(alt_stack[idx1], axis=0)

# === Compute mean delta suitability across all crops ===
CROP_FILENAMES = {
    "spring_wheat":  "combined_spring_wheat_mean_delta_suit.tif",
    "spring_barley": "combined_spring_barley_mean_delta_suit.tif",
    "winter_wheat":  "combined_winter_wheat_mean_delta_suit.tif",
    "winter_barley": "combined_winter_barley_mean_delta_suit.tif",
    "silage_maize":  "combined_silage_maize_mean_delta_suit.tif",
    "white_potato":  "combined_white_potato_mean_delta_suit.tif",
    "oat":           "combined_oat_mean_delta_suit.tif",
    "dry_pea":       "combined_dry_pea_mean_delta_suit.tif",
    "spring_rape":   "combined_spring_rape_mean_delta_suit.tif",
    "winter_rape":   "combined_winter_rape_mean_delta_suit.tif",
}

suit_dir = "../results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/5_spatial/1_mean_delta"

suit_stack = []
for crop, fname in CROP_FILENAMES.items():
    fpath = os.path.join(suit_dir, fname)
    if not os.path.exists(fpath):
        print(f"Missing: {fpath}, skipping")
        continue
    ds_suit = gdal.Open(fpath)
    arr = ds_suit.GetRasterBand(1).ReadAsArray().astype(float)
    nd  = ds_suit.GetRasterBand(1).GetNoDataValue()
    ds_suit = None
    if nd is not None:
        arr[arr == nd] = np.nan
    suit_stack.append(arr)

suit_delta = np.nanmean(np.array(suit_stack), axis=0)
# # === Compute mean delta suitability across all crops ===
# suit_stack = []
# for crop in CROPS:
#     delta_path = os.path.join(suit_root, f"delta_suitability_{crop}.npy")
#     if not os.path.exists(delta_path):
#         print(f"Missing: {delta_path}, skipping")
#         continue
#     suit_stack.append(np.load(delta_path))

# suit_delta = np.nanmean(np.array(suit_stack), axis=0)  # (rows, cols)

# === Valid pixels ===
px = (valid_mask
      & ~np.isnan(ice_delta)
      & ~np.isnan(asm_delta)
      & ~np.isnan(alt_delta)
      & ~np.isnan(suit_delta))
print(f"Valid pixels: {px.sum()}")

ice_v  = ice_delta[px]
asm_v  = asm_delta[px]
alt_v  = alt_delta[px]
suit_v = suit_delta[px]

# === Partial correlation helper ===
def partial_spearman(x, y, z):
    """Partial Spearman r(x, y | z) via OLS residuals on ranks."""
    xr = rankdata(x).astype(float)
    yr = rankdata(y).astype(float)
    zr = rankdata(z).astype(float)
    def resid(a, b):
        slope, intercept, _, _, _ = stats.linregress(b, a)
        return a - (slope * b + intercept)
    r, p = spearmanr(resid(xr, zr), resid(yr, zr))
    return r, p

# === Baselines ===
r_alt_base, p_alt_base = spearmanr(alt_v, suit_v)
r_asm_base, p_asm_base = spearmanr(asm_v, suit_v)

# === Partial correlations ===
r_alt_ctrl_ice, p_alt_ctrl_ice = partial_spearman(alt_v, suit_v, ice_v)  # ΔALT vs Δsuit | Δice
r_asm_ctrl_alt, p_asm_ctrl_alt = partial_spearman(asm_v, suit_v, alt_v)  # ΔASM vs Δsuit | ΔALT

# === Print results ===
print(f"\n{'='*55}")
print(f"  ΔALT vs Δsuit (baseline):        r={r_alt_base:.3f}  p={p_alt_base:.4f}")
print(f"  ΔALT vs Δsuit | Δice:            r={r_alt_ctrl_ice:.3f}  p={p_alt_ctrl_ice:.4f}")
print(f"{'='*55}")
print(f"  ΔASM vs Δsuit (baseline):        r={r_asm_base:.3f}  p={p_asm_base:.4f}")
print(f"  ΔASM vs Δsuit | ΔALT:            r={r_asm_ctrl_alt:.3f}  p={p_asm_ctrl_alt:.4f}")
print(f"{'='*55}")

# === Bar chart ===
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax, baseline, partial, label_base, label_partial, title in [
    (axes[0],
     r_alt_base, r_alt_ctrl_ice,
     "ΔALT vs Δsuit\n(baseline)",
     "ΔALT vs Δsuit\n| Δice",
     "ALT effect: before/after\ncontrolling for ice"),
    (axes[1],
     r_asm_base, r_asm_ctrl_alt,
     "ΔASM vs Δsuit\n(baseline)",
     "ΔASM vs Δsuit\n| ΔALT",
     "ASM effect: before/after\ncontrolling for ALT"),
]:
    bars = ax.bar([0, 1], [baseline, partial],
                  color=["steelblue", "darkorange"], width=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([label_base, label_partial])
    ax.set_ylabel("Spearman r")
    ax.set_title(title)
    ax.axhline(0, color='black', linewidth=0.8)
    for bar, val in zip(bars, [baseline, partial]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                f"{val:.3f}", ha='center', va='bottom', fontsize=10)

plt.suptitle("Partial Correlations: Environmental Drivers of Δ Suitability\n(all crops, all valid pixels)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "partial_correlation_plot.png"), dpi=150)
plt.close()
print("Saved partial_correlation_plot.png")
print("Done.")
# After loading ice_delta, add:
ice_mean_arr = np.load(os.path.join(ice_root, "permafrost_ice_annual.npy"))
ice_mean_arr = np.nanmean(ice_mean_arr, axis=0)  # (rows, cols)

# Then after defining px, add:
ice_mean_v = ice_mean_arr[px]
r_alt_ctrl_icemean, p_alt_ctrl_icemean = partial_spearman(alt_v, suit_v, ice_mean_v)
print(f"ΔALT vs Δsuit | ice mean:   r={r_alt_ctrl_icemean:.3f}  p={p_alt_ctrl_icemean:.4f}")