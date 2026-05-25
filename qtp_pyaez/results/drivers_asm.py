"""
asm_driver_scatter.py

Three scatter plots showing what drives ΔASM across pixels:
1. Δice vs ΔASM
2. ΔALT vs ΔASM  
3. Δprecip vs ΔASM

Each panel shows Spearman r and p-value.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from osgeo import gdal

# === Config ===
pf_root      = "../data_input/permafrost_yearly"
clim_root    = "../data_input/climate_yearly"
ice_root     = "../data_input/ice_processed"
pf_mask_path = "../data_input/permafrost_qilian.tif"
output_dir   = "../results/ice_diagnostics"
os.makedirs(output_dir, exist_ok=True)

years = list(range(1979, 2019))
idx1  = [i for i, y in enumerate(years) if 1979 <= y <= 1998]
idx2  = [i for i, y in enumerate(years) if 1999 <= y <= 2018]

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

# === Compute ΔASM, ΔALT, Δprecip ===
asm_stack   = []
alt_stack   = []
prec_stack  = []

for year in years:
    asm  = np.load(os.path.join(pf_root,   str(year), "avail_soil_moisture.npy"))
    alt  = np.load(os.path.join(pf_root,   str(year), "active_layer_depth.npy"))
    prec = np.load(os.path.join(clim_root, str(year), "Precip.npy"))

    asm_stack.append(np.nanmean(asm,  axis=2))
    alt_stack.append(np.nanmax(alt,   axis=2))
    prec_stack.append(np.nansum(prec, axis=2))  # annual total precipitation

asm_stack  = np.array(asm_stack)
alt_stack  = np.array(alt_stack)
prec_stack = np.array(prec_stack)

asm_delta  = np.nanmean(asm_stack[idx2],  axis=0) - np.nanmean(asm_stack[idx1],  axis=0)
alt_delta  = np.nanmean(alt_stack[idx2],  axis=0) - np.nanmean(alt_stack[idx1],  axis=0)
prec_delta = np.nanmean(prec_stack[idx2], axis=0) - np.nanmean(prec_stack[idx1], axis=0)

# === Valid pixels ===
px = (valid_mask
      & ~np.isnan(ice_delta)
      & ~np.isnan(asm_delta)
      & ~np.isnan(alt_delta)
      & ~np.isnan(prec_delta))
print(f"Valid pixels: {px.sum()}")

ice_v  = ice_delta[px]
asm_v  = asm_delta[px]
alt_v  = alt_delta[px]
prec_v = prec_delta[px]

# === Plot ===
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, x, xlabel, color in [
    (axes[0], ice_v,  "Δ Peak ice content (mm)",       "steelblue"),
    (axes[1], alt_v,  "Δ Max active layer depth (m)",  "darkorange"),
    (axes[2], prec_v, "Δ Annual precipitation (mm)",   "seagreen"),
]:
    r, p = spearmanr(x, asm_v)
    ax.scatter(x, asm_v, alpha=0.3, s=8, color=color)
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.axvline(0, color='grey', linewidth=0.5)
    # OLS line
    m, b, _, _, _ = __import__('scipy').stats.linregress(x, asm_v)
    xr = np.linspace(x.min(), x.max(), 100)
    ax.plot(xr, m * xr + b, 'k-', linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Δ Available soil moisture (mm)")
    ax.set_title(f"Spearman r={r:.3f}, p={p:.4f}\nn={px.sum()}")

plt.suptitle("Drivers of Change in Available Soil Moisture (1979–1998 vs 1999–2018)",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "asm_driver_scatter.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved asm_driver_scatter.png")
print("Done.")

# === Δice vs ΔALT ===
r_ice_alt, p_ice_alt = spearmanr(ice_v, alt_v)
print(f"\nΔice vs ΔALT: Spearman r={r_ice_alt:.3f}, p={p_ice_alt:.4f}, n={px.sum()}")
r_ice_precip, p_ice_precip = spearmanr(ice_v, prec_v)
print(f"\nΔice vs Δprecip: Spearman r={r_ice_precip:.3f}, p={p_ice_precip:.4f}, n={px.sum()}")