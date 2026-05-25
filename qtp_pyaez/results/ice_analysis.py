"""
diagnose_ice_trend.py

Quick diagnostics for permafrost ice content trend:
1. Regional mean timeseries + MK trend test
2. Spatial map of Sen's slope (trend per pixel)
3. Spearman correlation of ice delta vs ASM delta across pixels
4. Scatter plot: delta ice vs delta ASM

Requires:
    ice_processed/permafrost_ice_annual.npy   (n_years, rows, cols)
    ice_processed/permafrost_ice_delta.npy    (rows, cols)
    permafrost_yearly/asm_period_means.npy    OR compute delta ASM inline
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pymannkendall import original_test as mk_test

# === Config ===
ice_root    = "../data_input/ice_processed"
asm_root    = "../data_input/permafrost_yearly"
pf_mask_path = "../data_input/permafrost_qilian.tif"
output_dir  = "../results/ice_diagnostics"
os.makedirs(output_dir, exist_ok=True)

years = list(range(1979, 2019))

# === Load data ===
ice_annual = np.load(os.path.join(ice_root, "permafrost_ice_annual.npy"))  # (40, rows, cols)
ice_delta  = np.load(os.path.join(ice_root, "permafrost_ice_delta.npy"))   # (rows, cols)

idx1 = [i for i, y in enumerate(years) if 1979 <= y <= 1998]
idx2 = [i for i, y in enumerate(years) if 1999 <= y <= 2018]
# === Compute ASM period means from annual files ===
asm_stack = []
for year in years:
    path = os.path.join(asm_root, str(year), "avail_soil_moisture.npy")
    arr = np.load(path)  # (rows, cols, 365)
    asm_stack.append(np.nanmean(arr, axis=2))  # annual mean → (rows, cols)

asm_stack = np.array(asm_stack)  # (40, rows, cols)
asm_p1 = np.nanmean(asm_stack[idx1], axis=0)
asm_p2 = np.nanmean(asm_stack[idx2], axis=0)
asm_delta = asm_p2 - asm_p1
asm_delta = asm_p2 - asm_p1

# === Load lake mask (exclude pixels where permafrost == 0 or nodata) ===
from osgeo import gdal
ds = gdal.Open(pf_mask_path)
pf_arr = ds.GetRasterBand(1).ReadAsArray()
ds = None
lake_mask = (pf_arr == 0) | (pf_arr == ds.GetRasterBand(1).GetNoDataValue() if False else False)
valid_mask = (pf_arr != 0) & (~np.isnan(pf_arr.astype(float)))

# === 1. Regional mean timeseries + MK test ===
regional_mean = np.array([
    np.nanmean(ice_annual[yi][valid_mask]) for yi in range(len(years))
])

result = mk_test(regional_mean)
print(f"MK test — trend: {result.trend}, tau: {result.Tau:.3f}, "
      f"p: {result.p:.4f}, slope: {result.slope:.4f} mm/yr")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(years, regional_mean, 'b-o', markersize=3, label='Regional mean ice (mm)')
# Sen's slope line
slope_line = result.slope * (np.array(years) - years[0]) + regional_mean[0]
ax.plot(years, slope_line, 'r--',
        label=f"Sen's slope: {result.slope:.3f} mm/yr (p={result.p:.3f})")
ax.axvline(1999, color='grey', linestyle=':', alpha=0.7, label='1999 divergence')
ax.set_xlabel("Year")
ax.set_ylabel("Mean peak ice content (mm)")
ax.set_title("Regional Mean Annual Maximum Soil Ice Content (1979–2018)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ice_trend_timeseries.png"), dpi=150)
plt.close()
print("Saved ice_trend_timeseries.png")

# === 2. Spatial map of ice delta ===
fig, ax = plt.subplots(figsize=(10, 5))
masked_delta = np.where(valid_mask, ice_delta, np.nan)
vmax = np.nanpercentile(np.abs(masked_delta), 95)
im = ax.imshow(masked_delta, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
plt.colorbar(im, ax=ax, label='Δ Ice content (mm), period2 - period1')
ax.set_title("Change in Peak Soil Ice Content (1999–2018 minus 1979–1998)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ice_delta_map.png"), dpi=150)
plt.close()
print("Saved ice_delta_map.png")

# === 3. Spearman correlation: delta ice vs delta ASM ===
valid_pixels = valid_mask & ~np.isnan(ice_delta) & ~np.isnan(asm_delta)
ice_flat  = ice_delta[valid_pixels]
asm_flat  = asm_delta[valid_pixels]

r, p = stats.spearmanr(ice_flat, asm_flat)
print(f"\nSpearman r (Δice vs ΔASM): {r:.3f}, p={p:.4f}, n={valid_pixels.sum()}")

# === 4. Scatter: delta ice vs delta ASM ===
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(ice_flat, asm_flat, alpha=0.3, s=10, color='steelblue')
ax.axhline(0, color='grey', linewidth=0.5)
ax.axvline(0, color='grey', linewidth=0.5)
ax.set_xlabel("Δ Peak ice content (mm)")
ax.set_ylabel("Δ Available soil moisture (mm)")
ax.set_title(f"Δ Ice vs Δ ASM across pixels\nSpearman r={r:.3f}, p={p:.4f}")
# Add OLS line
m, b, _, _, _ = stats.linregress(ice_flat, asm_flat)
x_range = np.linspace(ice_flat.min(), ice_flat.max(), 100)
ax.plot(x_range, m * x_range + b, 'r-', linewidth=1.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ice_vs_asm_scatter.png"), dpi=150)
plt.close()
print("Saved ice_vs_asm_scatter.png")

print("\nDone. Check results/ice_diagnostics/")