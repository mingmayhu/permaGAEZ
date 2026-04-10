"""
Part 1: Baseline Characterisation — 1979–1998
==============================================
Analyses scenario 1 (control) for the pre-thaw period only.
Both scenarios are identical here, so this describes the
reference state before permafrost thaw diverges the runs.

Outputs:
  - baseline_mean_suitability_map.png   : mean suitability class per pixel
  - baseline_variability_map.png        : std dev per pixel (stable vs unstable)
  - baseline_consistency_map.png        : % of years each pixel is suitable (class >= 2)
  - baseline_suitable_area_timeseries.png
  - baseline_class_composition.png      : mean area per class across 1979-1998
  - baseline_summary.csv                : annual statistics
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import rasterio
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
SCENARIO1_DIR  = Path("./data_output/final_classification/combined_barley")   # your 40-year control run
OUTPUT_DIR     = Path("./analysis_output/combined_barley")   # where to save the results
OUTPUT_DIR.mkdir(exist_ok=True)

YEARS_EARLY    = list(range(1979, 1999))   # 20 years
N_CLASSES      = 6
PIXEL_AREA_KM2 = 1.0    # adjust to your pixel size in km²

CLASS_COLORS = ["#1f4e79", "#2196F3", "#4CAF50", "#CDDC39", "#FF9800", "#00BCD4"]
CLASS_LABELS = [f"Class {i}" for i in range(N_CLASSES)]

def tif_path(d, year):
    return d / f"{year}_final_yield_class.tif"   # adjust filename pattern if needed


# ── LOAD 1979–1998 ────────────────────────────────────────────────────────────
print("Loading 1979–1998 (scenario 1)...")
arrays = []
meta   = None
for year in YEARS_EARLY:
    with rasterio.open(tif_path(SCENARIO1_DIR, year)) as src:
        arrays.append(src.read(1).astype(np.float32))
        if meta is None:
            meta = src.meta

s1_early = np.stack(arrays, axis=0)   # shape: (20, rows, cols)
print(f"Loaded: {s1_early.shape}")

# Mask nodata if present (common value: -9999 or 255)
NODATA = meta.get("nodata", None)
if NODATA is not None:
    s1_early[s1_early == NODATA] = np.nan


# ── 1. MEAN SUITABILITY MAP ───────────────────────────────────────────────────
# Average class value per pixel across all 20 years
# Excludes class 0 (unsuitable) from the mean so we see the quality
# of land that IS suitable, not dragged down by zeros

mean_all   = np.nanmean(s1_early, axis=0)                         # includes 0s
s1_masked  = np.where(s1_early == 0, np.nan, s1_early)
mean_suit  = np.nanmean(s1_masked, axis=0)                        # excludes 0s

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im0 = axes[0].imshow(mean_all, cmap="YlGn", vmin=0, vmax=5, interpolation="nearest")
axes[0].set_title("Mean Suitability Class\n(all pixels, incl. unsuitable)", fontsize=11)
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.04, label="Mean class")

im1 = axes[1].imshow(mean_suit, cmap="YlOrRd", vmin=1, vmax=5, interpolation="nearest")
axes[1].set_title("Mean Suitability Class\n(suitable pixels only, class ≥ 1)", fontsize=11)
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.04, label="Mean class")

plt.suptitle("Baseline Mean Suitability 1979–1998", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "baseline_mean_suitability_map.png", dpi=150)
plt.close()
print("Saved: baseline_mean_suitability_map.png")


# ── 2. VARIABILITY MAP ────────────────────────────────────────────────────────
# Std dev per pixel — high values = suitability fluctuates a lot year to year
# Low values = pixel is consistently one class (stable)

std_map = np.nanstd(s1_early, axis=0)

fig, ax = plt.subplots(figsize=(9, 4))
im = ax.imshow(std_map, cmap="hot_r", vmin=0, interpolation="nearest")
ax.set_title("Interannual Variability (Std Dev) 1979–1998\nHigh = unstable class assignment", fontsize=11)
ax.axis("off")
plt.colorbar(im, ax=ax, fraction=0.04, label="Std dev (class units)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "baseline_variability_map.png", dpi=150)
plt.close()
print("Saved: baseline_variability_map.png")


# ── 3. CONSISTENCY MAP ────────────────────────────────────────────────────────
# % of years each pixel has class >= 2 (suitable)
# 100% = always suitable, 0% = never suitable, 50% = marginal

suitable_count = np.sum(s1_early >= 2, axis=0)
consistency    = (suitable_count / len(YEARS_EARLY)) * 100   # percentage

fig, ax = plt.subplots(figsize=(9, 4))
im = ax.imshow(consistency, cmap="RdYlGn", vmin=0, vmax=100, interpolation="nearest")
ax.set_title("Suitability Consistency 1979–1998\n% of years with class ≥ 2", fontsize=11)
ax.axis("off")
plt.colorbar(im, ax=ax, fraction=0.04, label="% years suitable")

# Add contour lines at 25%, 50%, 75%
try:
    for level, ls in [(25, ":"), (50, "--"), (75, "-")]:
        ax.contour(consistency, levels=[level], colors="black",
                   linewidths=0.8, linestyles=ls, alpha=0.6)
except Exception:
    pass

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "baseline_consistency_map.png", dpi=150)
plt.close()
print("Saved: baseline_consistency_map.png")


# ── 4. SUITABLE AREA TIME SERIES ──────────────────────────────────────────────
suitable_area = np.array([(s1_early[i] >= 2).sum() * PIXEL_AREA_KM2
                           for i in range(len(YEARS_EARLY))])
mean_area = suitable_area.mean()
std_area  = suitable_area.std()

# Linear trend
z = np.polyfit(YEARS_EARLY, suitable_area, 1)
trend = np.poly1d(z)

fig, ax = plt.subplots(figsize=(10, 4))
ax.fill_between(YEARS_EARLY,
                mean_area - std_area,
                mean_area + std_area,
                alpha=0.15, color="steelblue", label="±1 std dev")
ax.axhline(mean_area, color="steelblue", linestyle="--", linewidth=1.2,
           label=f"Mean: {mean_area:.1f} km²")
ax.plot(YEARS_EARLY, suitable_area, color="steelblue", marker="o",
        markersize=5, linewidth=1.8, label="Annual suitable area")
ax.plot(YEARS_EARLY, trend(YEARS_EARLY), color="navy", linestyle=":",
        linewidth=1.5, label=f"Trend: {z[0]:+.2f} km²/yr")

ax.set(title="Baseline Suitable Area (Class ≥ 2) 1979–1998",
       xlabel="Year", ylabel="Area (km²)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "baseline_suitable_area_timeseries.png", dpi=150)
plt.close()
print("Saved: baseline_suitable_area_timeseries.png")


# ── 5. CLASS COMPOSITION BAR CHART ───────────────────────────────────────────
mean_class_area = []
std_class_area  = []
for cls in range(N_CLASSES):
    areas = np.array([(s1_early[i] == cls).sum() * PIXEL_AREA_KM2
                       for i in range(len(YEARS_EARLY))])
    mean_class_area.append(areas.mean())
    std_class_area.append(areas.std())

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(N_CLASSES)
bars = ax.bar(x, mean_class_area, color=CLASS_COLORS, alpha=0.85,
              yerr=std_class_area, capsize=5, edgecolor="white", linewidth=0.5)
ax.set(title="Mean Area per Suitability Class 1979–1998\n(error bars = interannual std dev)",
       xlabel="Suitability class", ylabel="Mean area (km²)")
ax.set_xticks(x)
ax.set_xticklabels([f"Class {i}" for i in range(N_CLASSES)])
ax.grid(axis="y", alpha=0.3)

# Annotate bars with values
for bar, val, err in zip(bars, mean_class_area, std_class_area):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 0.5,
            f"{val:.0f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "baseline_class_composition.png", dpi=150)
plt.close()
print("Saved: baseline_class_composition.png")


# ── 6. SUMMARY CSV ───────────────────────────────────────────────────────────
records = []
for i, year in enumerate(YEARS_EARLY):
    row = {"year": year,
           "suitable_area_km2": float((s1_early[i] >= 2).sum() * PIXEL_AREA_KM2),
           "mean_class_all_pixels": float(np.nanmean(s1_early[i])),
           "mean_class_suitable_only": float(np.nanmean(
               np.where(s1_early[i] == 0, np.nan, s1_early[i])))}
    for cls in range(N_CLASSES):
        row[f"class{cls}_area_km2"] = float((s1_early[i] == cls).sum() * PIXEL_AREA_KM2)
    records.append(row)

df = pd.DataFrame(records)
df.to_csv(OUTPUT_DIR / "baseline_summary.csv", index=False)
print("Saved: baseline_summary.csv")

print(f"\n── Baseline Summary ─────────────────────────────────────────────────")
print(f"  Mean suitable area:  {mean_area:.1f} ± {std_area:.1f} km²")
print(f"  Min suitable area:   {suitable_area.min():.1f} km² ({YEARS_EARLY[suitable_area.argmin()]})")
print(f"  Max suitable area:   {suitable_area.max():.1f} km² ({YEARS_EARLY[suitable_area.argmax()]})")
print(f"  Area trend:          {z[0]:+.2f} km²/year over 1979–1998")
print(f"\n✓ Part 1 complete. Outputs saved to: {OUTPUT_DIR.resolve()}")