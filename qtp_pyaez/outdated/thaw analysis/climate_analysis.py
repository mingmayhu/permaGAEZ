"""
Part 2: Climate Signal — 1979–1998 vs 1999–2018 (Scenario 1 Only)
==================================================================
Compares the two periods WITHIN scenario 1 (control) only.
Since scenario 1 has evolving permafrost AND correct climate throughout,
any difference between periods reflects BOTH climate change AND permafrost thaw.

This is your "total change" signal. When combined with Part 3 (permafrost-only
signal from scenario 1 vs scenario 2), you can partition:

    Total change  =  Climate effect  +  Permafrost thaw effect
    Climate effect = Total change  -  Permafrost thaw effect

Outputs:
  - climate_mean_maps_by_period.png       : mean suitability per period side by side
  - climate_difference_map.png            : late minus early mean (where did things change?)
  - climate_suitable_area_full_timeseries.png  : full 1979-2018 with period means + baseline band
  - climate_class_shift.png               : how class composition changed between periods
  - climate_transition_matrix.png         : pixel-level class transitions between period means
  - climate_period_boxplots.png           : distribution of annual suitable area per period
  - climate_summary.csv
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import rasterio
import seaborn as sns
from pathlib import Path
from scipy import stats

# ── CONFIG ────────────────────────────────────────────────────────────────────
SCENARIO1_DIR  = Path("./data_output/final_classification/combined_barley")   # your 40-year control run
OUTPUT_DIR     = Path("./analysis_output/combined_barley")
OUTPUT_DIR.mkdir(exist_ok=True)

YEARS_ALL      = list(range(1979, 2019))
YEARS_EARLY    = list(range(1979, 1999))
YEARS_LATE     = list(range(1999, 2019))
EARLY_IDX      = [YEARS_ALL.index(y) for y in YEARS_EARLY]
LATE_IDX       = [YEARS_ALL.index(y) for y in YEARS_LATE]

N_CLASSES      = 6
PIXEL_AREA_KM2 = 1.0    # adjust to your pixel size in km²

CLASS_COLORS = ["#1f4e79", "#2196F3", "#4CAF50", "#CDDC39", "#FF9800", "#00BCD4"]

def tif_path(d, year):
    return d / f"{year}_final_yield_class.tif"   # adjust if needed


# ── LOAD ALL 40 YEARS ─────────────────────────────────────────────────────────
print("Loading scenario 1 (all 40 years)...")
arrays = []
for year in YEARS_ALL:
    with rasterio.open(tif_path(SCENARIO1_DIR, year)) as src:
        arrays.append(src.read(1).astype(np.float32))

s1 = np.stack(arrays, axis=0)   # shape: (40, rows, cols)
print(f"Loaded: {s1.shape}")

NODATA = None  # set to your nodata value if applicable e.g. -9999
if NODATA is not None:
    s1[s1 == NODATA] = np.nan

s1_early_arr = s1[EARLY_IDX]   # (20, rows, cols)
s1_late_arr  = s1[LATE_IDX]    # (20, rows, cols)


# ── 1. MEAN MAPS PER PERIOD ───────────────────────────────────────────────────
mean_early = np.nanmean(s1_early_arr, axis=0)
mean_late  = np.nanmean(s1_late_arr,  axis=0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
vmin, vmax = 0, 5

im0 = axes[0].imshow(mean_early, cmap="YlGn", vmin=vmin, vmax=vmax, interpolation="nearest")
axes[0].set_title("Mean Suitability\n1979–1998 (baseline)", fontsize=11)
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.04, label="Mean class")

im1 = axes[1].imshow(mean_late, cmap="YlGn", vmin=vmin, vmax=vmax, interpolation="nearest")
axes[1].set_title("Mean Suitability\n1999–2018 (total change)", fontsize=11)
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.04, label="Mean class")

plt.suptitle("Scenario 1: Mean Suitability by Period\n(change reflects climate + permafrost thaw combined)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "climate_mean_maps_by_period.png", dpi=150)
plt.close()
print("Saved: climate_mean_maps_by_period.png")


# ── 2. DIFFERENCE MAP (LATE − EARLY) ─────────────────────────────────────────
diff_periods = mean_late - mean_early   # positive = improved, negative = degraded

vmax_d = max(abs(np.nanmin(diff_periods)), abs(np.nanmax(diff_periods)), 0.5)

fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(diff_periods, cmap="RdBu_r", vmin=-vmax_d, vmax=vmax_d,
               interpolation="nearest")
ax.set_title("Total Change: Mean(1999–2018) − Mean(1979–1998)\n"
             "Blue = improved, Red = degraded  |  Signal = climate + permafrost thaw",
             fontsize=11)
ax.axis("off")
plt.colorbar(im, ax=ax, fraction=0.03, label="Class change")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "climate_difference_map.png", dpi=150)
plt.close()
print("Saved: climate_difference_map.png")


# ── 3. FULL 40-YEAR TIME SERIES WITH PERIOD ANNOTATIONS ──────────────────────
suitable_all = np.array([(s1[i] >= 2).sum() * PIXEL_AREA_KM2
                          for i in range(len(YEARS_ALL))])

early_mean = suitable_all[EARLY_IDX].mean()
early_std  = suitable_all[EARLY_IDX].std()
late_mean  = suitable_all[LATE_IDX].mean()

# Trend lines per period
z_early = np.polyfit(YEARS_EARLY, suitable_all[EARLY_IDX], 1)
z_late  = np.polyfit(YEARS_LATE,  suitable_all[LATE_IDX],  1)

fig, ax = plt.subplots(figsize=(13, 5))

# Baseline variability band (±1 std from early period)
ax.axhspan(early_mean - early_std, early_mean + early_std,
           alpha=0.12, color="steelblue", label="Baseline ±1 std (1979–1998)")
ax.axhline(early_mean, color="steelblue", linestyle="--", linewidth=1.2,
           label=f"Baseline mean: {early_mean:.1f} km²")

# Period divider
ax.axvline(1998.5, color="black", linestyle="--", linewidth=1.2, alpha=0.7,
           label="Period boundary")

# Data
ax.plot(YEARS_ALL, suitable_all, color="darkgreen", marker="o",
        markersize=4, linewidth=1.5, label="Annual suitable area (scenario 1)")

# Trend lines
ax.plot(YEARS_EARLY, np.poly1d(z_early)(YEARS_EARLY),
        color="steelblue", linewidth=2, linestyle=":",
        label=f"Early trend: {z_early[0]:+.2f} km²/yr")
ax.plot(YEARS_LATE, np.poly1d(z_late)(YEARS_LATE),
        color="tomato", linewidth=2, linestyle=":",
        label=f"Late trend: {z_late[0]:+.2f} km²/yr")

# Period mean for late period
ax.hlines(late_mean, 1999, 2018, colors="tomato", linewidth=1.5,
          linestyles="--", label=f"Late mean: {late_mean:.1f} km²")

# Shade periods
ax.axvspan(1979, 1998.5, alpha=0.04, color="steelblue")
ax.axvspan(1998.5, 2018, alpha=0.04, color="tomato")
ax.text(1988, ax.get_ylim()[0], "Early period", ha="center",
        fontsize=9, color="steelblue", alpha=0.8)
ax.text(2008, ax.get_ylim()[0], "Late period",  ha="center",
        fontsize=9, color="tomato",    alpha=0.8)

ax.set(title="Suitable Area 1979–2018 (Scenario 1)\nShaded band = baseline variability range",
       xlabel="Year", ylabel="Suitable area (km²)")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "climate_suitable_area_full_timeseries.png", dpi=150)
plt.close()
print("Saved: climate_suitable_area_full_timeseries.png")


# ── 4. CLASS COMPOSITION SHIFT ────────────────────────────────────────────────
# How much area is in each class, compared between periods

mean_early_cls = [np.mean([(s1_early_arr[i] == c).sum() * PIXEL_AREA_KM2
                             for i in range(len(YEARS_EARLY))]) for c in range(N_CLASSES)]
mean_late_cls  = [np.mean([(s1_late_arr[i]  == c).sum() * PIXEL_AREA_KM2
                             for i in range(len(YEARS_LATE))])  for c in range(N_CLASSES)]
shift = [mean_late_cls[c] - mean_early_cls[c] for c in range(N_CLASSES)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Side-by-side bars
x = np.arange(N_CLASSES)
w = 0.35
axes[0].bar(x - w/2, mean_early_cls, w, color=CLASS_COLORS, alpha=0.7,
            label="1979–1998", edgecolor="white")
axes[0].bar(x + w/2, mean_late_cls,  w, color=CLASS_COLORS, alpha=1.0,
            label="1999–2018", edgecolor="black", linewidth=0.5)
axes[0].set(title="Mean Area per Class by Period",
            xlabel="Suitability class", ylabel="Mean area (km²)")
axes[0].set_xticks(x)
axes[0].legend()
axes[0].grid(axis="y", alpha=0.3)

# Shift (late - early)
bar_colors = ["tomato" if s < 0 else "steelblue" for s in shift]
axes[1].bar(x, shift, color=bar_colors, alpha=0.8, edgecolor="white")
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set(title="Class Area Shift: Late − Early\n(blue = gain, red = loss)",
            xlabel="Suitability class", ylabel="Δ area (km²)")
axes[1].set_xticks(x)
axes[1].grid(axis="y", alpha=0.3)

plt.suptitle("Climate + Permafrost Signal: Class Composition Change", fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "climate_class_shift.png", dpi=150)
plt.close()
print("Saved: climate_class_shift.png")


# ── 5. PIXEL-LEVEL TRANSITION MATRIX (period means) ──────────────────────────
# Assign each pixel its "modal" (most common) class in each period
# then build a transition matrix

def modal_class(arr):
    """Most common class per pixel across years."""
    result = np.zeros(arr.shape[1:], dtype=np.int8)
    for r in range(arr.shape[1]):
        for c_col in range(arr.shape[2]):
            vals = arr[:, r, c_col].astype(int)
            result[r, c_col] = np.bincount(vals[~np.isnan(vals)].astype(int),
                                           minlength=N_CLASSES).argmax()
    return result

print("Computing modal class maps (may take a moment)...")
modal_early = modal_class(s1_early_arr)
modal_late  = modal_class(s1_late_arr)

trans_matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
for c1 in range(N_CLASSES):
    for c2 in range(N_CLASSES):
        trans_matrix[c1, c2] = int(((modal_early == c1) & (modal_late == c2)).sum())

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(trans_matrix, annot=True, fmt=",d", cmap="YlOrRd",
            xticklabels=range(N_CLASSES), yticklabels=range(N_CLASSES), ax=ax)
ax.set(title="Pixel Transition Matrix: Modal Class 1979–1998 → 1999–2018\n"
             "Row = early period class, Col = late period class",
       xlabel="Late period class", ylabel="Early period class")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "climate_transition_matrix.png", dpi=150)
plt.close()
print("Saved: climate_transition_matrix.png")


# ── 6. BOX PLOTS: PERIOD DISTRIBUTIONS ───────────────────────────────────────
early_area = suitable_all[EARLY_IDX]
late_area  = suitable_all[LATE_IDX]

fig, ax = plt.subplots(figsize=(7, 5))
bp = ax.boxplot([early_area, late_area], patch_artist=True,
                labels=["1979–1998", "1999–2018"], widths=0.5)
colors = ["steelblue", "tomato"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Overlay individual points
for i, (data, x_pos) in enumerate([(early_area, 1), (late_area, 2)]):
    ax.scatter(np.full(len(data), x_pos) + np.random.uniform(-0.05, 0.05, len(data)),
               data, color=colors[i], alpha=0.6, zorder=3, s=30)

ax.set(title="Distribution of Annual Suitable Area by Period\n(Scenario 1 — total signal)",
       ylabel="Suitable area (km²)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "climate_period_boxplots.png", dpi=150)
plt.close()
print("Saved: climate_period_boxplots.png")


# ── 7. STATISTICAL TESTS ─────────────────────────────────────────────────────
print("\n── Statistical Tests ─────────────────────────────────────────────────")

# Mann-Whitney U: are the two periods significantly different?
u_stat, p_val = stats.mannwhitneyu(early_area, late_area, alternative="two-sided")
print(f"\nMann-Whitney U (early vs late suitable area):")
print(f"  Early mean: {early_area.mean():.1f} km²")
print(f"  Late  mean: {late_area.mean():.1f} km²")
print(f"  Difference: {late_area.mean() - early_area.mean():+.1f} km²")
print(f"  U={u_stat:.2f}, p={p_val:.4f} → {'SIGNIFICANT' if p_val < 0.05 else 'not significant'} at α=0.05")

# Cohen's d effect size
pooled_std = np.sqrt((early_area.std()**2 + late_area.std()**2) / 2)
cohens_d   = (late_area.mean() - early_area.mean()) / pooled_std if pooled_std > 0 else 0
print(f"  Cohen's d: {cohens_d:.3f}  ({'large' if abs(cohens_d)>0.8 else 'medium' if abs(cohens_d)>0.5 else 'small'} effect)")

# Linear trend over full 40 years
z_all = np.polyfit(YEARS_ALL, suitable_all, 1)
_, p_trend = stats.spearmanr(YEARS_ALL, suitable_all)
print(f"\nSpearman rank correlation (area vs year, 1979–2018):")
print(f"  Trend: {z_all[0]:+.2f} km²/year")
print(f"  p={p_trend:.4f} → {'SIGNIFICANT' if p_trend < 0.05 else 'not significant'} trend")


# ── 8. SUMMARY CSV ───────────────────────────────────────────────────────────
records = []
for i, year in enumerate(YEARS_ALL):
    period = "early" if year < 1999 else "late"
    row = {"year": year, "period": period,
           "suitable_area_km2": float((s1[i] >= 2).sum() * PIXEL_AREA_KM2),
           "mean_class": float(np.nanmean(s1[i]))}
    for cls in range(N_CLASSES):
        row[f"class{cls}_area_km2"] = float((s1[i] == cls).sum() * PIXEL_AREA_KM2)
    records.append(row)

pd.DataFrame(records).to_csv(OUTPUT_DIR / "climate_summary.csv", index=False)
print("\nSaved: climate_summary.csv")
print(f"\n✓ Part 2 complete. Outputs saved to: {OUTPUT_DIR.resolve()}")