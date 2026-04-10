"""
Permafrost Thaw Impact Analysis on Winter Barley Suitability
============================================================
Compares two scenarios of yield classification GeoTIFFs:
  - Scenario 1 (control):    1979-2018 permafrost data throughout
  - Scenario 2 (experiment): 1999-2018 permafrost held fixed (correct climate)

Assumes one GeoTIFF per year per scenario, named e.g.:
  scenario1/yield_class_1979.tif ... yield_class_2018.tif
  scenario2/yield_class_1979.tif ... yield_class_2018.tif

Adjust SCENARIO1_DIR, SCENARIO2_DIR, and the filename pattern to match your files.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import rasterio
from pathlib import Path
from scipy import stats
import seaborn as sns
from osgeo import gdal

# ─────────────────────────────────────────────
# CONFIGURATION — edit these to match your data
# ─────────────────────────────────────────────
SCENARIO1_DIR = Path("./data_output/final_classification/combined_wheat")   # control: original permafrost
SCENARIO2_DIR = Path("./data_output/final_classification_nothaw/combined_wheat")   # experiment: fixed permafrost
OUTPUT_DIR    = Path("./analysis_output/combined_wheat")
OUTPUT_DIR.mkdir(exist_ok=True)

YEARS = list(range(1999, 2019))     # 1979–2018
N_CLASSES = 6                        # classes 0–5
PIXEL_AREA_KM2 = 1.0                 # set to your pixel area in km² (e.g. 0.09 for 300m)

# Filename pattern — edit to match your naming convention
def tif_path(scenario_dir: Path, year: int) -> Path:
    if scenario_dir == SCENARIO2_DIR and year in range(1979, 1999):
        scenario_dir = SCENARIO1_DIR  # use control files for pre-1999 years in the experiment scenario
    
    return scenario_dir / f"{year}_raw_yield.tif"


# ─────────────────────────────────────────────
# STEP 1: Load all GeoTIFFs into 3D arrays
# ─────────────────────────────────────────────
def load_scenario(scenario_dir: Path) -> tuple[np.ndarray, dict]:
    arrays = []
    meta = None
    
    # load mask once outside the loop
    mask_ds = gdal.Open("./data_input/qilian mask.tif")
    mask = mask_ds.ReadAsArray().astype(float)
    
    for year in YEARS:
        path = tif_path(scenario_dir, year)
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            arr[np.isclose(arr, -999.0)] = 0.0  # treat nodata as 0
            arr[mask == 0] = np.nan           # mask outside study area as NaN
            arrays.append(arr)
            if meta is None:
                meta = src.meta
    
    return np.stack(arrays, axis=0), meta

print("Loading scenario 1 (control)...")
s1, meta = load_scenario(SCENARIO1_DIR)

print("Loading scenario 2 (experiment — fixed permafrost)...")
s2, _    = load_scenario(SCENARIO2_DIR)

print(f"Array shape: {s1.shape}  (years={len(YEARS)}, rows, cols)")


# ─────────────────────────────────────────────
# STEP 2: Difference array
# ─────────────────────────────────────────────
diff = s1 - s2
# shape: (40, rows, cols)
# Positive = suitability INCREASED when permafrost thaw removed
# Negative = suitability DECREASED when permafrost thaw removed


# ─────────────────────────────────────────────
# STEP 3: Area per class per year
# ─────────────────────────────────────────────
def count_classes(arr: np.ndarray) -> pd.DataFrame:
    """Count pixels per class per year."""
    records = []
    for i, year in enumerate(YEARS):
        layer = arr[i]
        for cls in range(N_CLASSES):
            records.append({"year": year, "class": cls,
                            "pixels": int((layer == cls).sum()),
                            "area_km2": int((layer == cls).sum()) * PIXEL_AREA_KM2})
    return pd.DataFrame(records)

df1 = count_classes(s1)
df2 = count_classes(s2)
df1["scenario"] = "Control (orig. permafrost)"
df2["scenario"] = "Experiment (fixed permafrost)"
df_all = pd.concat([df1, df2])

# Plot: suitable area (class >= 2) over time
fig, ax = plt.subplots(figsize=(12, 5))
for label, df in [("Control", df1), ("Experiment", df2)]:
    suitable = df[df["class"] >= 2].groupby("year")["area_km2"].sum()
    ax.plot(suitable.index, suitable.values, marker="o", markersize=4, label=label)
ax.set(title="Total Suitable Area (Class ≥ 2) Over Time",
       xlabel="Year", ylabel=f"Area (km²)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "suitable_area_timeseries.png", dpi=150)
plt.close()
print("Saved: suitable_area_timeseries.png")

# Plot: stacked bar of class areas per year (both scenarios side by side)
class_colors = ["#1f4e79", "#2196F3", "#4CAF50", "#CDDC39", "#FF9800", "#00BCD4"]
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for ax, (label, df) in zip(axes, [("Control", df1), ("Experiment", df2)]):
    pivot = df.pivot(index="year", columns="class", values="area_km2").fillna(0)
    pivot.plot(kind="bar", stacked=True, ax=ax, color=class_colors, legend=False, width=0.85)
    ax.set_title(f"{label}: Area by Suitability Class")
    ax.set_ylabel("Area (km²)")
    ax.grid(axis="y", alpha=0.3)
axes[0].legend(title="Class", labels=[str(c) for c in range(N_CLASSES)], loc="upper left")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "class_area_stacked_bar.png", dpi=150)
plt.close()
print("Saved: class_area_stacked_bar.png")


# ─────────────────────────────────────────────
# STEP 4: Difference maps (per year)
# ─────────────────────────────────────────────
diff_vals = np.unique(diff)
vmax = max(abs(diff.min()), abs(diff.max()), 1)
cmap = plt.cm.RdBu_r  # red = degradation, blue = improvement

n_cols = 8
n_rows = int(np.ceil(len(YEARS) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 2.5))
axes = axes.flatten()
cmap = plt.cm.RdBu_r.copy()
cmap.set_bad(color='lightgrey')

for i, year in enumerate(YEARS):
    im = axes[i].imshow(diff[i], cmap=cmap, vmin=-5, vmax=5, interpolation="nearest")
    axes[i].set_title(str(year), fontsize=8)
    axes[i].axis("off")

for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.colorbar(im, ax=axes[:len(YEARS)], orientation="vertical",
             fraction=0.01, pad=0.01, label="Class change (Exp − Control)")
plt.suptitle("Difference Maps: Experiment − Control\n(Blue = improved, Red = degraded with permafrost thaw removed)",
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "difference_maps.png", dpi=150)
plt.close()
print("Saved: difference_maps.png")


# ─────────────────────────────────────────────
# STEP 5: Transition matrices (per year)
# ─────────────────────────────────────────────
# Aggregate transition matrix across all years
agg_matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
for i in range(len(YEARS)):
    for c1 in range(N_CLASSES):
        mask = s1[i] == c1
        for c2 in range(N_CLASSES):
            agg_matrix[c1, c2] += int(((s2[i] == c2) & mask).sum())

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(agg_matrix, annot=True, fmt=",d", cmap="YlOrRd",
            xticklabels=range(N_CLASSES), yticklabels=range(N_CLASSES), ax=ax)
ax.set(title="Aggregate Transition Matrix (all years)\nRow = Control class, Col = Experiment class",
       xlabel="Experiment class", ylabel="Control class")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "transition_matrix_aggregate.png", dpi=150)
plt.close()
print("Saved: transition_matrix_aggregate.png")

# Also save per-year transition matrices as a CSV summary
records = []
for i, year in enumerate(YEARS):
    mat = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for c1 in range(N_CLASSES):
        mask = s1[i] == c1
        for c2 in range(N_CLASSES):
            mat[c1, c2] = int(((s2[i] == c2) & mask).sum())
    changed = mat.sum() - np.diag(mat).sum()
    improved = int(np.triu(mat, k=1).sum())   # experiment class > control class
    degraded  = int(np.tril(mat, k=-1).sum())
    records.append({"year": year, "unchanged_px": int(np.diag(mat).sum()),
                    "changed_px": int(changed), "improved_px": improved, "degraded_px": degraded})

pd.DataFrame(records).to_csv(OUTPUT_DIR / "yearly_transition_summary.csv", index=False)
print("Saved: yearly_transition_summary.csv")


# ─────────────────────────────────────────────
# STEP 6: Elevation stratification
# (Requires a DEM GeoTIFF aligned to your data)
# ─────────────────────────────────────────────
DEM_PATH = Path("dem.tif")  # <-- set to your DEM path

if DEM_PATH.exists():
    with rasterio.open(DEM_PATH) as src:
        dem = src.read(1).astype(np.float32)

    ELEV_BANDS = [(2000, 2500), (2500, 3000), (3000, 3500), (3500, 4000), (4000, 5000)]
    elev_records = []
    for low, high in ELEV_BANDS:
        elev_mask = (dem >= low) & (dem < high)
        for i, year in enumerate(YEARS):
            for scenario, arr in [("control", s1), ("experiment", s2)]:
                for cls in range(N_CLASSES):
                    px = int(((arr[i] == cls) & elev_mask).sum())
                    elev_records.append({"year": year, "elev_band": f"{low}-{high}m",
                                         "scenario": scenario, "class": cls, "pixels": px})

    df_elev = pd.DataFrame(elev_records)
    df_elev.to_csv(OUTPUT_DIR / "elevation_stratified.csv", index=False)

    # Plot suitable area by elevation band
    fig, axes = plt.subplots(1, len(ELEV_BANDS), figsize=(18, 4), sharey=True)
    for ax, (low, high) in zip(axes, ELEV_BANDS):
        band_label = f"{low}-{high}m"
        for scenario, color in [("control", "steelblue"), ("experiment", "tomato")]:
            sub = df_elev[(df_elev["elev_band"] == band_label) &
                          (df_elev["scenario"] == scenario) &
                          (df_elev["class"] >= 2)]
            area = sub.groupby("year")["pixels"].sum() * PIXEL_AREA_KM2
            ax.plot(area.index, area.values, color=color, label=scenario, linewidth=1.5)
        ax.set_title(band_label, fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlabel("Year")
    axes[0].set_ylabel("Suitable area (km²)")
    axes[0].legend(fontsize=8)
    plt.suptitle("Suitable Area by Elevation Band: Control vs Experiment")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "elevation_stratified_area.png", dpi=150)
    plt.close()
    print("Saved: elevation_stratified_area.png")
else:
    print("No DEM found — skipping elevation stratification. Set DEM_PATH to your DEM file.")


# ─────────────────────────────────────────────
# STEP 7: Statistical summary
# ─────────────────────────────────────────────
for i in range(len(YEARS)):
    mask = (s1[i] > 0) | (s2[i] > 0)
    print(f"{YEARS[i]}: n_suitable={mask.sum()}")

mean_s1, mean_s2 = [], []
suitable_s1, suitable_s2 = [], []
for i in range(len(YEARS)):
    mask = (s1[i] > 0) | (s2[i] > 0)
    m1 = s1[i][mask]
    m2 = s2[i][mask]
    mean_s1.append(np.nanmean(m1) if mask.sum() > 0 else np.nan)
    mean_s2.append(np.nanmean(m2) if mask.sum() > 0 else np.nan)
    suitable_s1.append(int((s1[i] > 0).sum()))
    suitable_s2.append(int((s2[i] > 0).sum()))

# Test 1: mean suitability class
t_stat, p_val = stats.wilcoxon(mean_s1, mean_s2)
print(f"\nWilcoxon (mean class): W={t_stat:.2f}, p={p_val:.4f}")
print(f"  {'Significant' if p_val < 0.05 else 'Not significant'} at α=0.05")

# Test 2: total suitable pixel count
t_stat2, p_val2 = stats.wilcoxon(suitable_s1, suitable_s2)
print(f"\nWilcoxon (suitable pixel count): W={t_stat2:.2f}, p={p_val2:.4f}")
print(f"  {'Significant' if p_val2 < 0.05 else 'Not significant'} at α=0.05")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(YEARS, mean_s1, label="Control", marker="o", markersize=4)
ax.plot(YEARS, mean_s2, label="Experiment", marker="s", markersize=4)
ax.fill_between(YEARS, mean_s1, mean_s2, alpha=0.15, color="purple", label="Gap")
ax.set(title=f"Mean Suitability Class (suitable pixels only)\nWilcoxon p={p_val:.4f}",
       xlabel="Year", ylabel="Mean class")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mean_class_timeseries.png", dpi=150)
plt.close()
print("Saved: mean_class_timeseries.png")

print("\n✓ All outputs saved to:", OUTPUT_DIR.resolve())

CROPS = ['combined_barley', 'combined_winter_barley', 'combined_spring_rape','combined_spring_wheat','combined_silage_maize', 'combined_wheat', 'combined_white_potato', 'combined_oat', 'combined_rape', 'combined_dry_pea']  # add your crops

all_s1, all_s2 = [], []

for crop in CROPS:
    s1_crop, _ = load_scenario(Path(f"./data_output/final_classification/{crop}"))
    s2_crop, _ = load_scenario(Path(f"./data_output/final_classification_nothaw/{crop}"))
    all_s1.append(s1_crop)
    all_s2.append(s2_crop)

# Stack along a new crop axis: shape (n_crops, n_years, rows, cols)
all_s1 = np.stack(all_s1, axis=0)
all_s2 = np.stack(all_s2, axis=0)

# Combined suitable pixel count across all crops per year
suitable_s1_combined = []
suitable_s2_combined = []
for i in range(len(YEARS)):
    # a pixel counts if ANY crop is suitable
    combined_mask_s1 = np.any(all_s1[:, i, :, :] > 0, axis=0)
    combined_mask_s2 = np.any(all_s2[:, i, :, :] > 0, axis=0)
    suitable_s1_combined.append(combined_mask_s1.sum())
    suitable_s2_combined.append(combined_mask_s2.sum())

for i, year in enumerate(YEARS):
    print(f"{year}: s1={suitable_s1_combined[i]}, s2={suitable_s2_combined[i]}, "
          f"diff={suitable_s1_combined[i]-suitable_s2_combined[i]}")
print(f"\nMean s1: {np.mean(suitable_s1_combined):.1f}")
print(f"Mean s2: {np.mean(suitable_s2_combined):.1f}")

t_stat, p_val = stats.wilcoxon(suitable_s1_combined, suitable_s2_combined)
print(f"Combined crops Wilcoxon: W={t_stat:.2f}, p={p_val:.4f}")

fig, axes = plt.subplots(len(CROPS) + 1, 1, figsize=(14, 4 * (len(CROPS) + 1)), sharex=True)

for ax, crop in zip(axes, CROPS):
    s1_crop, _ = load_scenario(Path(f"./data_output/final_classification/{crop}"))
    s2_crop, _ = load_scenario(Path(f"./data_output/final_classification_nothaw/{crop}"))
    
    suit_s1 = [(s1_crop[i] > 0).sum() for i in range(len(YEARS))]
    suit_s2 = [(s2_crop[i] > 0).sum() for i in range(len(YEARS))]
    
    x = np.arange(len(YEARS))
    width = 0.35
    ax.bar(x - width/2, suit_s1, width, label='Control (with thaw)', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, suit_s2, width, label='Experiment (fixed permafrost)', color='tomato', alpha=0.8)
    ax.set_title(crop, fontsize=11)
    ax.set_ylabel('Suitable pixels')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(YEARS, rotation=45)

# Combined panel
ax = axes[-1]
x = np.arange(len(YEARS))
ax.bar(x - width/2, suitable_s1_combined, width, label='Control (with thaw)', color='steelblue', alpha=0.8)
ax.bar(x + width/2, suitable_s2_combined, width, label='Experiment (fixed permafrost)', color='tomato', alpha=0.8)
ax.set_title('All crops combined', fontsize=11)
ax.set_ylabel('Suitable pixels')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)
ax.set_xticks(x)
ax.set_xticklabels(YEARS, rotation=45)

plt.suptitle('Suitable Pixels: Control vs Experiment by Crop', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'suitable_pixels_per_crop.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: suitable_pixels_per_crop.png")

print("\nWilcoxon test per crop:")
for crop in CROPS:
    s1_crop, _ = load_scenario(Path(f"./data_output/final_classification/{crop}"))
    s2_crop, _ = load_scenario(Path(f"./data_output/final_classification_nothaw/{crop}"))
    
    suit_s1 = [(s1_crop[i] > 0).sum() for i in range(len(YEARS))]
    suit_s2 = [(s2_crop[i] > 0).sum() for i in range(len(YEARS))]
    
    t_stat, p_val = stats.wilcoxon(suit_s1, suit_s2)
    direction = "↑ thaw increases suitability" if np.mean(suit_s1) > np.mean(suit_s2) else "↓ thaw decreases suitability"
    print(f"  {crop}: W={t_stat:.2f}, p={p_val:.4f} "
          f"{'*significant*' if p_val < 0.05 else 'not significant'} — {direction} "
          f"(mean control={np.mean(suit_s1):.1f}, experiment={np.mean(suit_s2):.1f})")

t_stat_combined, p_val_combined = stats.wilcoxon(suitable_s1_combined, suitable_s2_combined)
print(f"  combined:  W={t_stat_combined:.2f}, p={p_val_combined:.4f} "
      f"{'*significant*' if p_val_combined < 0.05 else 'not significant'}")