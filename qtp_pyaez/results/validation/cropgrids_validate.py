"""
Suitability vs CROPGRIDS Harvested Area Validation
===================================================
Validates model suitability predictions against actual cropland extent
from CROPGRIDS harvested area data (independent of FAO GAEZ).

Three model scenarios compared:
  1. Permafrost-considered  — data_output/final_classification_fixed
  2. No-thaw counterfactual — data_output/final_classification_nothaw_fixed
  3. Original FAO model     — data_output/original/final_classification_fixed

Suitability classes:
  0/1 = not suitable (combined as "unsuitable")
  2–5 = suitable (increasing suitability)

Two comparisons per model × crop:
  A) Continuous: mean suitability class (1–5) vs CROPGRIDS harvested area fraction
  B) Binary:     suitable (mean class ≥ 2) vs CROPGRIDS farmed (harvested area > 0)
     → Precision, Recall, F1, confusion matrix

Outputs:
  results/validation/suitability/
    {crop}/
      metrics_summary.csv          — Spearman r + binary metrics per model
      confusion_matrix_{model}.png — confusion matrix heatmap
      scatter_{model}.png          — continuous suitability vs HA fraction
    suitability_validation_all_crops.csv — combined table
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from osgeo import gdal
from scipy import stats

# ── Config ────────────────────────────────────────────────────────────────────

BASE = "/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez"

CROPS = [
    ("Barley",   "combined_barley",   "barley_harvested_area.tif"),
    ("Wheat",    "combined_wheat",    "wheat_harvested_area.tif"),
    ("Rapeseed", "combined_rape",     "rapeseed_harvested_area.tif"),
]

MODELS = [
    ("Permafrost Model", "data_output/final_classification_fixed"),
    ("No-Thaw Model",    "data_output/final_classification_nothaw_fixed"),
    ("Original Model",   "data_output/original/final_classification_fixed"),
]

AVG_YEARS    = list(range(1999, 2019))   # 1999–2018 inclusive
SUITABLE_MIN = 2                          # classes >= this are "suitable"
MASK_PATH    = os.path.join(BASE, "data_input/qilian mask.tif")
LAKE_PATH    = os.path.join(BASE, "data_input/permafrost_qilian.tif")
CROPGRIDS_DIR = os.path.join(BASE, "data_input/cropgrids")
OUTPUT_DIR   = os.path.join(BASE, "results/validation/suitability")

# Reference grid — use first permafrost model barley file
REFERENCE_TIF = os.path.join(
    BASE, "data_output/final_classification_fixed/combined_barley/1999_suitability_class.tif")

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_raster_raw(path):
    """Load raster as float array with nodata -> nan."""
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(f"Cannot open: {path}")
    band   = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    arr    = band.ReadAsArray().astype(float)
    arr[arr < -1e10] = np.nan
    if nodata is not None:
        arr[np.abs(arr - nodata) < 1e-6] = np.nan
    ds = None
    return arr


def get_ref_geo():
    """Return (geo, shape) from reference raster."""
    ds  = gdal.Open(REFERENCE_TIF)
    geo = ds.GetGeoTransform()
    shp = (ds.RasterYSize, ds.RasterXSize)
    ds  = None
    return geo, shp


def warp_to_ref(path, ref_geo, ref_shape, resample=gdal.GRA_Average):
    """Warp any raster to the reference grid in memory."""
    rows, cols = ref_shape
    x_min = ref_geo[0]
    y_max = ref_geo[3]
    x_res = ref_geo[1]
    y_res = abs(ref_geo[5])
    x_max = x_min + cols * x_res
    y_min = y_max - rows * y_res
    ds = gdal.Warp("", path, format="MEM",
                   outputBounds=(x_min, y_min, x_max, y_max),
                   xRes=x_res, yRes=y_res,
                   resampleAlg=resample)
    band   = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    arr    = band.ReadAsArray().astype(float)
    arr[arr < -1e10] = np.nan
    if nodata is not None:
        arr[np.abs(arr - nodata) < 1e-6] = np.nan
    ds = None
    return arr


def build_lake_keep(ref_geo, ref_shape):
    """Build boolean keep mask: True = valid study area pixel."""
    mask_arr = warp_to_ref(MASK_PATH, ref_geo, ref_shape,
                           resample=gdal.GRA_NearestNeighbour)
    keep = np.isfinite(mask_arr) & (mask_arr != 0)
    pf_arr = warp_to_ref(LAKE_PATH, ref_geo, ref_shape,
                         resample=gdal.GRA_NearestNeighbour)
    lake = (pf_arr == 0) | ~np.isfinite(pf_arr)
    keep[keep & lake] = False
    print(f"  Lake mask: {keep.sum()} valid pixels")
    return keep


def load_mean_suitability(model_dir, crop_folder, years, ref_geo, ref_shape):
    """
    Load annual suitability class rasters, combine classes 0 and 1 as
    'not suitable' (set to 1), average across years.
    Returns mean class array (1.0–5.0) on reference grid.
    """
    stacked = []
    missing = []
    for year in years:
        path = os.path.join(BASE, model_dir, crop_folder,
                            f"{year}_suitability_class.tif")
        if not os.path.exists(path):
            missing.append(year)
            continue
        arr = warp_to_ref(path, ref_geo, ref_shape,
                          resample=gdal.GRA_NearestNeighbour)
        # Class 0 = outside mask/no data (nan), class 1 = not suitable, keep as is
        arr = np.where(np.isfinite(arr), arr, np.nan)
        stacked.append(arr)

    if missing:
        print(f"  WARNING: missing years {missing} in {model_dir}/{crop_folder}")
    if not stacked:
        raise FileNotFoundError(f"No files found: {model_dir}/{crop_folder}")

    stack    = np.array(stacked)
    mean_arr = np.nanmean(stack, axis=0)
    print(f"  Loaded {len(stacked)} years for {crop_folder} "
          f"({model_dir.split('/')[-1]}): mean class = {np.nanmean(mean_arr):.2f}")
    return mean_arr


def load_cropgrids_ha_fraction(ha_tif, ref_geo, ref_shape):
    """
    Load CROPGRIDS harvested area raster and convert to fraction of pixel area.
    Pixel area computed from geotransform at centre latitude.
    """
    arr = warp_to_ref(ha_tif, ref_geo, ref_shape, resample=gdal.GRA_Average)
    arr = np.where(arr < 0, np.nan, arr)   # remove nodata

    # Pixel area in ha
    x_res_deg  = abs(ref_geo[1])
    y_res_deg  = abs(ref_geo[5])
    rows, cols = ref_shape
    centre_lat = ref_geo[3] + (rows / 2) * ref_geo[5]
    km_lon     = 111.32 * math.cos(math.radians(centre_lat))
    km_lat     = 110.574
    pixel_ha   = x_res_deg * km_lon * y_res_deg * km_lat * 100
    print(f"  Pixel area: {pixel_ha:.0f} ha")

    ha_fraction = arr / pixel_ha
    ha_fraction = np.clip(ha_fraction, 0, 1)
    return ha_fraction, pixel_ha


# ── Metrics ───────────────────────────────────────────────────────────────────

def spearman_continuous(suit_arr, ha_frac, keep):
    """Spearman r between mean suitability class and HA fraction."""
    mask = keep & np.isfinite(suit_arr) & np.isfinite(ha_frac)
    if mask.sum() < 3:
        return np.nan, np.nan, 0
    r, p = stats.spearmanr(suit_arr[mask], ha_frac[mask])
    return round(r, 4), round(p, 4), int(mask.sum())


def binary_metrics(suit_arr, ha_frac, keep):
    """
    Binary confusion matrix metrics.
    Predicted positive: mean class >= SUITABLE_MIN
    Actual positive:    harvested area fraction > 0
    """
    mask = keep & np.isfinite(suit_arr) & np.isfinite(ha_frac)
    pred_pos = suit_arr >= SUITABLE_MIN   # model says suitable
    act_pos  = ha_frac  > 0              # CROPGRIDS says farmed

    tp = int(( pred_pos &  act_pos & mask).sum())
    fp = int(( pred_pos & ~act_pos & mask).sum())
    fn = int((~pred_pos &  act_pos & mask).sum())
    tn = int((~pred_pos & ~act_pos & mask).sum())
    n  = tp + fp + fn + tn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / n if n > 0 else 0.0

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Precision": round(precision, 4),
        "Recall":    round(recall,    4),
        "F1":        round(f1,        4),
        "Accuracy":  round(accuracy,  4),
        "N":         n,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_confusion_matrix(metrics, model_label, crop_label, out_path):
    """Plot a 2×2 confusion matrix heatmap."""
    cm = np.array([[metrics["TP"], metrics["FP"]],
                   [metrics["FN"], metrics["TN"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Suitable", "Pred Unsuitable"])
    ax.set_yticklabels(["Act Farmed", "Act Not Farmed"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14, color="white" if cm[i, j] > cm.max() * 0.6
                    else "black")
    ax.set_title(f"{crop_label} — {model_label}\n"
                 f"P={metrics['Precision']:.2f}  "
                 f"R={metrics['Recall']:.2f}  "
                 f"F1={metrics['F1']:.2f}", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_scatter(suit_arr, ha_frac, keep, r, p, model_label, crop_label,
                 out_path):
    """Scatter plot: mean suitability class vs HA fraction."""
    mask = keep & np.isfinite(suit_arr) & np.isfinite(ha_frac)
    x = suit_arr[mask]
    y = ha_frac[mask]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, alpha=0.4, s=12, color="#2166ac", linewidths=0)
    ax.set_xlabel("Mean Suitability Class (1–5)", fontsize=11)
    ax.set_ylabel("CROPGRIDS Harvested Area Fraction", fontsize=11)
    ax.set_title(f"{crop_label} — {model_label}\n"
                 f"Spearman r = {r:.3f}, p = {p:.3f}  (n={mask.sum()})",
                 fontsize=10)
    # Add class boundary line
    ax.axvline(x=SUITABLE_MIN - 0.5, color="red", linestyle="--",
               lw=1, label=f"Suitable threshold (class ≥ {SUITABLE_MIN})")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_spatial(suit_arr, ha_frac, keep, crop_label, model_label, out_path):
    """Side-by-side spatial maps: suitability class and HA fraction."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{crop_label} — {model_label} (mean 1999–2018 vs CROPGRIDS 2020)",
                 fontsize=11, fontweight="bold")

    suit_plot = np.where(keep, suit_arr, np.nan)
    ha_plot   = np.where(keep, ha_frac,  np.nan)

    ha_valid = ha_plot[np.isfinite(ha_plot) & (ha_plot > 0)]
    vmax_ha  = np.percentile(ha_valid, 95) if len(ha_valid) > 0 else 0.01
    vmax_ha  = max(vmax_ha, 1e-4)  # avoid zero range

    im0 = axes[0].imshow(suit_plot, cmap="RdYlGn", vmin=0, vmax=5,
                         interpolation="nearest")
    axes[0].set_title("Mean Suitability Class", fontsize=10)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04,
                 label="Class (0=no data, 1=unsuitable, 5=very suitable)")

    im1 = axes[1].imshow(ha_plot, cmap="YlGn", vmin=0, vmax=vmax_ha,
                         interpolation="nearest")
    axes[1].set_title(f"CROPGRIDS Harvested Area Fraction (max={vmax_ha:.4f})",
                      fontsize=10)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04,
                 label="Fraction of pixel area harvested")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ref_geo, ref_shape = get_ref_geo()
    print("\nBuilding lake mask...")
    lake_keep = build_lake_keep(ref_geo, ref_shape)

    all_rows = []

    for crop_label, crop_folder, cropgrids_file in CROPS:
        print(f"\n{'='*60}")
        print(f"  {crop_label}")
        print(f"{'='*60}")

        crop_dir = os.path.join(OUTPUT_DIR, crop_label.lower())
        os.makedirs(crop_dir, exist_ok=True)

        # Load CROPGRIDS harvested area fraction
        ha_tif = os.path.join(CROPGRIDS_DIR, cropgrids_file)
        if not os.path.exists(ha_tif):
            print(f"  SKIP: CROPGRIDS file not found: {ha_tif}")
            continue
        ha_frac, pixel_ha = load_cropgrids_ha_fraction(ha_tif, ref_geo, ref_shape)
        ha_frac_masked = np.where(lake_keep, ha_frac, np.nan)

        print(f"  CROPGRIDS: {(ha_frac_masked > 0).sum()} farmed pixels, "
              f"mean fraction = {np.nanmean(ha_frac_masked[ha_frac_masked > 0]):.4f}")

        for model_label, model_dir in MODELS:
            print(f"\n  [{model_label}]")

            try:
                suit_arr = load_mean_suitability(
                    model_dir, crop_folder, AVG_YEARS, ref_geo, ref_shape)
            except FileNotFoundError as e:
                print(f"  SKIP: {e}")
                continue

            suit_masked = np.where(lake_keep, suit_arr, np.nan)
            farmed_mask = lake_keep & np.isfinite(suit_masked) & (ha_frac_masked > 0)
            print(f"Mean suitability at farmed pixels: {np.nanmean(suit_masked[farmed_mask]):.3f}")
            print(f"Class distribution at farmed pixels: {np.unique(np.round(suit_masked[farmed_mask]), return_counts=True)}")

            # A) Continuous Spearman correlation
            r, p, n = spearman_continuous(suit_masked, ha_frac_masked, lake_keep)
            print(f"    Spearman r = {r}, p = {p}, n = {n}")

            # B) Binary metrics
            bm = binary_metrics(suit_masked, ha_frac_masked, lake_keep)
            print(f"    Precision={bm['Precision']}, Recall={bm['Recall']}, "
                  f"F1={bm['F1']}, Accuracy={bm['Accuracy']}")
            print(f"    TP={bm['TP']} FP={bm['FP']} FN={bm['FN']} TN={bm['TN']}")

            # Save row
            safe_label = model_label.lower().replace(" ", "_")
            all_rows.append({
                "Crop":          crop_label,
                "Model":         model_label,
                "N pixels":      bm["N"],
                "Spearman r":    r,
                "Spearman p":    p,
                "TP":            bm["TP"],
                "FP":            bm["FP"],
                "FN":            bm["FN"],
                "TN":            bm["TN"],
                "Precision":     bm["Precision"],
                "Recall":        bm["Recall"],
                "F1":            bm["F1"],
                "Accuracy":      bm["Accuracy"],
            })

            # Plots
            plot_confusion_matrix(
                bm, model_label, crop_label,
                os.path.join(crop_dir, f"confusion_{safe_label}.png"))

            plot_scatter(
                suit_masked, ha_frac_masked, lake_keep, r, p,
                model_label, crop_label,
                os.path.join(crop_dir, f"scatter_{safe_label}.png"))

            plot_spatial(
                suit_masked, ha_frac_masked, lake_keep,
                crop_label, model_label,
                os.path.join(crop_dir, f"spatial_{safe_label}.png"))

        # Per-crop CSV
        crop_rows = [r for r in all_rows if r["Crop"] == crop_label]
        if crop_rows:
            pd.DataFrame(crop_rows).to_csv(
                os.path.join(crop_dir, "metrics_summary.csv"), index=False)

    # Combined CSV
    if all_rows:
        df = pd.DataFrame(all_rows)
        cols = ["Crop", "Model", "N pixels", "Spearman r", "Spearman p",
                "Precision", "Recall", "F1", "Accuracy",
                "TP", "FP", "FN", "TN"]
        df[cols].to_csv(
            os.path.join(OUTPUT_DIR, "suitability_validation_all_crops.csv"),
            index=False)
        print(f"\n  Saved combined: {OUTPUT_DIR}/suitability_validation_all_crops.csv")

    print(f"\n{'='*60}")
    print(f"  Complete. Results in: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()