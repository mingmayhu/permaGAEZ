"""
Model Validation: Attainable Yield vs Actual Production Yield
=============================================================
Qilian Mountain Region — PyAEZ Output Validation

Compares four model outputs against actual rainfed yield for 4 crops
(Barley, Silage Maize, Rapeseed, Wheat):
  1. Permafrost-considered model  — mean 2001–2018 raw yield
  2. No-thaw counterfactual model — mean 2001–2018 raw yield
  3. Original FAO GAEZ model      — mean 2001–2018 raw yield
  4. FAO GAEZ v5                  — static attainable yield raster

Key note on interpretation:
  Both actual and modelled yields are under rainfed, low-input conditions.
  Validation checks:
    (a) Spatial correlation — does the model rank pixels correctly?
    (b) False negative rate — what % of actual cropland has zero modelled yield?
    (c) Yield gap ratio     — how close is modelled to actual at non-zero pixels?
    (d) Which model is closest to actual yield spatially?

Outputs (per crop subfolder + combined CSV):
  - validation_summary.csv          : metrics for all 4 models
  - scatter_plots.png               : actual vs modelled scatter per model
  - spatial_diff_maps.png           : maps of actual - modelled
  - yield_gap_map.png               : spatial yield gap ratio (permafrost model)
  - diff_actual_minus_*.tif         : difference rasters for GIS use
  - validation_summary_all_crops.csv: combined table across all crops and models
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from osgeo import gdal
from scipy import stats

# ─── Paths ──────────────────────────────────────────────────────────────────

BASE = "/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez"

# ── Crop definitions ─────────────────────────────────────────────────────────
# Each entry: (label, actual_filename, model_folder, fao_filename)
CROPS = [
    ("Barley",       "barley.tif",    "combined_barley",       "barley_yield.tif"),
    ("Silage Maize", "maize.tif",     "combined_silage_maize", "silage_maize_yield.tif"),
    ("Rapeseed",     "rapeseed.tif",  "combined_rape",         "rapeseed_yield.tif"),
    ("Wheat",        "wheat.tif",     "combined_wheat",        "wheat_yield.tif"),
]

# Years to average for permafrost, no-thaw, and original model outputs
AVG_YEARS = list(range(2001, 2019))  # 2001–2018 inclusive

# Static paths (mask/lake are shared across all crops)
MASK_PATH = os.path.join(BASE, "data_input/qilian mask.tif")
LAKE_PATH = os.path.join(BASE, "data_input/permafrost_qilian.tif")

def get_crop_paths(actual_f, model_folder, fao_f):
    """Return path dict for a single crop. Model entries point to folders —
    load_mean_yield() will load and average annual files within them."""
    return {
        "actual":     os.path.join(BASE, "data_input/actual_yield", actual_f),
        "permafrost": os.path.join(BASE, "data_output/final_classification", model_folder),
        "nothaw":     os.path.join(BASE, "data_output/final_classification_nothaw", model_folder),
        "original":   os.path.join(BASE, "data_output/original/final_classification", model_folder),
        "fao":        os.path.join(BASE, "data_input/gaez_v5", fao_f),
        "mask":       MASK_PATH,
        "lake":       LAKE_PATH,
    }

# The permafrost model output is the reference grid (0.1°).
REFERENCE_KEY = "permafrost"

# Actual production raster is in 1000 tonnes per pixel (as per FAO GAEZ v5 documentation).
# To convert to kg/ha of total pixel area (output density, same basis as model):
#   output density (kg/ha) = production (1000 t) × 1,000,000 (kg per 1000 t) ÷ pixel area (ha)
# Division by pixel area happens in run_crop().
# Actual production raster: raw pixel values are in units of 1000 tonnes (FAO scale factor = 1000).
# Full conversion to kg/ha of total pixel area:
#   raw value × 1000 (FAO scale factor → tonnes) × 1000 (tonnes → kg) ÷ pixel_area_ha
#   = raw value × 1,000,000 ÷ pixel_area_ha
ACTUAL_YIELD_SCALE = 1_000_000.0  # (×1000 scale factor) × (×1000 tonnes→kg)

OUTPUT_DIR = os.path.join(BASE, "results/validation/outputs")

# ─── Helpers ────────────────────────────────────────────────────────────────

def load_raster(path):
    """Load a GeoTIFF as a masked numpy array. NoData and zeros are masked."""
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(f"Cannot open raster: {path}")
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    arr = band.ReadAsArray().astype(float)
    # Mask nodata and non-positive values
    mask = np.zeros(arr.shape, dtype=bool)
    if nodata is not None:
        mask |= (arr == nodata)
    mask |= (arr <= 0)
    arr = np.ma.masked_array(arr, mask=mask)
    geo = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ds = None
    return arr, geo, proj


def load_mean_yield(folder, years, ref_arr, ref_geo):
    """
    Load and average annual raw yield rasters across a list of years.
    Files are expected at: {folder}/{year}_raw_yield.tif
    Missing years are skipped with a warning.
    Returns a masked array on the reference grid.
    """
    stacked = []
    missing = []
    for year in years:
        path = os.path.join(folder, f"{year}_raw_yield.tif")
        if not os.path.exists(path):
            missing.append(year)
            continue
        arr = align_rasters(ref_arr, ref_geo, path, mask_zeros=False)
        stacked.append(arr.data)

    if missing:
        print(f"  WARNING: Missing years in {os.path.basename(folder)}: {missing}")
    if not stacked:
        raise FileNotFoundError(f"No annual files found in: {folder}")

    stack = np.array(stacked)         # shape: (n_years, rows, cols)
    # Mean ignoring zeros (treat 0 as no-data for averaging, but keep in output)
    with np.errstate(invalid="ignore"):
        mean_arr = np.nanmean(np.where(stack > 0, stack, np.nan), axis=0)
    # Where mean is nan (all years were 0), set back to 0
    mean_arr = np.where(np.isfinite(mean_arr), mean_arr, 0.0)
    print(f"  Averaged {len(stacked)} years ({years[0]}–{years[-1]}): "
          f"{os.path.basename(folder)}")
    return mean_arr




def load_lake_mask(reference_geo, reference_shape):
    """
    Build a boolean keep-mask (True = valid, False = exclude) on the reference grid.
    Combines the study area mask with lake exclusion from permafrost_qilian.tif,
    matching the approach used in other analysis scripts.

    Pixels are excluded if:
      - Outside the study area mask (mask == 0 or nodata)
      - Lake pixels: value == 0 or nodata in permafrost_qilian.tif
    """
    rows, cols = reference_shape

    def _warp_to_ref(path):
        x_min = reference_geo[0]
        y_max = reference_geo[3]
        x_res = reference_geo[1]
        y_res = abs(reference_geo[5])
        x_max = x_min + cols * x_res
        y_min = y_max - rows * y_res
        ds = gdal.Warp(
            "", path, format="MEM",
            outputBounds=(x_min, y_min, x_max, y_max),
            xRes=x_res, yRes=y_res,
            resampleAlg=gdal.GRA_NearestNeighbour,
        )
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        arr = band.ReadAsArray().astype(float)
        arr[arr < -1e10] = np.nan
        if nodata is not None:
            arr[arr == nodata] = np.nan
        ds = None
        return arr

    # Study area mask
    mask_arr = _warp_to_ref(MASK_PATH)
    keep = np.isfinite(mask_arr) & (mask_arr != 0)

    # Lake exclusion
    pf_arr = _warp_to_ref(LAKE_PATH)
    lake_pixels = (pf_arr == 0) | ~np.isfinite(pf_arr)
    excluded = keep & lake_pixels
    keep[excluded] = False
    print(f"  Lake mask: excluded {excluded.sum()} lake/nodata pixels, "
          f"{keep.sum()} pixels remain")

    return keep



def get_raster_info(path):
    """Return (rows, cols, x_res, y_res, x_origin, y_origin, crs) for a raster."""
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(f"Cannot open raster: {path}")
    geo = ds.GetGeoTransform()
    info = {
        "rows":     ds.RasterYSize,
        "cols":     ds.RasterXSize,
        "x_res":    round(geo[1], 8),
        "y_res":    round(abs(geo[5]), 8),
        "x_origin": round(geo[0], 6),   # top-left x (longitude)
        "y_origin": round(geo[3], 6),   # top-left y (latitude)
        "crs":      ds.GetProjection(),
    }
    ds = None
    return info




def check_raster_alignment(paths_dict, tolerance=1e-5):
    """
    Check that all rasters share the same resolution and pixel origin.
    Prints a detailed report and returns True if all match, False otherwise.

    Args:
        paths_dict : dict of {label: path}
        tolerance  : floating point tolerance for coordinate comparison
    """
    print("\n" + "─" * 60)
    print("  RASTER ALIGNMENT CHECK")
    print("─" * 60)

    infos = {}
    for label, path in paths_dict.items():
        try:
            infos[label] = get_raster_info(path)
        except FileNotFoundError as e:
            print(f"  ✗ {label}: FILE NOT FOUND — {path}")
            return False

    # Print info table
    header = f"  {'Label':<22} {'Rows':>6} {'Cols':>6} {'X res':>10} {'Y res':>10} {'X origin':>12} {'Y origin':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, info in infos.items():
        print(f"  {label:<22} {info['rows']:>6} {info['cols']:>6} "
              f"{info['x_res']:>10.6f} {info['y_res']:>10.6f} "
              f"{info['x_origin']:>12.6f} {info['y_origin']:>12.6f}")

    # Compare each raster against the first (actual yield = reference)
    ref_label = list(infos.keys())[0]
    ref = infos[ref_label]
    all_ok = True
    issues = []

    for label, info in list(infos.items())[1:]:
        label_issues = []

        if abs(info["x_res"] - ref["x_res"]) > tolerance:
            label_issues.append(
                f"X resolution mismatch: {info['x_res']} vs reference {ref['x_res']}"
            )
        if abs(info["y_res"] - ref["y_res"]) > tolerance:
            label_issues.append(
                f"Y resolution mismatch: {info['y_res']} vs reference {ref['y_res']}"
            )
        if abs(info["x_origin"] - ref["x_origin"]) > tolerance:
            label_issues.append(
                f"X origin mismatch: {info['x_origin']} vs reference {ref['x_origin']}"
            )
        if abs(info["y_origin"] - ref["y_origin"]) > tolerance:
            label_issues.append(
                f"Y origin mismatch: {info['y_origin']} vs reference {ref['y_origin']}"
            )
        if info["rows"] != ref["rows"] or info["cols"] != ref["cols"]:
            label_issues.append(
                f"Shape mismatch: ({info['rows']}×{info['cols']}) "
                f"vs reference ({ref['rows']}×{ref['cols']})"
            )

        if label_issues:
            all_ok = False
            issues.append((label, label_issues))

    print()
    if all_ok:
        print("  ✓ All rasters have matching resolution and pixel origin.")
        print("  ✓ Proceeding with direct pixel-wise comparison.")
    else:
        print("  ⚠ Alignment issues detected — will auto-warp to reference grid:")
        for label, label_issues in issues:
            print(f"\n  [{label}]")
            for issue in label_issues:
                print(f"    • {issue}")
        print(
            "\n  Auto-warping will resample misaligned rasters to match the actual yield"
            "\n  raster grid using GDAL average resampling. Pixel-wise comparisons will"
            "\n  only be made at co-located valid pixels after alignment."
        )

    print("─" * 60 + "\n")
    return all_ok


def align_rasters(reference_arr, reference_geo, target_path, mask_zeros=True):
    """
    Warp target raster to match reference grid exactly using GDAL in-memory.

    Args:
        mask_zeros : if True, mask nodata and <= 0 values (use for actual yield).
                     if False, only mask nodata — keep 0 as valid (use for model output,
                     where 0 means 'not suitable' and is a meaningful value for spatial maps).
    """
    ref_rows, ref_cols = reference_arr.shape
    x_min = reference_geo[0]
    y_max = reference_geo[3]
    x_res = reference_geo[1]
    y_res = abs(reference_geo[5])
    x_max = x_min + ref_cols * x_res
    y_min = y_max - ref_rows * y_res

    warped_ds = gdal.Warp(
        "",
        target_path,
        format="MEM",
        outputBounds=(x_min, y_min, x_max, y_max),
        xRes=x_res,
        yRes=y_res,
        resampleAlg=gdal.GRA_Average,
    )
    band = warped_ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    arr = band.ReadAsArray().astype(float)
    arr[arr < -1e10] = np.nan  # catch large negative sentinels
    mask = ~np.isfinite(arr)
    if nodata is not None:
        mask |= (np.abs(arr - nodata) < 1e-6)
    if mask_zeros:
        mask |= (arr <= 0)
    warped_ds = None
    return np.ma.masked_array(arr, mask=mask)


def compute_metrics(actual, modelled, label):
    """
    Compute validation metrics between actual and modelled 1D arrays.
    modelled may contain zeros (false negatives — model predicted no yield).
    """
    pearson_r, pearson_p = stats.pearsonr(actual, modelled)
    spearman_r, spearman_p = stats.spearmanr(actual, modelled)
    rmse = np.sqrt(np.mean((modelled - actual) ** 2))
    mae = np.mean(np.abs(modelled - actual))
    bias = np.mean(modelled - actual)
    bias_pct = (bias / np.mean(actual)) * 100
    # Yield gap ratio only for pixels where model > 0 (avoid division by zero)
    # Both actual and modelled are rainfed low-input — ratio should be close to 1
    # Values >> 1 indicate model underprediction, values < 1 indicate overprediction
    modelled_nonzero = modelled[modelled > 0]
    actual_nonzero   = actual[modelled > 0]
    yield_gap_ratio  = (np.mean(actual_nonzero / modelled_nonzero)
                        if len(modelled_nonzero) > 0 else np.nan)
    false_neg_count  = int(np.sum(modelled == 0))
    false_neg_pct    = round(false_neg_count / len(actual) * 100, 1)

    return {
        "Model": label,
        "N pixels": len(actual),
        "False Negatives (n)": false_neg_count,
        "False Negatives (%)": false_neg_pct,
        "Mean Actual (kg/ha)": round(np.mean(actual), 2),
        "Mean Modelled (kg/ha)": round(np.mean(modelled), 2),
        "Pearson r": round(pearson_r, 4),
        "Pearson p": round(pearson_p, 4),
        "Spearman r": round(spearman_r, 4),
        "Spearman p": round(spearman_p, 4),
        "RMSE (kg/ha)": round(rmse, 2),
        "MAE (kg/ha)": round(mae, 2),
        "Mean Bias (kg/ha)": round(bias, 2),
        "Mean Bias (%)": round(bias_pct, 2),
        "Yield Gap Ratio (actual/mod)": round(yield_gap_ratio, 4) if not np.isnan(yield_gap_ratio) else "N/A",
    }


def get_valid_pairs(actual_arr, modelled_arr, min_pixels=2):
    """
    Return 1D arrays of co-located valid pixels for metric computation.
    Includes pixels where model = 0 (unsuitable) but actual yield > 0 —
    these are false negatives and should be penalized in RMSE/bias.
    Only excludes pixels where actual yield is masked/zero.
    Returns empty arrays if fewer than min_pixels overlap.
    """
    # Valid = actual yield exists (> 0) AND is within the study mask
    # Model value of 0 is kept — it means the model predicted no suitability
    combined_mask = actual_arr.mask | modelled_arr.mask
    a = actual_arr.data[~combined_mask]
    m = modelled_arr.data[~combined_mask]
    # Additional filter: actual must be positive (0 = no crop data)
    valid = a > 0
    a, m = a[valid], m[valid]
    if len(a) < min_pixels:
        return np.array([]), np.array([])
    return a, m


def save_raster_like(reference_path, out_path, array):
    """Save a numpy array as GeoTIFF matching the reference raster's grid."""
    ref_ds = gdal.Open(reference_path)
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(out_path, ref_ds.RasterXSize, ref_ds.RasterYSize, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
    out_ds.SetProjection(ref_ds.GetProjection())
    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(-9999)
    data = np.where(np.ma.getmaskarray(array), -9999, array.data).astype(np.float32)
    out_band.WriteArray(data)
    out_ds.FlushCache()
    out_ds = None
    ref_ds = None


# ─── Plotting ───────────────────────────────────────────────────────────────

def plot_scatter(actual_dict, modelled_dict, crop_label, out_path):
    """
    3-panel scatter plot: actual vs each model.
    actual_dict: {label: 1D array}, modelled_dict: {label: 1D array}
    """
    labels = list(modelled_dict.keys())
    colors = ["#2166ac", "#4dac26", "#d6604d"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{crop_label} — Actual vs Attainable Yield (2018)", fontsize=13, fontweight="bold")

    for ax, label, color in zip(axes, labels, colors):
        a = actual_dict[label]
        m = modelled_dict[label]
        r, p = stats.pearsonr(a, m)
        # Density scatter
        ax.scatter(m, a, alpha=0.4, s=10, color=color, linewidths=0)
        # 1:1 line
        lim_max = max(m.max(), a.max()) * 1.05
        ax.plot([0, lim_max], [0, lim_max], "k--", lw=1, label="1:1 line")
        # OLS fit line
        slope, intercept, _, _, _ = stats.linregress(m, a)
        x_fit = np.linspace(0, lim_max, 100)
        ax.plot(x_fit, slope * x_fit + intercept, color="red", lw=1.2, label=f"OLS (slope={slope:.2f})")
        ax.set_xlabel("Attainable Yield (kg/ha)", fontsize=10)
        ax.set_ylabel("Actual Yield (kg/ha)", fontsize=10)
        ax.set_title(f"{label}\nr = {r:.3f}, p = {p:.3f}", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlim(0, lim_max)
        ax.set_ylim(0, lim_max)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved scatter plot: {out_path}")


def plot_spatial_diff(actual_arr, model_arrs, geo, crop_label, out_path):
    """
    Spatial maps of actual - model for each of the 3 models.
    Blue = actual > model (model underestimates).
    Red  = actual < model (model overestimates).
    Grey background = valid model pixels with no actual yield data.
    """
    labels = list(model_arrs.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{crop_label} — Actual minus Attainable Yield (2018)", fontsize=13, fontweight="bold")

    # Diagnostic: report overlap counts
    for label, mod_arr in model_arrs.items():
        n_mod   = mod_arr.count()
        n_act   = actual_arr.count()
        n_both  = (~(actual_arr.mask | mod_arr.mask)).sum()
        print(f"  [{label}] model valid={n_mod}, actual valid={n_act}, overlap={n_both}")

    vmax = 0
    diff_arrs = {}
    for label, mod_arr in model_arrs.items():
        # Only compute diff where BOTH have valid data
        combined_mask = actual_arr.mask | mod_arr.mask
        diff = np.ma.masked_array(actual_arr.data - mod_arr.data, mask=combined_mask)
        diff_arrs[label] = diff
        if diff.count() > 0:
            vmax = max(vmax, np.abs(diff.compressed()).max())

    # Fall back so colorbar is never zero-range
    if vmax == 0:
        vmax = 1.0

    for ax, label in zip(axes, labels):
        mod_arr = model_arrs[label]
        diff    = diff_arrs[label]

        # Grey background: show all pixels where model has valid yield
        bg = np.where(~mod_arr.mask, 0.0, np.nan)
        ax.imshow(bg, cmap="Greys", vmin=-1, vmax=1,
                  interpolation="nearest", alpha=0.25)

        # Diff overlay: only where actual and model both valid
        im = ax.imshow(
            np.ma.masked_where(diff.mask, diff),
            cmap="RdBu",
            vmin=-vmax, vmax=vmax,
            interpolation="nearest",
        )
        n_overlap = diff.count()
        ax.set_title(f"{label}\n(n overlap pixels = {n_overlap})", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="kg/ha")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved spatial diff map: {out_path}")


def plot_yield_gap(actual_arr, permafrost_arr, crop_label, out_path):
    """
    Yield gap ratio map: actual / attainable (permafrost model).
    Values <1 indicate gap; closer to 1 = model matches actual well.
    """
    combined_mask = actual_arr.mask | permafrost_arr.mask
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            (~combined_mask) & (permafrost_arr.data > 0),
            actual_arr.data / permafrost_arr.data,
            np.nan,
        )
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(ratio, cmap="YlGn", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"{crop_label} — Yield Gap Ratio (Actual / Permafrost Model, 2018)", fontsize=11)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Actual / Attainable")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved yield gap map: {out_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def run_crop(crop_label, paths, lake_keep, ref_arr, ref_geo):
    """Run full validation pipeline for a single crop."""
    crop_dir = os.path.join(OUTPUT_DIR, crop_label.lower().replace(" ", "_"))
    os.makedirs(crop_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {crop_label}")
    print(f"{'='*60}")

    # Alignment check — use a representative single year file for folder-based models
    check_raster_alignment({
        "Actual yield":     paths["actual"],
        "Permafrost model": os.path.join(paths["permafrost"], "2001_raw_yield.tif"),
        "No-thaw model":    os.path.join(paths["nothaw"],     "2001_raw_yield.tif"),
        "Original model":   os.path.join(paths["original"],   "2001_raw_yield.tif"),
        "FAO GAEZ v5":      paths["fao"],
    })

    def load_and_mask(path, label, mask_zeros=True):
        arr = align_rasters(ref_arr, ref_geo, path, mask_zeros=mask_zeros)
        arr = np.ma.masked_array(arr.data, mask=(arr.mask | ~lake_keep))
        print(f"  {label}: {arr.count()} valid pixels after lake mask")
        return arr

    def load_mean_and_mask(folder, label):
        """Average annual files 2001–2018 then apply lake mask. Keeps zeros."""
        data = load_mean_yield(folder, AVG_YEARS, ref_arr, ref_geo)
        arr  = np.ma.masked_array(data, mask=~lake_keep)
        print(f"  {label}: {(~arr.mask).sum()} valid pixels after lake mask")
        return arr

    # Actual yield: production (tonnes/pixel) -> output density (kg/ha of total pixel area)
    # Pixel area in ha: x_res (degrees) * y_res (degrees) converted to km², then to ha.
    # Uses the reference geotransform; cos(lat) accounts for longitude compression at latitude.
    import math
    x_res_deg = abs(ref_geo[1])
    y_res_deg = abs(ref_geo[5])
    centre_lat = ref_geo[3] + (ref_arr.shape[0] / 2) * ref_geo[5]  # approx centre latitude
    km_per_deg_lon = 111.32 * math.cos(math.radians(centre_lat))
    km_per_deg_lat = 110.574
    pixel_area_km2 = x_res_deg * km_per_deg_lon * y_res_deg * km_per_deg_lat
    pixel_area_ha  = pixel_area_km2 * 100  # 1 km² = 100 ha
    print(f"  Pixel area: {pixel_area_km2:.2f} km² = {pixel_area_ha:.1f} ha")

    actual_arr = load_and_mask(paths["actual"], "Actual yield (production)", mask_zeros=True)
    actual_arr = np.ma.masked_array(
        actual_arr.data * ACTUAL_YIELD_SCALE / pixel_area_ha,
        mask=actual_arr.mask
    )
    if actual_arr.count() > 0:
        print(f"  Actual output density range: min={actual_arr.min():.1f}, "
              f"max={actual_arr.max():.1f}, mean={actual_arr.mean():.1f} kg/ha "
              f"(production ÷ pixel area)")

    # Model outputs: mean 2001–2018, keep zeros (0 = unsuitable)
    model_arrs = {
        "Permafrost Model": load_mean_and_mask(paths["permafrost"], "Permafrost model"),
        "No-Thaw Model":    load_mean_and_mask(paths["nothaw"],     "No-thaw model"),
        "Original Model":   load_mean_and_mask(paths["original"],   "Original model"),
        "FAO GAEZ v5":      load_and_mask(paths["fao"], "FAO GAEZ v5", mask_zeros=False),
    }

    # Metrics
    print("\nComputing validation metrics...")
    metrics_rows = []
    actual_dict  = {}
    modelled_dict = {}

    for label, mod_arr in model_arrs.items():
        a, m = get_valid_pairs(actual_arr, mod_arr)
        if len(a) == 0:
            print(f"  WARNING: [{label}] fewer than 2 overlapping valid pixels — skipping metrics.")
            continue
        actual_dict[label]   = a
        modelled_dict[label] = m
        row = compute_metrics(a, m, label)
        row["Crop"] = crop_label
        metrics_rows.append(row)
        print(f"\n  [{label}]")
        for k, v in row.items():
            if k not in ("Model", "Crop"):
                print(f"    {k}: {v}")

    # Per-crop CSV
    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(
            os.path.join(crop_dir, "validation_summary.csv"), index=False)

    # Plots
    print("\nGenerating plots...")
    if actual_dict:
        plot_scatter(actual_dict, modelled_dict, crop_label,
                     os.path.join(crop_dir, "scatter_plots.png"))
    plot_spatial_diff(actual_arr, model_arrs, ref_geo, crop_label,
                      os.path.join(crop_dir, "spatial_diff_maps.png"))
    if "Permafrost Model" in model_arrs:
        plot_yield_gap(actual_arr, model_arrs["Permafrost Model"], crop_label,
                       os.path.join(crop_dir, "yield_gap_map.png"))

    # Difference rasters
    ref_file = os.path.join(paths[REFERENCE_KEY], "2001_raw_yield.tif")
    print("Saving difference rasters...")
    for label, mod_arr in model_arrs.items():
        combined_mask = actual_arr.mask | mod_arr.mask
        diff = np.ma.masked_array(actual_arr.data - mod_arr.data, mask=combined_mask)
        safe_label = label.lower().replace(" ", "_")
        save_raster_like(ref_file,
                         os.path.join(crop_dir, f"diff_actual_minus_{safe_label}.tif"),
                         diff)

    return metrics_rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load reference grid and lake mask once — shared across all crops
    ref_paths = get_crop_paths(*CROPS[0][1:])
    ref_file  = os.path.join(ref_paths[REFERENCE_KEY], "2001_raw_yield.tif")
    print("Loading reference grid and lake mask...")
    ref_arr, ref_geo, _ = load_raster(ref_file)
    print(f"  Reference grid shape: {ref_arr.shape}")
    lake_keep = load_lake_mask(ref_geo, ref_arr.shape)

    # Run each crop
    all_metrics = []
    for crop_label, actual_f, model_folder, fao_f in CROPS:
        paths = get_crop_paths(actual_f, model_folder, fao_f)
        rows  = run_crop(crop_label, paths, lake_keep, ref_arr, ref_geo)
        all_metrics.extend(rows)

    # Combined summary CSV across all crops
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        # Reorder so Crop and Model are first columns
        cols = ["Crop", "Model"] + [c for c in df.columns if c not in ("Crop", "Model")]
        df[cols].to_csv(os.path.join(OUTPUT_DIR, "validation_summary_all_crops.csv"),
                        index=False)
        print(f"\n  Saved combined summary: {OUTPUT_DIR}/validation_summary_all_crops.csv")

    print(f"\n{'='*60}")
    print(f"  All crops complete. Results in: {OUTPUT_DIR}")
    print(f"{'='*60}\n")



if __name__ == "__main__":
    main()