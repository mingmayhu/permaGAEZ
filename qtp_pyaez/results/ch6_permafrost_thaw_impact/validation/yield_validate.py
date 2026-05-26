"""
Model Validation: Attainable Yield vs Actual Production Output Density
=======================================================================
Qilian Mountain Region — PyAEZ Output Validation

Compares three model outputs against actual production output density
from two independent sources:
  A) FAO GAEZ v5 Production (RES06-PRD) — barley, rapeseed, wheat only
       raw values in 1000 tonnes (scale factor 1000)
       → raw × 1,000,000 ÷ pixel_area_ha = kg/ha total pixel area
  B) Monfreda et al. (1997–2003) Production — all 5 crops
       raw values in tonnes
       → raw × 1000 ÷ pixel_area_ha = kg/ha total pixel area

Three model scenarios:
  1. Permafrost-considered model  — mean 2001–2018 raw yield
  2. No-thaw counterfactual model — mean 2001–2018 raw yield
  3. Original FAO GAEZ model      — mean 2001–2018 raw yield
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from osgeo import gdal
from scipy import stats

# ── Config ────────────────────────────────────────────────────────────────────

BASE         = "/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez"
FAO_PROD_DIR = os.path.join(BASE, "data_input/actual_prod")
MRF_BASE     = "/Users/ming-mayhu/Downloads/HarvestedAreaYield175Crops_Geotiff/GeoTiff"

# (crop_label, model_folder, fao_prod_file_or_None, monfreda_subfolder)
# fao_prod_file: filename in FAO_PROD_DIR, or None if not available
# monfreda_subfolder: subfolder name under MRF_BASE (e.g. "barley")
CROPS = [
    ("Barley",     "combined_barley",    "barley.tif",   "barley"),
    ("Spring Oat", "combined_oat",       None,           "oats"),
    ("Dry Pea",    "combined_dry_pea",   None,           "pea"),
    ("Rapeseed",   "combined_rape",      "rapeseed.tif", "rapeseed"),
    ("Wheat",      "combined_wheat",     "wheat.tif",    "wheat"),
    ("Potato", "combined_white_potato", None, "potato"),
    ("Silage Maize", "combined_silage_maize", None, "maizefor")
]

MODELS = [
    ("Permafrost Model", "data_output/final_classification"),
    ("No-Thaw Model",    "data_output/final_classification_nothaw"),
    ("Original Model",   "data_output/original/final_classification"),
]

AVG_YEARS_RECENT   = list(range(1999, 2019))  # 1999–2018: compared against FAO (2019–2021 stats)
AVG_YEARS_BASELINE = list(range(1979, 1999))  # 1979–1998: compared against Monfreda (1997–2003)

# Models to skip for Monfreda baseline comparison (e.g. no-thaw lacks 1979–1998 files)
MONFREDA_SKIP_MODELS = {"No-Thaw Model"}
MASK_PATH  = os.path.join(BASE, "data_input/qilian mask.tif")
LAKE_PATH  = os.path.join(BASE, "data_input/permafrost_qilian.tif")
OUTPUT_DIR = os.path.join(BASE, "results/validation/outputs")

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_raster(path):
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(f"Cannot open: {path}")
    band   = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    arr    = band.ReadAsArray().astype(float)
    mask   = np.zeros(arr.shape, dtype=bool)
    if nodata is not None:
        mask |= (arr == nodata)
    mask |= (arr <= 0)
    arr  = np.ma.masked_array(arr, mask=mask)
    geo  = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ds   = None
    return arr, geo, proj


def align_rasters(reference_arr, reference_geo, target_path, mask_zeros=True):
    ref_rows, ref_cols = reference_arr.shape
    x_min = reference_geo[0]
    y_max = reference_geo[3]
    x_res = reference_geo[1]
    y_res = abs(reference_geo[5])
    x_max = x_min + ref_cols * x_res
    y_min = y_max - ref_rows * y_res
    warped_ds = gdal.Warp(
        "", target_path, format="MEM",
        outputBounds=(x_min, y_min, x_max, y_max),
        xRes=x_res, yRes=y_res,
        resampleAlg=gdal.GRA_Average,
    )
    band   = warped_ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    arr    = band.ReadAsArray().astype(float)
    arr[arr < -1e10] = np.nan
    mask   = ~np.isfinite(arr)
    if nodata is not None:
        mask |= (np.abs(arr - nodata) < 1e-6)
    if mask_zeros:
        mask |= (arr <= 0)
    warped_ds = None
    return np.ma.masked_array(arr, mask=mask)


def load_mean_yield(folder, years, ref_arr, ref_geo):
    stacked = []
    missing = []
    corrupt = []
    for year in years:
        path = os.path.join(folder, f"{year}_raw_yield.tif")
        if not os.path.exists(path):
            missing.append(year)
            continue
        # Check file is valid before warping
        test_ds = gdal.Open(path)
        if test_ds is None:
            corrupt.append(year)
            continue
        test_ds = None
        arr = align_rasters(ref_arr, ref_geo, path, mask_zeros=False)
        stacked.append(arr.data)
    if missing:
        print(f"  WARNING: Missing years in {os.path.basename(folder)}: {missing}")
    if corrupt:
        print(f"  WARNING: Corrupt/unreadable files skipped: {corrupt}")
    if not stacked:
        raise FileNotFoundError(f"No valid annual files found in: {folder}")
    stack    = np.array(stacked)
    with np.errstate(invalid="ignore"):
        mean_arr = np.nanmean(np.where(stack > 0, stack, np.nan), axis=0)
    mean_arr = np.where(np.isfinite(mean_arr), mean_arr, 0.0)
    print(f"  Averaged {len(stacked)} years ({years[0]}–{years[-1]}): "
          f"{os.path.basename(folder)}")
    return mean_arr


def load_lake_mask(reference_geo, reference_shape):
    rows, cols = reference_shape

    def _warp(path):
        x_min = reference_geo[0];  y_max = reference_geo[3]
        x_res = reference_geo[1];  y_res = abs(reference_geo[5])
        ds = gdal.Warp("", path, format="MEM",
                       outputBounds=(x_min, y_max - rows*y_res,
                                     x_min + cols*x_res, y_max),
                       xRes=x_res, yRes=y_res,
                       resampleAlg=gdal.GRA_NearestNeighbour)
        band   = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        arr    = band.ReadAsArray().astype(float)
        arr[arr < -1e10] = np.nan
        if nodata is not None:
            arr[np.abs(arr - nodata) < 1e-6] = np.nan
        ds = None
        return arr

    keep  = np.isfinite(_warp(MASK_PATH)) & (_warp(MASK_PATH) != 0)
    pf    = _warp(LAKE_PATH)
    lakes = (pf == 0) | ~np.isfinite(pf)
    keep[keep & lakes] = False
    print(f"  Lake mask: {keep.sum()} valid pixels remain")
    return keep


def get_raster_info(path):
    ds  = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(f"Cannot open: {path}")
    geo = ds.GetGeoTransform()
    info = {
        "rows": ds.RasterYSize, "cols": ds.RasterXSize,
        "x_res": round(geo[1], 8), "y_res": round(abs(geo[5]), 8),
        "x_origin": round(geo[0], 6), "y_origin": round(geo[3], 6),
    }
    ds = None
    return info


def check_raster_alignment(paths_dict, tolerance=1e-5):
    print("\n" + "─"*60)
    print("  RASTER ALIGNMENT CHECK")
    print("─"*60)
    infos = {}
    for label, path in paths_dict.items():
        try:
            infos[label] = get_raster_info(path)
        except FileNotFoundError:
            print(f"  ✗ {label}: FILE NOT FOUND — {path}")
            return False
    header = f"  {'Label':<22} {'Rows':>6} {'Cols':>6} {'X res':>10} {'Y res':>10} {'X origin':>12} {'Y origin':>12}"
    print(header)
    print("  " + "-"*(len(header)-2))
    for label, info in infos.items():
        print(f"  {label:<22} {info['rows']:>6} {info['cols']:>6} "
              f"{info['x_res']:>10.6f} {info['y_res']:>10.6f} "
              f"{info['x_origin']:>12.6f} {info['y_origin']:>12.6f}")
    ref = list(infos.values())[0]
    issues = []
    for label, info in list(infos.items())[1:]:
        li = []
        if abs(info["x_res"]    - ref["x_res"])    > tolerance: li.append(f"X res mismatch: {info['x_res']} vs {ref['x_res']}")
        if abs(info["y_res"]    - ref["y_res"])    > tolerance: li.append(f"Y res mismatch: {info['y_res']} vs {ref['y_res']}")
        if abs(info["x_origin"] - ref["x_origin"]) > tolerance: li.append(f"X origin mismatch: {info['x_origin']} vs {ref['x_origin']}")
        if abs(info["y_origin"] - ref["y_origin"]) > tolerance: li.append(f"Y origin mismatch: {info['y_origin']} vs {ref['y_origin']}")
        if info["rows"] != ref["rows"] or info["cols"] != ref["cols"]: li.append(f"Shape mismatch: ({info['rows']}×{info['cols']}) vs ({ref['rows']}×{ref['cols']})")
        if li:
            issues.append((label, li))
    print()
    if not issues:
        print("  ✓ All rasters aligned — proceeding with direct comparison.")
    else:
        print("  ⚠ Misalignment detected — auto-warping to reference grid:")
        for label, li in issues:
            print(f"\n  [{label}]")
            for i in li: print(f"    • {i}")
    print("─"*60 + "\n")


def compute_metrics(actual, modelled, label):
    pearson_r,  pearson_p  = stats.pearsonr(actual, modelled)
    spearman_r, spearman_p = stats.spearmanr(actual, modelled)
    rmse = np.sqrt(np.mean((modelled - actual)**2))
    mae  = np.mean(np.abs(modelled - actual))
    bias = np.mean(modelled - actual)
    bias_pct = (bias / np.mean(actual)) * 100
    mod_nz   = modelled[modelled > 0]
    act_nz   = actual[modelled > 0]
    ygr      = np.mean(act_nz / mod_nz) if len(mod_nz) > 0 else np.nan
    fn_n     = int(np.sum(modelled == 0))
    fn_pct   = round(fn_n / len(actual) * 100, 1)
    return {
        "Model": label,
        "N pixels": len(actual),
        "False Negatives (n)": fn_n,
        "False Negatives (%)": fn_pct,
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
        "Yield Gap Ratio (actual/mod)": round(ygr, 4) if not np.isnan(ygr) else "N/A",
    }


def get_valid_pairs(actual_arr, modelled_arr, min_pixels=2):
    combined_mask = actual_arr.mask | modelled_arr.mask
    a = actual_arr.data[~combined_mask]
    m = modelled_arr.data[~combined_mask]
    valid = a > 0
    a, m = a[valid], m[valid]
    if len(a) < min_pixels:
        return np.array([]), np.array([])
    return a, m


def save_raster_like(ref_path, out_path, array):
    ref_ds  = gdal.Open(ref_path)
    driver  = gdal.GetDriverByName("GTiff")
    out_ds  = driver.Create(out_path, ref_ds.RasterXSize, ref_ds.RasterYSize,
                             1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
    out_ds.SetProjection(ref_ds.GetProjection())
    band = out_ds.GetRasterBand(1)
    band.SetNoDataValue(-9999)
    band.WriteArray(np.where(np.ma.getmaskarray(array), -9999,
                             array.data).astype(np.float32))
    out_ds.FlushCache()
    out_ds = None;  ref_ds = None


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_scatter(actual_dict, modelled_dict, crop_label, out_path):
    labels = list(modelled_dict.keys())
    colors = ["#2166ac", "#4dac26", "#d6604d"]
    fig, axes = plt.subplots(1, len(labels), figsize=(5*len(labels), 5))
    if len(labels) == 1:
        axes = [axes]
    fig.suptitle(f"{crop_label} — Actual vs Attainable Yield",
                 fontsize=13, fontweight="bold")
    for ax, label, color in zip(axes, labels, colors):
        a = actual_dict[label];  m = modelled_dict[label]
        r, p = stats.pearsonr(a, m)
        ax.scatter(m, a, alpha=0.4, s=10, color=color, linewidths=0)
        lim_max = max(m.max(), a.max()) * 1.05
        ax.plot([0, lim_max], [0, lim_max], "k--", lw=1, label="1:1 line")
        slope, intercept, *_ = stats.linregress(m, a)
        x_fit = np.linspace(0, lim_max, 100)
        ax.plot(x_fit, slope*x_fit+intercept, color="red", lw=1.2,
                label=f"OLS (slope={slope:.2f})")
        ax.set_xlabel("Attainable Yield (kg/ha)", fontsize=10)
        ax.set_ylabel("Actual Yield (kg/ha)", fontsize=10)
        ax.set_title(f"{label}\nr={r:.3f}, p={p:.3f}", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlim(0, lim_max);  ax.set_ylim(0, lim_max)
        ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved scatter: {out_path}")


def plot_spatial_diff(actual_arr, model_arrs, crop_label, out_path):
    labels = list(model_arrs.keys())
    fig, axes = plt.subplots(1, len(labels), figsize=(6*len(labels), 5))
    if len(labels) == 1:
        axes = [axes]
    fig.suptitle(f"{crop_label} — Actual minus Attainable Yield",
                 fontsize=13, fontweight="bold")
    vmax = 0
    diff_arrs = {}
    for label, mod_arr in model_arrs.items():
        diff = np.ma.masked_array(actual_arr.data - mod_arr.data,
                                  mask=actual_arr.mask | mod_arr.mask)
        diff_arrs[label] = diff
        if diff.count() > 0:
            vmax = max(vmax, np.abs(diff.compressed()).max())
    vmax = vmax or 1.0
    for ax, label in zip(axes, labels):
        mod_arr = model_arrs[label]
        diff    = diff_arrs[label]
        bg = np.where(~mod_arr.mask, 0.0, np.nan)
        ax.imshow(bg, cmap="Greys", vmin=-1, vmax=1,
                  interpolation="nearest", alpha=0.25)
        im = ax.imshow(np.ma.masked_where(diff.mask, diff),
                       cmap="RdBu", vmin=-vmax, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(f"{label}\n(n={diff.count()})", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="kg/ha")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved spatial diff: {out_path}")


def plot_yield_gap(actual_arr, permafrost_arr, crop_label, out_path):
    combined_mask = actual_arr.mask | permafrost_arr.mask
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            (~combined_mask) & (permafrost_arr.data > 0),
            actual_arr.data / permafrost_arr.data, np.nan)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(ratio, cmap="YlGn", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_title(f"{crop_label} — Yield Gap Ratio (Actual / Permafrost Model)",
                 fontsize=11)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Actual / Attainable")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved yield gap map: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def load_actual_production(path, scale_kg_per_raw, pixel_ha,
                           ref_arr, ref_geo, lake_keep, label):
    """
    Load a production raster and convert to kg/ha of total pixel area.
    scale_kg_per_raw: multiply raw value by this to get kg/pixel
    pixel_ha: total pixel area in hectares
    """
    arr = align_rasters(ref_arr, ref_geo, path, mask_zeros=True)
    arr = np.ma.masked_array(arr.data, mask=(arr.mask | ~lake_keep))
    arr = np.ma.masked_array(arr.data * scale_kg_per_raw / pixel_ha,
                             mask=arr.mask)
    print(f"  {label}: {arr.count()} valid pixels, "
          f"mean={arr.mean():.1f} kg/ha" if arr.count() > 0
          else f"  {label}: 0 valid pixels")
    return arr


def run_crop_source(crop_label, model_folder, actual_path, source_label,
                    scale_kg_per_raw, pixel_ha, lake_keep, ref_arr, ref_geo,
                    avg_years, skip_models=None):
    """Run validation for one crop × one actual data source × all models.
    skip_models: set of model labels to exclude (e.g. {'No-Thaw Model'})
    """
    if skip_models is None:
        skip_models = set()
    """Run validation for one crop × one actual data source × all models."""
    crop_dir = os.path.join(OUTPUT_DIR,
                            crop_label.lower().replace(" ", "_"),
                            source_label.lower().replace(" ", "_"))
    os.makedirs(crop_dir, exist_ok=True)

    period_label = f"{avg_years[0]}–{avg_years[-1]}"
    print(f"\n  [{source_label}] — model period {period_label}")

    actual_arr = load_actual_production(
        actual_path, scale_kg_per_raw, pixel_ha,
        ref_arr, ref_geo, lake_keep, "Actual production")
    if actual_arr.count() == 0:
        print("  No valid actual production pixels — skipping.")
        return []

    model_paths = {
        label: os.path.join(BASE, model_dir, model_folder)
        for label, model_dir in MODELS
        if label not in skip_models
    }

    def load_mean_and_mask(folder, label):
        data = load_mean_yield(folder, avg_years, ref_arr, ref_geo)
        arr  = np.ma.masked_array(data, mask=~lake_keep)
        print(f"  {label}: {(~arr.mask).sum()} valid pixels")
        return arr

    model_arrs = {
        label: load_mean_and_mask(path, label)
        for label, path in model_paths.items()
    }

    # Metrics
    metrics_rows = []
    actual_dict  = {}
    modelled_dict = {}
    for label, mod_arr in model_arrs.items():
        a, m = get_valid_pairs(actual_arr, mod_arr)
        if len(a) == 0:
            print(f"  WARNING: [{label}] fewer than 2 valid pixels — skipping.")
            continue
        actual_dict[label]   = a
        modelled_dict[label] = m
        row = compute_metrics(a, m, label)
        row["Crop"]          = crop_label
        row["Source"]        = source_label
        row["Model Period"]  = period_label
        metrics_rows.append(row)
        print(f"\n    [{label}]")
        for k, v in row.items():
            if k not in ("Model", "Crop", "Source"):
                print(f"      {k}: {v}")

    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(
            os.path.join(crop_dir, "validation_summary.csv"), index=False)

    # Plots
    if actual_dict:
        plot_scatter(actual_dict, modelled_dict, f"{crop_label} ({source_label})",
                     os.path.join(crop_dir, "scatter_plots.png"))
    plot_spatial_diff(actual_arr, model_arrs,
                      f"{crop_label} ({source_label})",
                      os.path.join(crop_dir, "spatial_diff_maps.png"))
    if "Permafrost Model" in model_arrs:
        plot_yield_gap(actual_arr, model_arrs["Permafrost Model"],
                       f"{crop_label} ({source_label})",
                       os.path.join(crop_dir, "yield_gap_map.png"))

    # Difference rasters
    ref_file = os.path.join(model_paths["Permafrost Model"], "2001_raw_yield.tif")
    for label, mod_arr in model_arrs.items():
        diff = np.ma.masked_array(actual_arr.data - mod_arr.data,
                                  mask=actual_arr.mask | mod_arr.mask)
        safe = label.lower().replace(" ", "_")
        save_raster_like(ref_file,
                         os.path.join(crop_dir,
                                      f"diff_actual_minus_{safe}.tif"), diff)
    return metrics_rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Reference grid from first crop's permafrost model
    ref_file = os.path.join(BASE, "data_output/final_classification",
                            CROPS[0][1], "2001_raw_yield.tif")
    print("Loading reference grid and lake mask...")
    ref_arr, ref_geo, _ = load_raster(ref_file)
    print(f"  Reference grid: {ref_arr.shape}")
    lake_keep = load_lake_mask(ref_geo, ref_arr.shape)

    # Pixel area (shared across all crops)
    x_res_deg  = abs(ref_geo[1])
    y_res_deg  = abs(ref_geo[5])
    centre_lat = ref_geo[3] + (ref_arr.shape[0] / 2) * ref_geo[5]
    pixel_ha   = (x_res_deg * 111.32 * math.cos(math.radians(centre_lat))
                  * y_res_deg * 110.574 * 100)
    print(f"  Pixel area: {pixel_ha:.0f} ha")

    all_metrics = []

    for crop_label, model_folder, fao_file, mrf_folder in CROPS:
        print(f"\n{'='*60}\n  {crop_label}\n{'='*60}")

        # ── FAO GAEZ v5 production — recent period (1999–2018)
        if fao_file is not None:
            fao_path = os.path.join(FAO_PROD_DIR, fao_file)
            if os.path.exists(fao_path):
                rows = run_crop_source(
                    crop_label, model_folder, fao_path,
                    "FAO Production",
                    scale_kg_per_raw=1_000_000.0,
                    pixel_ha=pixel_ha,
                    lake_keep=lake_keep, ref_arr=ref_arr, ref_geo=ref_geo,
                    avg_years=AVG_YEARS_RECENT)
                all_metrics.extend(rows)
            else:
                print(f"  SKIP FAO: file not found: {fao_path}")

        # ── Monfreda production — baseline period (1979–1998)
        mrf_path = os.path.join(
            MRF_BASE, mrf_folder, f"{mrf_folder}_Production.tif")
        if os.path.exists(mrf_path):
            rows = run_crop_source(
                crop_label, model_folder, mrf_path,
                "Monfreda (2000)",
                scale_kg_per_raw=1000.0,
                pixel_ha=pixel_ha,
                lake_keep=lake_keep, ref_arr=ref_arr, ref_geo=ref_geo,
                avg_years=AVG_YEARS_BASELINE,
                skip_models=MONFREDA_SKIP_MODELS)
            all_metrics.extend(rows)
        else:
            print(f"  SKIP Monfreda: file not found: {mrf_path}")

    # Combined CSV
    if all_metrics:
        df   = pd.DataFrame(all_metrics)
        cols = ["Crop", "Source", "Model Period", "Model"] + [
            c for c in df.columns if c not in ("Crop", "Source", "Model Period", "Model")]
        df[cols].to_csv(
            os.path.join(OUTPUT_DIR, "validation_summary_all_crops.csv"),
            index=False)
        print(f"\n  Saved: {OUTPUT_DIR}/validation_summary_all_crops.csv")

    print(f"\n{'='*60}\n  Complete. Results in: {OUTPUT_DIR}\n{'='*60}\n")


if __name__ == "__main__":
    main()