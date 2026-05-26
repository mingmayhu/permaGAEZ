"""
Monfreda Data Quality Check
============================
Checks the DataQuality_Yield raster for each crop used in validation.

Quality values:
  1.00 = county-level census data      (best)
  0.75 = state-level census data
  0.50 = interpolated within 2° lat/lon
  0.25 = country-level census data
  0.00 = missing census data           (worst)

Extracts quality values for pixels within the Qilian Mountain study area
and reports the distribution of quality levels.
"""

import os
import numpy as np
from osgeo import gdal

# ── Config ────────────────────────────────────────────────────────────────────

BASE      = "/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez"
MRF_BASE  = "/Users/ming-mayhu/Downloads/HarvestedAreaYield175Crops_Geotiff/GeoTiff"
CGRIDS_DIR = os.path.join(BASE, "data_input/cropgrids")
MASK_PATH = os.path.join(BASE, "data_input/qilian mask.tif")
LAKE_PATH = os.path.join(BASE, "data_input/permafrost_qilian.tif")

REFERENCE_TIF = os.path.join(
    BASE, "data_output/final_classification/combined_barley/2001_raw_yield.tif")

# (label, monfreda_subfolder, cropgrids_quality_tif_or_None)
CROPS = [
    ("Barley",       "barley",    "barley_quality.tif"),
    ("Spring Oat",   "oats",      "spring oat_quality.tif"),
    ("Dry Pea",      "pea",       "dry pea_quality.tif"),
    ("Rapeseed",     "rapeseed",  "rapeseed_quality.tif"),
    ("Wheat",        "wheat",     "wheat_quality.tif"),
    ("White Potato", "potato",      "white potato_quality.tif"),
    ("Silage Maize", "maizefor", "silage maize_quality.tif"),
]

# Monfreda quality levels
MONFREDA_QUALITY_LABELS = {
    1.00: "county-level census",
    0.75: "state-level census",
    0.50: "interpolated (within 2°)",
    0.25: "country-level census",
    0.00: "missing data",
}

# CROPGRIDS quality is a continuous 0–1 score
# Values closer to 1 = higher quality / better agreement with statistics

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_ref_info():
    ds  = gdal.Open(REFERENCE_TIF)
    geo = ds.GetGeoTransform()
    shp = (ds.RasterYSize, ds.RasterXSize)
    ds  = None
    return geo, shp


def warp_to_ref(path, ref_geo, ref_shape, resample=gdal.GRA_NearestNeighbour):
    rows, cols = ref_shape
    x_min = ref_geo[0];  y_max = ref_geo[3]
    x_res = ref_geo[1];  y_res = abs(ref_geo[5])
    ds = gdal.Warp("", path, format="MEM",
                   outputBounds=(x_min, y_max - rows*y_res,
                                 x_min + cols*x_res, y_max),
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


def build_mask(ref_geo, ref_shape):
    mask_arr = warp_to_ref(MASK_PATH, ref_geo, ref_shape)
    keep = np.isfinite(mask_arr) & (mask_arr != 0)
    pf   = warp_to_ref(LAKE_PATH, ref_geo, ref_shape)
    keep[keep & ((pf == 0) | ~np.isfinite(pf))] = False
    return keep


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ref_geo, ref_shape = get_ref_info()
    print("Building study area mask...")
    keep = build_mask(ref_geo, ref_shape)
    n_pixels = int(keep.sum())
    print(f"  Study area: {n_pixels} valid pixels\n")

    print("=" * 65)
    print("  DATA QUALITY CHECK — Qilian Mountain Region")
    print("=" * 65)

    all_results = []

    for crop_label, mrf_folder, cg_qual_file in CROPS:
        print(f"\n{'─'*65}")
        print(f"  {crop_label}")
        print(f"{'─'*65}")

        row = {"Crop": crop_label}

        # ── Monfreda quality ─────────────────────────────────────────
        if mrf_folder is not None:
            mrf_path = os.path.join(MRF_BASE, mrf_folder,
                                    f"{mrf_folder}_DataQuality_Yield.tif")
            if os.path.exists(mrf_path):
                arr     = warp_to_ref(mrf_path, ref_geo, ref_shape)
                quality = arr[keep]
                quality = quality[np.isfinite(quality)]
                quality_rounded = np.round(quality * 4) / 4

                print(f"\n  Monfreda (1997–2003) — N={len(quality)} pixels")
                print(f"  Mean quality: {np.mean(quality):.3f}")
                for q_val in [1.00, 0.75, 0.50, 0.25, 0.00]:
                    count = int(np.sum(quality_rounded == q_val))
                    pct   = count / len(quality) * 100
                    label = MONFREDA_QUALITY_LABELS[q_val]
                    mark  = " ✓" if q_val >= 0.75 else (" ⚠" if q_val == 0.5 else " ✗")
                    print(f"    {q_val:.2f}  {label:<35} {count:>5} ({pct:>5.1f}%){mark}")

                mean_q = np.mean(quality)
                if mean_q >= 0.75:   mrf_assess = "HIGH"
                elif mean_q >= 0.50: mrf_assess = "MODERATE"
                elif mean_q >= 0.25: mrf_assess = "LOW"
                else:                mrf_assess = "VERY LOW"
                print(f"  → Assessment: {mrf_assess}")

                row.update({
                    "Monfreda mean quality":     round(mean_q, 3),
                    "Monfreda % county (1.0)":   round(np.sum(quality_rounded == 1.00) / len(quality) * 100, 1),
                    "Monfreda % state (0.75)":   round(np.sum(quality_rounded == 0.75) / len(quality) * 100, 1),
                    "Monfreda % interp (0.50)":  round(np.sum(quality_rounded == 0.50) / len(quality) * 100, 1),
                    "Monfreda % country (0.25)": round(np.sum(quality_rounded == 0.25) / len(quality) * 100, 1),
                    "Monfreda % missing (0.0)":  round(np.sum(quality_rounded == 0.00) / len(quality) * 100, 1),
                    "Monfreda assessment":        mrf_assess,
                })
            else:
                print(f"  Monfreda: file not found — {mrf_path}")
        else:
            print(f"  Monfreda: not available for {crop_label}")

        # ── CROPGRIDS quality ─────────────────────────────────────────
        if cg_qual_file is not None:
            cg_path = os.path.join(CGRIDS_DIR, cg_qual_file)
            if os.path.exists(cg_path):
                arr     = warp_to_ref(cg_path, ref_geo, ref_shape,
                                      resample=gdal.GRA_Average)
                quality = arr[keep]
                quality = quality[np.isfinite(quality) & (quality >= 0)]

                print(f"\n  CROPGRIDS (2020) — N={len(quality)} pixels")
                print(f"  Mean quality score: {np.mean(quality):.3f}")
                print(f"  Min: {np.min(quality):.3f}  Max: {np.max(quality):.3f}")

                # Bin into quartiles for readability
                bins = [(0.75, 1.00, "high (0.75–1.0)"),
                        (0.50, 0.75, "moderate (0.50–0.75)"),
                        (0.25, 0.50, "low (0.25–0.50)"),
                        (0.00, 0.25, "very low (0.00–0.25)")]
                for lo, hi, label in bins:
                    count = int(np.sum((quality >= lo) & (quality <= hi)))
                    pct   = count / len(quality) * 100
                    mark  = " ✓" if lo >= 0.75 else (" ⚠" if lo >= 0.50 else " ✗")
                    print(f"    {label:<28} {count:>5} ({pct:>5.1f}%){mark}")

                mean_q = np.mean(quality)
                if mean_q >= 0.75:   cg_assess = "HIGH"
                elif mean_q >= 0.50: cg_assess = "MODERATE"
                elif mean_q >= 0.25: cg_assess = "LOW"
                else:                cg_assess = "VERY LOW"
                print(f"  → Assessment: {cg_assess}")

                row.update({
                    "CROPGRIDS mean quality": round(mean_q, 3),
                    "CROPGRIDS % high":       round(np.sum(quality >= 0.75) / len(quality) * 100, 1),
                    "CROPGRIDS % moderate":   round(np.sum((quality >= 0.50) & (quality < 0.75)) / len(quality) * 100, 1),
                    "CROPGRIDS % low":        round(np.sum((quality >= 0.25) & (quality < 0.50)) / len(quality) * 100, 1),
                    "CROPGRIDS % very low":   round(np.sum(quality < 0.25) / len(quality) * 100, 1),
                    "CROPGRIDS assessment":   cg_assess,
                })
            else:
                print(f"  CROPGRIDS quality: file not found — {cg_path}")
                print(f"  (Run cropgrids_to_tif.py first to generate quality rasters)")

        all_results.append(row)

    # Summary
    print(f"\n{'='*65}")
    print("  SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Crop':<15} {'Monfreda Q':>12}  {'Monfreda':>10}  {'CROPGRIDS Q':>12}  {'CROPGRIDS':>10}")
    print(f"  {'----':<15} {'----------':>12}  {'--------':>10}  {'-----------':>12}  {'---------':>10}")
    for r in all_results:
        mq = f"{r.get('Monfreda mean quality', 'N/A')}"
        ma = r.get('Monfreda assessment', 'N/A')
        cq = f"{r.get('CROPGRIDS mean quality', 'N/A')}"
        ca = r.get('CROPGRIDS assessment', 'N/A')
        print(f"  {r['Crop']:<15} {mq:>12}  {ma:>10}  {cq:>12}  {ca:>10}")

    # Save CSV
    import pandas as pd
    out_path = os.path.join(BASE, "results/validation/outputs/data_quality_check.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(all_results).to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()