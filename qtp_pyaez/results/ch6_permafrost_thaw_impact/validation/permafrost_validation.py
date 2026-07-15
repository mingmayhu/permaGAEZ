"""
compare_nieer.py
----------------
Compares SHAW-derived ALT and permafrost presence against the NIEER reference
dataset (Zhang et al. 2000) for the overlapping period 2000–2016.

Outputs printed to console (no figures).

ALT comparison:
  SHAW  : nanmax over day axis per year → nanmean across 2000–2016
  NIEER : single-band mean ALT raster, reprojected to Qilian grid

Permafrost presence comparison:
  SHAW  : modal binary value (1 = permafrost) across 2000–2016
  FAO   : modal class across 2000–2016, binarised (i/ii = permafrost)
  NIEER : MAGT ≤ 0 °C = permafrost (reprojected to Qilian grid)

All comparisons are masked to valid Qilian agricultural pixels
(qilian_mask_new.tif > 0, excluding lake pixels where
permafrost_qilian.tif == 0 or NaN).
"""

import os
import numpy as np
from scipy import stats
from osgeo import gdal, osr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = "/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez"

MASK_PATH        = os.path.join(BASE, "data_input/qilian_mask_new.tif")
PERM_MASK_PATH   = os.path.join(BASE, "data_input/permafrost_qilian.tif")

SHAW_ALT_DIR     = os.path.join(BASE, "data_input/permafrost_yearly")   # {year}/active_layer_depth.npy
SHAW_PERM_DIR    = os.path.join(BASE, "data_output/module1/permafrost_maps")  # permafrost_{year}.npy
FAO_PERM_DIR     = os.path.join(BASE, "data_output/original/module1")   # {year}/permafrost.tif

NIEER_ALT_PATH   = "/Users/ming-mayhu/Desktop/NIEER_permafrost_dataset_released/NIEER_permafrost_dataset_released/NIEER_ALT.tif"
NIEER_MAGT_PATH  = "/Users/ming-mayhu/Desktop/NIEER_permafrost_dataset_released/NIEER_permafrost_dataset_released/NIEER_MAGT.tif"
NIEER_PROB_PATH  = "/Users/ming-mayhu/Desktop/NIEER_permafrost_dataset_released/NIEER_permafrost_dataset_released/NIEER_Probability.tif"

YEARS = list(range(2000, 2017))   # 2000–2016 inclusive

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_tif(path):
    """Return (array, geotransform, projection) for a single-band GeoTIFF."""
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Cannot open: {path}")
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    if nodata is not None:
        arr[arr == nodata] = np.nan
    gt  = ds.GetGeoTransform()
    prj = ds.GetProjection()
    ds  = None
    return arr, gt, prj


def reproject_to_match(src_path, ref_gt, ref_prj, ref_rows, ref_cols,
                       resample_alg=gdal.GRA_Bilinear):
    """
    Reproject/resample a GeoTIFF to match a reference grid in memory.
    Nodata values in the source are replaced with NaN before reprojection
    so bilinear interpolation does not blend sentinel values with valid data.
    Returns a 2-D float32 array (NaN for nodata).
    """
    src_ds  = gdal.Open(src_path, gdal.GA_ReadOnly)
    if src_ds is None:
        raise FileNotFoundError(f"Cannot open: {src_path}")

    # Copy source to memory and replace nodata with NaN
    src_band   = src_ds.GetRasterBand(1)
    src_arr    = src_band.ReadAsArray().astype(np.float32)
    src_nodata = src_band.GetNoDataValue()
    if src_nodata is not None:
        src_arr[src_arr == src_nodata] = np.nan

    mem_drv  = gdal.GetDriverByName("MEM")
    clean_ds = mem_drv.Create("", src_ds.RasterXSize, src_ds.RasterYSize,
                              1, gdal.GDT_Float32)
    clean_ds.SetGeoTransform(src_ds.GetGeoTransform())
    clean_ds.SetProjection(src_ds.GetProjection())
    clean_ds.GetRasterBand(1).WriteArray(src_arr)
    clean_ds.GetRasterBand(1).SetNoDataValue(float('nan'))
    src_ds = None

    # Reproject clean source to target grid
    dst_ds = mem_drv.Create("", ref_cols, ref_rows, 1, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(ref_gt)
    dst_ds.SetProjection(ref_prj)
    dst_ds.GetRasterBand(1).Fill(float('nan'))
    dst_ds.GetRasterBand(1).SetNoDataValue(float('nan'))

    gdal.ReprojectImage(
        clean_ds, dst_ds,
        clean_ds.GetProjection(), ref_prj,
        resample_alg
    )

    arr = dst_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    arr[~np.isfinite(arr)] = np.nan

    clean_ds = dst_ds = None
    return arr


def build_valid_mask(mask_path, perm_mask_path):
    """
    Valid pixels: qilian_mask_new > 0 AND permafrost_qilian != 0 AND not NaN.
    Returns boolean 2-D array.
    """
    mask_arr, _, _ = read_tif(mask_path)
    perm_arr, _, _ = read_tif(perm_mask_path)
    valid = (mask_arr > 0) & (~np.isnan(mask_arr)) & (perm_arr != 0) & (~np.isnan(perm_arr))
    return valid


def get_ref_grid(mask_path):
    """Return (geotransform, projection, nrows, ncols) from the mask raster."""
    ds  = gdal.Open(mask_path, gdal.GA_ReadOnly)
    gt  = ds.GetGeoTransform()
    prj = ds.GetProjection()
    nr  = ds.RasterYSize
    nc  = ds.RasterXSize
    ds  = None
    return gt, prj, nr, nc


# ---------------------------------------------------------------------------
# 1. Load reference grid and mask
# ---------------------------------------------------------------------------
print("Loading mask and reference grid …")
ref_gt, ref_prj, ref_rows, ref_cols = get_ref_grid(MASK_PATH)
valid = build_valid_mask(MASK_PATH, PERM_MASK_PATH)
n_valid = int(valid.sum())
print(f"  Valid pixels: {n_valid}")

# ---------------------------------------------------------------------------
# 2. SHAW ALT: annual peak → mean 2000–2016
# ---------------------------------------------------------------------------
print("\nLoading SHAW ALT 2000–2016 …")
shaw_alt_stack = []
for yr in YEARS:
    path = os.path.join(SHAW_ALT_DIR, str(yr), "active_layer_depth.npy")
    arr  = np.load(path)          # shape: (rows, cols, days)
    peak = np.nanmax(arr, axis=2) # annual peak ALT per pixel
    shaw_alt_stack.append(peak)

shaw_alt_mean = np.nanmean(shaw_alt_stack, axis=0)   # shape: (rows, cols)
print(f"  SHAW ALT grid shape: {shaw_alt_mean.shape}")

# ---------------------------------------------------------------------------
# 3. SHAW permafrost presence: modal binary across 2000–2016
# ---------------------------------------------------------------------------
print("\nLoading SHAW permafrost maps 2000–2016 …")
shaw_perm_stack = []
for yr in YEARS:
    path = os.path.join(SHAW_PERM_DIR, f"permafrost_{yr}.npy")
    arr  = np.load(path).astype(np.float32)
    shaw_perm_stack.append(arr)

shaw_perm_arr   = np.stack(shaw_perm_stack, axis=0)  # (17, rows, cols)
shaw_perm_modal = stats.mode(shaw_perm_arr, axis=0, keepdims=False).mode  # (rows, cols)

# ---------------------------------------------------------------------------
# 4. FAO permafrost: modal class → binary (1=continuous, 2=discontinuous → permafrost)
# ---------------------------------------------------------------------------
print("\nLoading FAO permafrost maps 2000–2016 …")
fao_perm_stack = []
for yr in YEARS:
    path = os.path.join(FAO_PERM_DIR, str(yr), "permafrost.tif")
    arr, _, _ = read_tif(path)
    fao_perm_stack.append(arr)

fao_perm_arr        = np.stack(fao_perm_stack, axis=0)  # (17, rows, cols)
# Replace NaN with a sentinel before mode
fao_perm_arr_filled = np.where(np.isnan(fao_perm_arr), -1, fao_perm_arr)
fao_perm_modal      = stats.mode(fao_perm_arr_filled, axis=0, keepdims=False).mode
fao_perm_modal      = fao_perm_modal.astype(np.float32)
fao_perm_modal[fao_perm_modal == -1] = np.nan

# Binary: classes 1 (continuous) or 2 (discontinuous) → permafrost present
fao_perm_binary = np.where(np.isin(fao_perm_modal, [1, 2]), 1.0, 0.0)
fao_perm_binary[np.isnan(fao_perm_modal)] = np.nan

# ---------------------------------------------------------------------------
# 5. Reproject NIEER rasters to Qilian grid
# ---------------------------------------------------------------------------
print("\nReprojecting NIEER rasters to Qilian grid …")

# Diagnostic: inspect raw NIEER ALT before reprojection
_ds    = gdal.Open(NIEER_ALT_PATH, gdal.GA_ReadOnly)
_band  = _ds.GetRasterBand(1)
_raw   = _band.ReadAsArray().astype(np.float32)
_nodata = _band.GetNoDataValue()
print(f"  NIEER ALT raw - nodata value in header : {_nodata}")
print(f"  NIEER ALT raw - min : {np.nanmin(_raw):.1f}  max : {np.nanmax(_raw):.1f}  mean : {np.nanmean(_raw):.1f}")
print(f"  NIEER ALT raw - 99th percentile        : {np.nanpercentile(_raw, 99):.1f}")
print(f"  NIEER ALT raw - top-5 unique extremes  : {np.sort(np.unique(_raw))[-5:]}")
_ds = None

nieer_alt  = reproject_to_match(NIEER_ALT_PATH,  ref_gt, ref_prj, ref_rows, ref_cols) / 100.0  # cm → m
nieer_magt = reproject_to_match(NIEER_MAGT_PATH, ref_gt, ref_prj, ref_rows, ref_cols,
                                 resample_alg=gdal.GRA_NearestNeighbour)

# NIEER permafrost presence: MAGT ≤ 0 °C
nieer_perm_binary = np.where(nieer_magt <= 0, 1.0, 0.0)
nieer_perm_binary[np.isnan(nieer_magt)] = np.nan

# NIEER permafrost region: probability > 0
# Nearest-neighbour resampling preserves original zero values exactly;
# bilinear would interpolate zeros with non-zero neighbours, inflating extent.
nieer_prob = reproject_to_match(NIEER_PROB_PATH, ref_gt, ref_prj, ref_rows, ref_cols,
                                resample_alg=gdal.GRA_NearestNeighbour)
nieer_prob_binary = np.where(nieer_prob > 0, 1.0, 0.0)
nieer_prob_binary[np.isnan(nieer_prob)] = np.nan

# ---------------------------------------------------------------------------
# 6. ALT comparison (masked valid pixels)
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("ALT COMPARISON (mean 2000–2016, valid pixels only)")
print("="*60)

shaw_alt_v  = shaw_alt_mean[valid]
nieer_alt_v = nieer_alt[valid]

# Pixels valid in both
both_alt = ~np.isnan(shaw_alt_v) & ~np.isnan(nieer_alt_v)
sa = shaw_alt_v[both_alt]
na = nieer_alt_v[both_alt]

print(f"  Pixels with valid ALT in both datasets: {both_alt.sum()}")
print(f"  SHAW  mean peak ALT : {np.nanmean(sa):.3f} m  (std: {np.nanstd(sa):.3f} m)")
print(f"  NIEER mean ALT      : {np.nanmean(na):.3f} m  (std: {np.nanstd(na):.3f} m)")
print(f"  Mean bias (SHAW−NIEER): {np.nanmean(sa - na):.3f} m")
print(f"  RMSE                : {np.sqrt(np.nanmean((sa - na)**2)):.3f} m")
r, p = stats.spearmanr(sa, na)
print(f"  Spearman r          : {r:.3f}  (p = {p:.3e})")

# ---------------------------------------------------------------------------
# 7. Permafrost presence comparison (masked valid pixels)
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("PERMAFROST PRESENCE COMPARISON (modal 2000–2016, valid pixels only)")
print("="*60)

def confusion_stats(ref, pred, label):
    """Print TP/FP/FN/TN, precision, recall, F1 for binary arrays."""
    both = ~np.isnan(ref) & ~np.isnan(pred)
    r = ref[both].astype(bool)
    p = pred[both].astype(bool)
    tp = int(( r &  p).sum())
    fp = int((~r &  p).sum())
    fn = int(( r & ~p).sum())
    tn = int((~r & ~p).sum())
    total = tp + fp + fn + tn
    acc   = (tp + tn) / total if total > 0 else np.nan
    prec  = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    rec   = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1    = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else np.nan
    print(f"\n  {label}")
    print(f"    Pixels compared : {total}")
    print(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"    Accuracy  : {acc:.3f}")
    print(f"    Precision : {prec:.3f}")
    print(f"    Recall    : {rec:.3f}")
    print(f"    F1        : {f1:.3f}")
    # Agreement breakdown
    perm_ref  = int(r.sum())
    perm_pred = int(p.sum())
    print(f"    NIEER permafrost pixels : {perm_ref}  ({100*perm_ref/total:.1f}%)")
    print(f"    {label.split('vs')[0].strip()} permafrost pixels : {perm_pred}  ({100*perm_pred/total:.1f}%)")

shaw_perm_v  = shaw_perm_modal[valid].astype(np.float32)
fao_perm_v   = fao_perm_binary[valid]
nieer_perm_v = nieer_perm_binary[valid]

confusion_stats(nieer_perm_v, shaw_perm_v,  "SHAW vs NIEER (MAGT ≤ 0°C)")
confusion_stats(nieer_perm_v, fao_perm_v,   "FAO  vs NIEER (MAGT ≤ 0°C)")

# SHAW vs FAO direct comparison
print("\n  --- SHAW vs FAO (internal consistency) ---")
both_sf = ~np.isnan(shaw_perm_v) & ~np.isnan(fao_perm_v)
agree   = (shaw_perm_v[both_sf] == fao_perm_v[both_sf]).sum()
total_sf = both_sf.sum()
print(f"    Pixels compared : {total_sf}")
print(f"    Agreement       : {agree}  ({100*agree/total_sf:.1f}%)")

# ---------------------------------------------------------------------------
# 7b. Permafrost presence: probability > 0 comparisons
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("PERMAFROST PRESENCE — NIEER PROBABILITY > 0 (permafrost region)")
print("="*60)

nieer_prob_v = nieer_prob_binary[valid]

confusion_stats(nieer_prob_v, shaw_perm_v,  "SHAW vs NIEER (probability > 0)")
confusion_stats(nieer_prob_v, fao_perm_v,   "FAO  vs NIEER (probability > 0)")

# MAGT vs probability: how different are the two NIEER definitions?
print("\n  --- NIEER MAGT ≤ 0°C vs NIEER probability > 0 (internal) ---")
both_nieer = ~np.isnan(nieer_perm_v) & ~np.isnan(nieer_prob_v)
agree_nieer = (nieer_perm_v[both_nieer] == nieer_prob_v[both_nieer]).sum()
total_nieer = both_nieer.sum()
magt_perm   = int(nieer_perm_v[both_nieer].sum())
prob_perm   = int(nieer_prob_v[both_nieer].sum())
print(f"    Pixels compared                  : {total_nieer}")
print(f"    Agreement                        : {agree_nieer}  ({100*agree_nieer/total_nieer:.1f}%)")
print(f"    NIEER MAGT permafrost pixels     : {magt_perm}  ({100*magt_perm/total_nieer:.1f}%)")
print(f"    NIEER probability permafrost pixels: {prob_perm}  ({100*prob_perm/total_nieer:.1f}%)")

# ---------------------------------------------------------------------------
# 8. Diagnostic: does NIEER ALT valid mask match NIEER MAGT permafrost mask?
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("DIAGNOSTIC: NIEER ALT valid pixels vs NIEER MAGT permafrost pixels")
print("="*60)

nieer_alt_valid  = np.isfinite(nieer_alt)          # pixels with valid ALT after reprojection
nieer_perm_mask  = (nieer_magt <= 0) & np.isfinite(nieer_magt)  # MAGT permafrost pixels

both_defined = nieer_alt_valid & nieer_perm_mask
alt_only     = nieer_alt_valid & ~nieer_perm_mask   # has ALT but MAGT > 0
perm_only    = ~nieer_alt_valid & nieer_perm_mask   # MAGT permafrost but no ALT

print(f"  Pixels with valid NIEER ALT                  : {nieer_alt_valid.sum()}")
print(f"  Pixels with NIEER MAGT ≤ 0°C                 : {nieer_perm_mask.sum()}")
print(f"  Pixels with both valid ALT and MAGT ≤ 0°C    : {both_defined.sum()}")
print(f"  Pixels with ALT but MAGT > 0 (unexpected)    : {alt_only.sum()}")
print(f"  Pixels with MAGT ≤ 0°C but no ALT (expected) : {perm_only.sum()}")

print("\nDone.")