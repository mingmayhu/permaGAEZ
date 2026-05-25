"""
Convert CROPGRIDS NetCDF to GeoTIFF aligned to PyAEZ reference grid
====================================================================
Exports three variables per crop:
  - harvested_area (ha) — {crop}_harvested_area.tif
  - crop area (ha)      — {crop}_crop_area.tif
  - quality score       — {crop}_quality.tif

Quality values (0–1): 1.0=best data, 0.0=missing data
"""

import os
import numpy as np
try:
    import netCDF4 as nc
except ImportError:
    raise ImportError("Install netCDF4: pip install netCDF4")
from osgeo import gdal, osr

# ── Config ────────────────────────────────────────────────────────────────────

BASE = "/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez"

# Reference raster — defines the target grid (extent, resolution, CRS)
REFERENCE_TIF = os.path.join(
    BASE, "data_output/final_classification/combined_barley/2001_raw_yield.tif")

# CROPGRIDS NetCDF files and output names
CROPS = [
    ("barley",   "CROPGRIDSv1.08_barley.nc"),
    ("wheat",    "CROPGRIDSv1.08_wheat.nc"),
    ("rapeseed", "CROPGRIDSv1.08_rapeseed.nc"),
    ("spring oat", "CROPGRIDSv1.08_oats.nc"),
    ("dry pea", "CROPGRIDSv1.08_pea.nc"),
    ("silage maize", "CROPGRIDSv1.08_maizefor.nc"),
    ("white potato", "CROPGRIDSv1.08_potato.nc")
]

NC_DIR  = os.path.join(BASE, "data_input/cropgrids")
OUT_DIR = os.path.join(BASE, "data_input/cropgrids")

# Variable names inside CROPGRIDS NetCDF files
HA_VAR   = "harvarea"   # harvested area
CA_VAR   = "croparea"   # crop (physical) area
QUAL_VAR = "qual"       # data quality score


# ── Helpers ───────────────────────────────────────────────────────────────────

def nc_to_tif(nc_path, out_path, var_name):
    """
    Extract a variable from a NetCDF file and save as a GeoTIFF.
    Assumes the variable has dimensions (lat, lon) or (longitude, latitude).
    Returns the output path.
    """
    ds = nc.Dataset(nc_path, "r")

    # Find lat/lon dimension names
    dims = list(ds.variables.keys())
    print(f"  Variables in {os.path.basename(nc_path)}: {dims}")

    # Try common lat/lon names
    lat_name = next((v for v in ["lat", "latitude", "y"] if v in dims), None)
    lon_name = next((v for v in ["lon", "longitude", "x"] if v in dims), None)

    if lat_name is None or lon_name is None:
        raise ValueError(f"Cannot find lat/lon variables in {nc_path}. "
                         f"Available: {dims}")

    if var_name not in ds.variables:
        # Try to find a suitable variable automatically
        candidates = [v for v in dims
                      if v not in [lat_name, lon_name]
                      and len(ds.variables[v].dimensions) == 2]
        if candidates:
            var_name = candidates[0]
            print(f"  Variable '{HA_VAR}' not found — using '{var_name}' instead")
        else:
            raise ValueError(f"Variable '{var_name}' not found in {nc_path}. "
                             f"Available: {dims}")

    lats = ds.variables[lat_name][:]
    lons = ds.variables[lon_name][:]
    data = ds.variables[var_name][:]

    # Handle masked arrays
    if hasattr(data, "filled"):
        fill_val = ds.variables[var_name]._FillValue if hasattr(
            ds.variables[var_name], "_FillValue") else -9999.0
        data = data.filled(fill_val)
    data = np.array(data, dtype=np.float32)
    # Treat -1 and other negative values as nodata
    data[data < 0] = -9999.0

    # Ensure lat is descending (north-up)
    if lats[0] < lats[-1]:
        lats = lats[::-1]
        data = data[::-1, :]

    # Build geotransform: top-left corner + pixel size
    lat_res = abs(float(lats[1] - lats[0]))
    lon_res = abs(float(lons[1] - lons[0]))
    x_min   = float(lons[0])  - lon_res / 2
    y_max   = float(lats[0])  + lat_res / 2

    rows, cols = data.shape
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(out_path, cols, rows, 1, gdal.GDT_Float32,
                           options=["COMPRESS=LZW"])
    out_ds.SetGeoTransform([x_min, lon_res, 0, y_max, 0, -lat_res])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    out_ds.SetProjection(srs.ExportToWkt())
    band = out_ds.GetRasterBand(1)
    band.SetNoDataValue(-9999.0)
    band.WriteArray(data)
    out_ds.FlushCache()
    out_ds = None
    ds.close()
    print(f"  Written: {out_path}  shape=({rows}×{cols}), "
          f"res={lon_res:.4f}°, range=[{np.nanmin(data[data>-9999]):.2f}, "
          f"{np.nanmax(data[data>-9999]):.2f}]")
    return out_path


def warp_to_reference(src_path, ref_path, out_path):
    """
    Warp src_path to exactly match the extent, resolution, and pixel
    alignment of ref_path. Uses average resampling (suitable for area data).
    """
    ref_ds  = gdal.Open(ref_path)
    ref_geo = ref_ds.GetGeoTransform()
    ref_proj = ref_ds.GetProjection()
    rows    = ref_ds.RasterYSize
    cols    = ref_ds.RasterXSize
    x_min   = ref_geo[0]
    y_max   = ref_geo[3]
    x_res   = ref_geo[1]
    y_res   = abs(ref_geo[5])
    x_max   = x_min + cols * x_res
    y_min   = y_max - rows * y_res
    ref_ds  = None

    warp_opts = gdal.WarpOptions(
        format="GTiff",
        outputBounds=(x_min, y_min, x_max, y_max),
        xRes=x_res,
        yRes=y_res,
        dstSRS=ref_proj,
        resampleAlg=gdal.GRA_Average,  # sum-conserving for area data
        creationOptions=["COMPRESS=LZW"],
        dstNodata=-9999.0,
    )
    gdal.Warp(out_path, src_path, options=warp_opts)

    # Report result
    out_ds  = gdal.Open(out_path)
    band    = out_ds.GetRasterBand(1)
    arr     = band.ReadAsArray().astype(float)
    arr[arr == -9999] = np.nan
    arr[arr < 0]      = np.nan  # catch -1 nodata values
    valid   = arr[np.isfinite(arr) & (arr > 0)]
    if len(valid) == 0:
        print(f"  Warped → {os.path.basename(out_path)}: "
              f"shape=({out_ds.RasterYSize}×{out_ds.RasterXSize}), "
              f"WARNING: no valid pixels in study area")
    else:
        print(f"  Warped → {os.path.basename(out_path)}: "
              f"shape=({out_ds.RasterYSize}×{out_ds.RasterXSize}), "
              f"valid pixels={len(valid)}, "
              f"mean={np.mean(valid):.2f} ha, max={np.max(valid):.2f} ha")
    out_ds = None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for crop, nc_file in CROPS:
        nc_path = os.path.join(NC_DIR, nc_file)
        if not os.path.exists(nc_path):
            print(f"\n  SKIP {crop}: file not found at {nc_path}")
            continue

        print(f"\n{'─'*55}")
        print(f"  Processing: {crop}")
        print(f"{'─'*55}")

        # Export harvested area
        global_tif = os.path.join(OUT_DIR, f"{crop}_ha_global.tif")
        nc_to_tif(nc_path, global_tif, HA_VAR)
        aligned_tif = os.path.join(OUT_DIR, f"{crop}_harvested_area.tif")
        warp_to_reference(global_tif, REFERENCE_TIF, aligned_tif)
        os.remove(global_tif)
        print(f"  → {aligned_tif}")

        # Export crop area
        global_tif = os.path.join(OUT_DIR, f"{crop}_ca_global.tif")
        nc_to_tif(nc_path, global_tif, CA_VAR)
        aligned_tif = os.path.join(OUT_DIR, f"{crop}_crop_area.tif")
        warp_to_reference(global_tif, REFERENCE_TIF, aligned_tif)
        os.remove(global_tif)
        print(f"  → {aligned_tif}")

        # Export quality
        global_tif = os.path.join(OUT_DIR, f"{crop}_qual_global.tif")
        nc_to_tif(nc_path, global_tif, QUAL_VAR)
        aligned_tif = os.path.join(OUT_DIR, f"{crop}_quality.tif")
        warp_to_reference(global_tif, REFERENCE_TIF, aligned_tif)
        os.remove(global_tif)
        print(f"  → {aligned_tif}")

    print(f"\n{'='*55}")
    print("  All crops processed.")
    print(f"  Output: {OUT_DIR}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()