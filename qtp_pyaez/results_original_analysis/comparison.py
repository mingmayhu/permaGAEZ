import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from osgeo import gdal
import os

permafrost_dir = "/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez/data_output/final_classification/winter_barley_59"
original_dir = "/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez/data_output/original/final_classification/winter_barley_59"

# years = range(1979, 1995)

# fig, axes = plt.subplots(len(years), 3, figsize=(15, 4 * len(years)))
# fig.suptitle("Winter Barley (1979–1990): Original vs Permafrost vs Difference", fontsize=14)

# # shared colormap for yield class (adjust vmin/vmax if needed)
# yield_cmap = "RdYlGn"
# diff_cmap = "coolwarm"

# for i, year in enumerate(years):
#     pf_path = os.path.join(permafrost_dir, f"{year}_final_yield_class.tif")
#     orig_path = os.path.join(original_dir, f"{year}_final_yield_class.tif")

#     pf_ds = gdal.Open(pf_path)
#     orig_ds = gdal.Open(orig_path)

#     pf_arr = pf_ds.GetRasterBand(1).ReadAsArray().astype(float)
#     orig_arr = orig_ds.GetRasterBand(1).ReadAsArray().astype(float)

#     # mask nodata (commonly 0 or -9999)
#     nodata = pf_ds.GetRasterBand(1).GetNoDataValue()
#     if nodata is not None:
#         pf_arr[pf_arr == nodata] = np.nan
#         orig_arr[orig_arr == nodata] = np.nan

#     diff = pf_arr - orig_arr

#     vmin = np.nanmin([pf_arr, orig_arr])
#     vmax = np.nanmax([pf_arr, orig_arr])
#     diff_abs = np.nanmax(np.abs(diff))

#     ax_orig = axes[i, 0]
#     ax_pf   = axes[i, 1]
#     ax_diff = axes[i, 2]

#     ax_orig.imshow(orig_arr, cmap=yield_cmap, vmin=vmin, vmax=vmax)
#     ax_orig.set_title(f"{year} — Original")
#     ax_orig.axis("off")

#     ax_pf.imshow(pf_arr, cmap=yield_cmap, vmin=vmin, vmax=vmax)
#     ax_pf.set_title(f"{year} — Permafrost")
#     ax_pf.axis("off")

#     im = ax_diff.imshow(diff, cmap=diff_cmap, vmin=-diff_abs, vmax=diff_abs)
#     ax_diff.set_title(f"{year} — Diff (PF − Orig)")
#     ax_diff.axis("off")
#     plt.colorbar(im, ax=ax_diff, fraction=0.046, pad=0.04)

#     # print summary stats
#     n_changed = np.sum(~np.isnan(diff) & (diff != 0))
#     print(f"{year}: pixels changed={n_changed}, mean diff={np.nanmean(diff):.4f}, max diff={np.nanmax(diff):.4f}, min diff={np.nanmin(diff):.4f}")

# plt.tight_layout()
# plt.savefig("comparison_winter_barley_1979_1990.png", dpi=150, bbox_inches="tight")
# plt.show()
# print("Saved to comparison_winter_barley_1979_1990.png")

# Run for one year to inspect where and why differences occur
year = 1987  # most changed pixels in your early period

pf_ds = gdal.Open(f"{permafrost_dir}/{year}_final_yield_class.tif")
orig_ds = gdal.Open(f"{original_dir}/{year}_final_yield_class.tif")

pf_arr = pf_ds.GetRasterBand(1).ReadAsArray().astype(float)
orig_arr = orig_ds.GetRasterBand(1).ReadAsArray().astype(float)

nodata = pf_ds.GetRasterBand(1).GetNoDataValue()
if nodata:
    pf_arr[pf_arr == nodata] = np.nan
    orig_arr[orig_arr == nodata] = np.nan

diff = pf_arr - orig_arr

# Find changed pixels
rows, cols = np.where((diff != 0) & ~np.isnan(diff))

print(f"Changed pixels in {year}: {len(rows)}")
print(f"{'Row':>6} {'Col':>6} {'Original':>10} {'Permafrost':>12} {'Diff':>6}")
print("-" * 45)
for r, c in zip(rows, cols):
    print(f"{r:>6} {c:>6} {orig_arr[r,c]:>10.0f} {pf_arr[r,c]:>12.0f} {diff[r,c]:>6.0f}")