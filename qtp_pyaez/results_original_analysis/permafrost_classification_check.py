import numpy as np
from osgeo import gdal

# Load permafrost class arrays
pf_class_arr = np.load("/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez/data_output/module1/permafrost_maps/permafrost_1987.npy")

orig_pf_ds = gdal.Open("/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez/data_output/original/module1/1987/permafrost.tif")
orig_class_arr = orig_pf_ds.GetRasterBand(1).ReadAsArray()

# Load final yield arrays
pf_ds = gdal.Open("/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez/data_output/final_classification/winter_barley_59/1987_final_yield_class.tif")
orig_ds = gdal.Open("/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez/data_output/original/final_classification/winter_barley_59/1987_final_yield_class.tif")
pf_yield = pf_ds.GetRasterBand(1).ReadAsArray().astype(float)
orig_yield = orig_ds.GetRasterBand(1).ReadAsArray().astype(float)

# The pixels that drop to zero
zero_rows = [25,25,25,25,25,26,26,26,26,26,26,26,26,26,26,
             27,27,27,27,27,27,27,27,27,28,28,28,28,28,28,
             28,28,29,29,29,29,29,29,29,29,29,29,30,30,30,
             30,30,30,30,30,30,31,31,31,31,31,31,31,31,32,32]
zero_cols = [61,62,63,64,66,58,59,60,61,62,63,64,65,66,67,
             60,62,63,64,65,66,67,68,69,61,62,63,64,65,66,
             67,68,62,63,64,65,66,67,68,69,70,63,64,65,66,
             67,68,69,70,64,65,66,67,68,69,70,71,70,71]

print(f"{'Row':>5} {'Col':>5} {'SHAW_class':>12} {'AirFrost_class':>16} {'Orig_yield':>12} {'PF_yield':>10} {'Driver':>15}")
print("-" * 80)

for r, c in zip(zero_rows, zero_cols):
    shaw = pf_class_arr[r, c]
    af = orig_class_arr[r, c]
    oy = orig_yield[r, c]
    py = pf_yield[r, c]
    
    # Determine what's driving the zero
    if shaw in [1, 2] and af not in [1, 2]:
        driver = "SHAW_class"       # SHAW flags it, air frost doesn't -> permafrost class is the cause
    elif shaw not in [1, 2] and af not in [1, 2]:
        driver = "soil_moisture"    # neither flags it -> must be soil moisture/AWC difference
    elif shaw in [1, 2] and af in [1, 2]:
        driver = "both_flagged"     # both flag it, but orig still has yield -> something else
    else:
        driver = "check_manually"
    
    print(f"{r:>5} {c:>5} {shaw:>12.0f} {af:>16.0f} {oy:>12.0f} {py:>10.0f} {driver:>15}")

# Summary
print("\nSummary of drivers:")
drivers = []
for r, c in zip(zero_rows, zero_cols):
    shaw = pf_class_arr[r, c]
    af = orig_class_arr[r, c]
    if shaw in [1, 2] and af not in [1, 2]:
        drivers.append("SHAW_class")
    elif shaw not in [1, 2] and af not in [1, 2]:
        drivers.append("soil_moisture")
    elif shaw in [1, 2] and af in [1, 2]:
        drivers.append("both_flagged")
    else:
        drivers.append("check_manually")

from collections import Counter
for driver, count in Counter(drivers).items():
    print(f"  {driver}: {count} pixels ({count/len(drivers)*100:.1f}%)")