from osgeo import gdal
import numpy as np

mask_arr = gdal.Open('./data_input/qilian mask.tif').ReadAsArray().astype(bool)
ds = gdal.Open('./data_output/module5/spring_barley_63/1999/yield_terrain.tif')
raw = ds.ReadAsArray().astype(float)

neg999_mask = raw == -999
print("total -999 pixels:", np.sum(neg999_mask))
print("outside mask -999:", np.sum(neg999_mask & ~mask_arr))
print("INSIDE mask -999:", np.sum(neg999_mask & mask_arr))
