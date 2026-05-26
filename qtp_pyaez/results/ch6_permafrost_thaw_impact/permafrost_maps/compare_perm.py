"""
compare_permafrost_maps.py

Compares FAO and P-GAEZ permafrost classifications for 2010
against the observed permafrost_qilian.tif reference.

Metrics per model:
    - Pixel-level agreement (% correctly classified)
    - Precision, Recall, F1 (permafrost = positive class)
    - Confusion matrix counts (TP, FP, FN, TN)

Outputs:
    results/permafrost_maps/permafrost_comparison_whole_period.csv
    results/permafrost_maps/permafrost_comparison_whole_period.txt
"""

import os
import numpy as np
import pandas as pd
from osgeo import gdal

WORK_DIR = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
os.chdir(WORK_DIR)

OUT_DIR  = r'./results/permafrost_maps'
os.makedirs(OUT_DIR, exist_ok=True)

MASK_PATH  = r'./data_input/qilian_mask_new.tif'
YEARS_FULL = list(range(1979, 2019))

# === Load valid mask ===
ds     = gdal.Open(MASK_PATH)
mask   = ds.GetRasterBand(1).ReadAsArray().astype(bool)
ds     = None

# === Load observed permafrost (reference) ===
ds_obs  = gdal.Open('./data_input/permafrost_qilian.tif')
obs     = ds_obs.GetRasterBand(1).ReadAsArray().astype(float)
nd_obs  = ds_obs.GetRasterBand(1).GetNoDataValue()
ds_obs  = None
if nd_obs is not None:
    obs[obs == nd_obs] = np.nan
obs_bin = np.where(~np.isnan(obs), (obs == 1).astype(float), np.nan)

# === FAO majority vote — whole period ===
print("Loading FAO whole period...")
fao_stack = []
for y in YEARS_FULL:
    ds_fao = gdal.Open(f'./data_output/original/module1/{y}/permafrost.tif')
    if ds_fao is None:
        continue
    arr    = ds_fao.GetRasterBand(1).ReadAsArray().astype(float)
    nd     = ds_fao.GetRasterBand(1).GetNoDataValue()
    ds_fao = None
    if nd is not None:
        arr[arr == nd] = np.nan
    fao_stack.append((arr <= 2).astype(float))
fao_bin = (np.nanmean(np.array(fao_stack), axis=0) >= 0.5).astype(float)

# === P-GAEZ majority vote — whole period ===
print("Loading P-GAEZ whole period...")
pgaez_stack = []
for y in YEARS_FULL:
    p = f'./data_output/module1/permafrost_maps/permafrost_{y}.npy'
    if not os.path.exists(p):
        continue
    pgaez_stack.append(np.load(p).astype(float))
pgaez_bin = (np.mean(np.array(pgaez_stack), axis=0) >= 0.5).astype(float)

# === Valid pixels: mask + no nans in any map ===
valid = mask & ~np.isnan(obs_bin) & ~np.isnan(fao_bin) & ~np.isnan(pgaez_bin)
print(f"Valid pixels: {valid.sum()}")

obs_v   = obs_bin[valid].astype(bool)
fao_v   = fao_bin[valid].astype(bool)
pgaez_v = pgaez_bin[valid].astype(bool)

# === Metrics function ===
def classification_metrics(pred, obs, name):
    TP = int(( pred &  obs).sum())
    FP = int(( pred & ~obs).sum())
    FN = int((~pred &  obs).sum())
    TN = int((~pred & ~obs).sum())
    n  = len(obs)

    accuracy  = (TP + TN) / n
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'model':     name,
        'n_valid':   n,
        'TP':        TP,
        'FP':        FP,
        'FN':        FN,
        'TN':        TN,
        'accuracy':  round(accuracy,  4),
        'precision': round(precision, 4),
        'recall':    round(recall,    4),
        'f1':        round(f1,        4),
    }

# === Run comparison ===
results = [
    classification_metrics(fao_v,   obs_v, 'FAO PyAEZ'),
    classification_metrics(pgaez_v, obs_v, 'P-GAEZ'),
]

df = pd.DataFrame(results)
print("\nPermafrost Classification Comparison — Whole Period (1979-2018)")
print(df.to_string(index=False))

# === Save outputs ===
csv_path = os.path.join(OUT_DIR, 'permafrost_comparison_whole_period.csv')
txt_path = os.path.join(OUT_DIR, 'permafrost_comparison_whole_period.txt')

df.to_csv(csv_path, index=False)

with open(txt_path, 'w') as f:
    f.write("Permafrost Classification Comparison — Whole Period (1979-2018)\n")
    f.write("=" * 60 + "\n")
    f.write(f"Reference: permafrost_qilian.tif (1 = permafrost)\n")
    f.write(f"Valid pixels: {valid.sum()}\n\n")
    f.write(df.to_string(index=False))
    f.write("\n\n")
    f.write("Observed permafrost pixels:   " + str(int(obs_v.sum())) + "\n")
    f.write("FAO permafrost pixels:        " + str(int(fao_v.sum())) + "\n")
    f.write("P-GAEZ permafrost pixels:     " + str(int(pgaez_v.sum())) + "\n")

print(f"\nSaved: {csv_path}")
print(f"Saved: {txt_path}")
print("Done.")