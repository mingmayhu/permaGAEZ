"""
Permafrost Thaw Impact Analysis — Aggregate Hotspots with Gains/Losses
======================================================================
Generates overall ΔYield and ΔSuitability hotspots across all crops:

1. Aggregate ΔYield magnitude (|ΔYield|)
2. ΔYield losses only (negative)
3. ΔYield gains only (positive)
4. Aggregate ΔSuitability (mean across crops)
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from osgeo import gdal
except ImportError:
    import gdal

# ------------------------
# CONFIGURATION
# ------------------------
WORK_DIR   = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH  = os.path.join(WORK_DIR, 'data_input/qilian mask.tif')
ELEV_PATH  = os.path.join(WORK_DIR, 'data_input/terrain/elevation.npy')
DATA_RAW   = os.path.join(WORK_DIR, 'data_output/final_classification')
DATA_RAW_NT= os.path.join(WORK_DIR, 'data_output/final_classification_nothaw')
OUT_ROOT   = os.path.join(WORK_DIR, 'thaw_analysis_output')

YEARS_COMPARISON = list(range(1999, 2019))

CROPS = [
    {'label': 'Winter Barley', 'tag': 'combined_winter_barley'},
    {'label': 'Spring Barley', 'tag': 'combined_spring_barley'},
    {'label': 'Winter Wheat',  'tag': 'combined_winter_wheat'},
    {'label': 'Spring Wheat',  'tag': 'combined_spring_wheat'},
    {'label': 'Silage Maize',  'tag': 'combined_silage_maize'},
    {'label': 'White Potato',  'tag': 'combined_white_potato'},
    {'label': 'Oat',           'tag': 'combined_oat'},
    {'label': 'Dry Pea',       'tag': 'combined_dry_pea'},
    {'label': 'Winter Rape',   'tag': 'combined_winter_rape'},
    {'label': 'Spring Rape',   'tag': 'combined_spring_rape'},
]

os.makedirs(OUT_ROOT, exist_ok=True)
os.chdir(WORK_DIR)

# ------------------------
# HELPERS
# ------------------------
def load_raster(path):
    ds = gdal.Open(path)
    if ds is None:
        print(f"⚠ Could not open: {path}")
        return None
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[arr < -1e10] = np.nan
    return arr

def load_mask():
    arr = load_raster(MASK_PATH)
    if arr is None:
        raise RuntimeError("Mask could not be loaded.")
    mask = arr.astype(bool)
    print(f"Mask loaded: shape={mask.shape}, valid pixels={mask.sum()}")
    return mask

def load_combined_raw(tag, year, nothaw=False):
    path = os.path.join(DATA_RAW_NT if nothaw else DATA_RAW, tag, f"{year}_raw_yield.tif")
    arr = load_raster(path)
    if arr is not None:
        arr[arr < 0] = np.nan
    return arr

def load_combined_class(tag, year, nothaw=False):
    path = os.path.join(DATA_RAW_NT if nothaw else DATA_RAW, tag, f"{year}_final_yield_class.tif")
    arr = load_raster(path)
    return arr

def compute_delta_stack(values):
    obs_stack, cf_stack = values
    if not obs_stack or not cf_stack:
        return None
    obs_mean = np.nanmean(obs_stack, axis=0)
    cf_mean  = np.nanmean(cf_stack, axis=0)
    return obs_mean - cf_mean

def plot_hotspot(delta, mask, out_path, metric='ΔYield', cmap='hot', vmin=None, vmax=None):
    delta_plot = delta.copy()
    delta_plot[~mask] = np.nan
    if vmin is None:
        vmin = np.nanmin(delta_plot)
    if vmax is None:
        vmax = np.nanpercentile(delta_plot, 98)
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(delta_plot, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label=metric)
    plt.title(f"Permafrost Thaw Impact — {metric}")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ {metric} hotspot saved: {out_path}")

def plot_gains_losses_overlay(delta_pos, delta_neg, mask, out_path, vmax_percentile=98):
    """
    Plot combined ΔYield gains and losses overlay.
    delta_pos: 2D array of positive ΔYield (gains)
    delta_neg: 2D array of negative ΔYield (losses, negative values)
    mask: boolean array of valid pixels
    out_path: path to save figure
    vmax_percentile: percentile for scaling max color intensity
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize

    # Mask invalid pixels
    delta_pos_plot = delta_pos.copy()
    delta_neg_plot = delta_neg.copy()
    delta_pos_plot[~mask] = np.nan
    delta_neg_plot[~mask] = np.nan

    # Determine vmax for scaling
    vmax_pos = np.nanpercentile(delta_pos_plot, vmax_percentile)
    vmax_neg = np.nanpercentile(-delta_neg_plot, vmax_percentile)  # take abs for scaling

    # Normalize
    norm_pos = Normalize(vmin=0, vmax=vmax_pos)
    norm_neg = Normalize(vmin=0, vmax=vmax_neg)

    # Create RGB overlay
    rgb = np.zeros((delta_pos_plot.shape[0], delta_pos_plot.shape[1], 3), dtype=np.float32)

    # Scale intensities to [0,1] using norms
    rgb[..., 0] = norm_neg(-delta_neg_plot)  # Red channel for losses
    rgb[..., 1] = norm_pos(delta_pos_plot)   # Green channel for gains
    rgb[..., 2] = 0                          # No blue

    # Set NaNs to white
    rgb[np.isnan(rgb[...,0]) & np.isnan(rgb[...,1]), :] = 1.0

    # Plot
    fig, ax = plt.subplots(figsize=(8,6))
    ax.imshow(rgb)
    ax.axis('off')
    plt.title("Permafrost Thaw Impact — ΔYield Gains vs Losses")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Gains vs Losses overlay saved: {out_path}")

# ------------------------
# MAIN SCRIPT
# ------------------------
if __name__ == '__main__':
    mask = load_mask()
    elevation = np.load(ELEV_PATH)

    delta_yield_list = []
    delta_class_list = []

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        obs_stack, cf_stack = [], []
        obs_class_stack, cf_class_stack = [], []

        for year in YEARS_COMPARISON:
            obs = load_combined_raw(tag, year, nothaw=False)
            cf  = load_combined_raw(tag, year, nothaw=True)
            if obs is not None and cf is not None:
                obs[~mask] = np.nan
                cf[~mask] = np.nan
                obs_stack.append(obs)
                cf_stack.append(cf)
            obs_cls = load_combined_class(tag, year, nothaw=False)
            cf_cls  = load_combined_class(tag, year, nothaw=True)
            if obs_cls is not None and cf_cls is not None:
                obs_cls[~mask] = np.nan
                cf_cls[~mask] = np.nan
                obs_class_stack.append(obs_cls)
                cf_class_stack.append(cf_cls)

        delta_yield_crop = compute_delta_stack((obs_stack, cf_stack))
        delta_class_crop = compute_delta_stack((obs_class_stack, cf_class_stack))
        if delta_yield_crop is not None:
            delta_yield_list.append(delta_yield_crop)
        if delta_class_crop is not None:
            delta_class_list.append(delta_class_crop)

    # ------------------------
    # Aggregate ΔYield
    # ------------------------
    if delta_yield_list:
        # 1. Absolute magnitude
        delta_yield_all_mag = np.nansum(np.stack([np.abs(d) for d in delta_yield_list]), axis=0)
        delta_yield_all_mag[~mask] = np.nan
        plot_hotspot(delta_yield_all_mag, mask,
                     os.path.join(OUT_ROOT, 'all_crops_delta_yield_magnitude.png'),
                     metric='Aggregate |ΔYield|', cmap='hot')
        np.savetxt(os.path.join(OUT_ROOT, 'all_crops_delta_yield_magnitude.csv'),
                   np.where(mask, delta_yield_all_mag, np.nan), delimiter=',')

        # 2. Losses only (negative)
        delta_yield_all_neg = np.zeros_like(delta_yield_list[0])
        for d in delta_yield_list:
            delta_yield_all_neg += np.where(d < 0, d, 0)
        delta_yield_all_neg[~mask] = np.nan
        plot_hotspot(delta_yield_all_neg, mask,
                     os.path.join(OUT_ROOT, 'all_crops_delta_yield_losses.png'),
                     metric='ΔYield Losses Only', cmap='Reds')
        np.savetxt(os.path.join(OUT_ROOT, 'all_crops_delta_yield_losses.csv'),
                   np.where(mask, delta_yield_all_neg, np.nan), delimiter=',')

        # 3. Gains only (positive)
        delta_yield_all_pos = np.zeros_like(delta_yield_list[0])
        for d in delta_yield_list:
            delta_yield_all_pos += np.where(d > 0, d, 0)
        delta_yield_all_pos[~mask] = np.nan
        plot_hotspot(delta_yield_all_pos, mask,
                     os.path.join(OUT_ROOT, 'all_crops_delta_yield_gains.png'),
                     metric='ΔYield Gains Only', cmap='Greens')
        np.savetxt(os.path.join(OUT_ROOT, 'all_crops_delta_yield_gains.csv'),
                   np.where(mask, delta_yield_all_pos, np.nan), delimiter=',')
        plot_gains_losses_overlay(delta_yield_all_pos, delta_yield_all_neg, mask,
                          os.path.join(OUT_ROOT, 'all_crops_delta_yield_overlay.png'))
    else:
        print("No ΔYield data available.")

    # ------------------------
    # Aggregate ΔSuitability
    # ------------------------
    if delta_class_list:
        delta_class_all = np.nanmean(np.stack(delta_class_list), axis=0)
        delta_class_all[~mask] = np.nan
        plot_hotspot(delta_class_all, mask,
                     os.path.join(OUT_ROOT, 'all_crops_delta_class.png'),
                     metric='Aggregate ΔSuitability', cmap='RdBu')
        np.savetxt(os.path.join(OUT_ROOT, 'all_crops_delta_class.csv'),
                   np.where(mask, delta_class_all, np.nan), delimiter=',')
    else:
        print("No ΔSuitability data available.")

    print(f"\nAll analyses complete. Outputs in: {OUT_ROOT}/")