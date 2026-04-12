"""
Step 4 — Elevation & Slope Stratification of ΔSuitability
==========================================================
Updated to:
  - Exclude lake pixels (nodata or 0 in permafrost_qilian.tif)
  - Load mean delta rasters from new spatial analysis output path

Outputs written to:
  ./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/7_elevation_slope/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from osgeo import gdal

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR        = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH       = r'./data_input/qilian mask.tif'
PERMAFROST_PATH = r'./data_input/permafrost_qilian.tif'
ELEV_PATH       = r'./data_input/terrain/elevation.npy'
SLOPE_PATH      = r'./data_input/terrain/slope.tif'
DELTA_DIR       = './results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/5_spatial/1_mean_delta'
OUT_ROOT        = './results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/7_elevation_slope'

ELEV_BINS  = list(range(2000, 6000, 500))

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

import matplotlib.colors as mc

def lighten_color(color, amount=0.5):
    """Lighten a color by mixing with white. amount=0 = original, amount=1 = white."""
    c = mc.to_rgb(color)
    return tuple(c[i] + (1 - c[i]) * amount for i in range(3))

_tab10 = plt.get_cmap('tab10', len(CROPS))
def CROP_COLORS(i):
    return lighten_color(_tab10(i), amount=0.45)

os.chdir(WORK_DIR)
os.makedirs(OUT_ROOT, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_raster(path):
    ds = gdal.Open(path)
    if ds is None:
        return None, None
    band   = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    arr    = ds.ReadAsArray().astype(float)
    arr[arr < -1e10] = np.nan
    if nodata is not None:
        arr[arr == nodata] = np.nan
    geo_info = (ds.GetGeoTransform(), ds.GetProjection(),
                ds.RasterXSize, ds.RasterYSize)
    return arr, geo_info

def load_mask():
    arr, _ = load_raster(MASK_PATH)
    mask = arr.astype(bool)
    # Exclude lake pixels (nodata or 0 in the permafrost map)
    pf_arr, _ = load_raster(PERMAFROST_PATH)
    if pf_arr is not None:
        lake_mask = ((pf_arr == 0) | ~np.isfinite(pf_arr)) & mask
        mask[lake_mask] = False
        print(f'  Excluded {lake_mask.sum()} lake/nodata pixels from mask')
    return mask

def bin_mean(delta, terrain, mask, b_lo, b_hi):
    in_bin = mask & (terrain >= b_lo) & (terrain < b_hi) & np.isfinite(delta)
    if not in_bin.any():
        return np.nan, 0
    return float(np.nanmean(delta[in_bin])), int(in_bin.sum())

def plot_profile(ax, mids, bin_means, label, color):
    valid = [(m, v) for m, v in zip(mids, bin_means) if not np.isnan(v)]
    if not valid:
        return
    x_vals, y_vals = zip(*valid)
    ax.plot(y_vals, x_vals, color=color, linewidth=1.8,
            marker='o', markersize=5, label=label)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(mask):
    # ── Load terrain ──────────────────────────────────────────────────────────
    elevation = np.load(ELEV_PATH)
    slope, _  = load_raster(SLOPE_PATH)

    # ── Slope diagnostic ──────────────────────────────────────────────────────
    slope_masked = slope[mask & np.isfinite(slope)]
    print('Slope diagnostic:')
    print(f'  Min={slope_masked.min():.2f}  Max={slope_masked.max():.2f}  '
          f'Mean={slope_masked.mean():.2f}  Median={np.median(slope_masked):.2f}')
    for p in [75, 90, 95, 99]:
        print(f'  P{p}: {np.percentile(slope_masked, p):.2f}')

    s_max        = np.percentile(slope_masked, 99)
    slope_edges  = np.append(np.linspace(0, s_max, 7), slope_masked.max() + 1)
    slope_mids   = slope_edges[:-1] + np.diff(slope_edges) / 2
    slope_labels = [f'{slope_edges[i]:.1f}-{slope_edges[i+1]:.1f}'
                    for i in range(len(slope_edges) - 1)]
    print(f'  Slope bin edges: {[round(b, 2) for b in slope_edges]}')

    # ── Load mean delta rasters ────────────────────────────────────────────────
    delta_arrays = {}
    for crop in CROPS:
        path = f'{DELTA_DIR}/{crop["tag"]}_mean_delta_suit.tif'
        arr, _ = load_raster(path)
        if arr is None:
            print(f'  Warning: missing {path}')
            continue
        arr[~mask] = np.nan
        delta_arrays[crop['label']] = arr

    if not delta_arrays:
        print('No delta rasters found.')
        return

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        agg_delta = np.nanmean(np.stack(list(delta_arrays.values())), axis=0)
    agg_delta[~mask] = np.nan
    delta_arrays['OVERALL'] = agg_delta

    records = []

    # ── Elevation stratification ───────────────────────────────────────────────
    elev_bins = np.array(ELEV_BINS)
    elev_mids = elev_bins[:-1] + np.diff(elev_bins) / 2

    # Per-crop
    fig_e, ax_e = plt.subplots(figsize=(10, 8))
    ax_e.axvline(0, color='black', linewidth=0.8, linestyle='--')
    for i, (label, delta) in enumerate(
            {k: v for k, v in delta_arrays.items() if k != 'OVERALL'}.items()):
        bin_means = []
        for b_lo, b_hi in zip(elev_bins[:-1], elev_bins[1:]):
            mean_val, n_px = bin_mean(delta, elevation, mask, b_lo, b_hi)
            bin_means.append(mean_val)
            records.append({'crop': label, 'variable': 'elevation',
                            'bin_lo': b_lo, 'bin_hi': b_hi,
                            'bin_mid': (b_lo + b_hi) / 2,
                            'mean_delta': round(mean_val, 5) if not np.isnan(mean_val) else np.nan,
                            'n_pixels': n_px})
        plot_profile(ax_e, elev_mids, bin_means, label, CROP_COLORS(i))

    # Add overall line on top
    agg_elev_pre = []
    for b_lo, b_hi in zip(elev_bins[:-1], elev_bins[1:]):
        mean_val, _ = bin_mean(agg_delta, elevation, mask, b_lo, b_hi)
        agg_elev_pre.append(mean_val)
    valid_ov_e = [(m, v) for m, v in zip(elev_mids, agg_elev_pre) if not np.isnan(v)]
    if valid_ov_e:
        x_ov, y_ov = zip(*valid_ov_e)
        ax_e.plot(y_ov, x_ov, color='black', linewidth=3,
                  marker='D', markersize=6, label='OVERALL', zorder=5)
    ax_e.set_ylabel('Elevation (m)', fontsize=12)
    ax_e.set_xlabel('Mean delta Suitability (class units)', fontsize=12)
    ax_e.set_title('delta Suitability by Elevation Band — Per Crop\n(Mean 1999-2018)',
                   fontsize=13, fontweight='bold')
    ax_e.legend(fontsize=9, loc='upper right')
    ax_e.set_yticks(elev_mids)
    ax_e.set_yticklabels([f'{int(m)}m' for m in elev_mids])
    plt.tight_layout()
    fig_e.savefig(f'{OUT_ROOT}/elevation_profile_per_crop.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Overall elevation
    agg_elev_means, agg_elev_ns = [], []
    for b_lo, b_hi in zip(elev_bins[:-1], elev_bins[1:]):
        mean_val, n_px = bin_mean(agg_delta, elevation, mask, b_lo, b_hi)
        agg_elev_means.append(mean_val)
        agg_elev_ns.append(n_px)
        records.append({'crop': 'OVERALL', 'variable': 'elevation',
                        'bin_lo': b_lo, 'bin_hi': b_hi,
                        'bin_mid': (b_lo + b_hi) / 2,
                        'mean_delta': round(mean_val, 5) if not np.isnan(mean_val) else np.nan,
                        'n_pixels': n_px})

    fig_eo, ax_eo = plt.subplots(figsize=(8, 7))
    valid_e = [(m, v, n) for m, v, n in
               zip(elev_mids, agg_elev_means, agg_elev_ns) if not np.isnan(v)]
    if valid_e:
        mids_v, means_v, ns_v = zip(*valid_e)
        ax_eo.barh(mids_v, means_v,
                   height=np.diff(elev_bins[:len(mids_v)+1]).mean() * 0.7,
                   color=['#2166AC' if v >= 0 else '#D6604D' for v in means_v],
                   edgecolor='white', alpha=0.85)
        for m, v, n in zip(mids_v, means_v, ns_v):
            ax_eo.text(v + 0.0002 * np.sign(v) if v != 0 else 0.0002,
                       m, f'n={n}', va='center', fontsize=8,
                       ha='left' if v >= 0 else 'right')
    ax_eo.axvline(0, color='black', linewidth=0.8)
    ax_eo.set_ylabel('Elevation (m)', fontsize=12)
    ax_eo.set_xlabel('Mean delta Suitability (class units)', fontsize=12)
    ax_eo.set_title('Overall delta Suitability by Elevation Band\n'
                    '(Aggregate across all crops, mean 1999-2018)',
                    fontsize=13, fontweight='bold')
    ax_eo.set_yticks(elev_mids)
    ax_eo.set_yticklabels([f'{int(m)}m' for m in elev_mids])
    plt.tight_layout()
    fig_eo.savefig(f'{OUT_ROOT}/elevation_profile_overall.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Elevation profiles saved.')

    # ── Slope stratification ───────────────────────────────────────────────────

    # Per-crop
    fig_s, ax_s = plt.subplots(figsize=(10, 8))
    ax_s.axvline(0, color='black', linewidth=0.8, linestyle='--')
    for i, (label, delta) in enumerate(
            {k: v for k, v in delta_arrays.items() if k != 'OVERALL'}.items()):
        bin_means = []
        for b_lo, b_hi in zip(slope_edges[:-1], slope_edges[1:]):
            mean_val, n_px = bin_mean(delta, slope, mask, b_lo, b_hi)
            bin_means.append(mean_val)
            records.append({'crop': label, 'variable': 'slope',
                            'bin_lo': round(b_lo, 2), 'bin_hi': round(b_hi, 2),
                            'bin_mid': round((b_lo + b_hi) / 2, 2),
                            'mean_delta': round(mean_val, 5) if not np.isnan(mean_val) else np.nan,
                            'n_pixels': n_px})
        plot_profile(ax_s, slope_mids, bin_means, label, CROP_COLORS(i))

    # Add overall line on top
    agg_slope_pre = []
    for b_lo, b_hi in zip(slope_edges[:-1], slope_edges[1:]):
        mean_val, _ = bin_mean(agg_delta, slope, mask, b_lo, b_hi)
        agg_slope_pre.append(mean_val)
    valid_ov_s = [(m, v) for m, v in zip(slope_mids, agg_slope_pre) if not np.isnan(v)]
    if valid_ov_s:
        x_ov, y_ov = zip(*valid_ov_s)
        ax_s.plot(y_ov, x_ov, color='black', linewidth=3,
                  marker='D', markersize=6, label='OVERALL', zorder=5)
    ax_s.set_ylabel('Slope (degrees)', fontsize=12)
    ax_s.set_xlabel('Mean delta Suitability (class units)', fontsize=12)
    ax_s.set_title('delta Suitability by Slope Class — Per Crop\n(Mean 1999-2018)',
                   fontsize=13, fontweight='bold')
    ax_s.legend(fontsize=9, loc='upper right')
    ax_s.set_yticks(slope_mids)
    ax_s.set_yticklabels(slope_labels, fontsize=8)
    plt.tight_layout()
    fig_s.savefig(f'{OUT_ROOT}/slope_profile_per_crop.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Overall slope
    agg_slope_means, agg_slope_ns = [], []
    for b_lo, b_hi in zip(slope_edges[:-1], slope_edges[1:]):
        mean_val, n_px = bin_mean(agg_delta, slope, mask, b_lo, b_hi)
        agg_slope_means.append(mean_val)
        agg_slope_ns.append(n_px)
        records.append({'crop': 'OVERALL', 'variable': 'slope',
                        'bin_lo': round(b_lo, 2), 'bin_hi': round(b_hi, 2),
                        'bin_mid': round((b_lo + b_hi) / 2, 2),
                        'mean_delta': round(mean_val, 5) if not np.isnan(mean_val) else np.nan,
                        'n_pixels': n_px})

    fig_so, ax_so = plt.subplots(figsize=(8, 7))
    valid_s = [(m, v, n, l) for m, v, n, l in
               zip(slope_mids, agg_slope_means, agg_slope_ns, slope_labels)
               if not np.isnan(v)]
    if valid_s:
        mids_v, means_v, ns_v, lbls_v = zip(*valid_s)
        ax_so.barh(range(len(mids_v)), means_v,
                   color=['#2166AC' if v >= 0 else '#D6604D' for v in means_v],
                   edgecolor='white', alpha=0.85)
        for i, (v, n) in enumerate(zip(means_v, ns_v)):
            ax_so.text(v + 0.0002 * np.sign(v) if v != 0 else 0.0002,
                       i, f'n={n}', va='center', fontsize=8,
                       ha='left' if v >= 0 else 'right')
        ax_so.set_yticks(range(len(mids_v)))
        ax_so.set_yticklabels(lbls_v, fontsize=9)
    ax_so.axvline(0, color='black', linewidth=0.8)
    ax_so.set_ylabel('Slope Class', fontsize=12)
    ax_so.set_xlabel('Mean delta Suitability (class units)', fontsize=12)
    ax_so.set_title('Overall delta Suitability by Slope Class\n'
                    '(Aggregate across all crops, mean 1999-2018)',
                    fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig_so.savefig(f'{OUT_ROOT}/slope_profile_overall.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Slope profiles saved.')

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    df.to_csv(f'{OUT_ROOT}/elevation_slope_stats.csv', index=False)

    print('\nOverall elevation profile:')
    print(df[(df['crop'] == 'OVERALL') & (df['variable'] == 'elevation')][
        ['bin_lo', 'bin_hi', 'mean_delta', 'n_pixels']].to_string(index=False))
    print('\nOverall slope profile:')
    print(df[(df['crop'] == 'OVERALL') & (df['variable'] == 'slope')][
        ['bin_lo', 'bin_hi', 'mean_delta', 'n_pixels']].to_string(index=False))
    print(f'\nAll outputs saved to: {OUT_ROOT}/')


if __name__ == '__main__':
    mask = load_mask()
    run(mask)