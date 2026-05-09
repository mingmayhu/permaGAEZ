"""
Chapter 4 — Regional Climate Change and Permafrost Thaw Figures
================================================================
Produces publication-quality figures for Chapter 4:

Section 4.1 — Climate Trends (1979–2018):
  - Regional mean time series: TempMax, TempMin, Precipitation
  - Spatial maps of mean values and change (post minus pre 1999)
  - MK trend test with Sen's slope for each variable

Section 4.2 — Permafrost Thaw Trends (1979–2018):
  - Regional mean time series: ALT, Soil Moisture
  - Spatial maps of mean values and change
  - MK trend test with Sen's slope
  - Permafrost presence/absence map
  - PCF not included here (assumed already in thesis Figure 1)

Outputs written to:
  ./results_analysis/outputs/chapter4/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pymannkendall import original_test as mk_test
from osgeo import gdal
from matplotlib.gridspec import GridSpec

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR      = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH     = r'./data_input/qilian_mask_new.tif'
PERM_MAP_PATH = r'./data_input/permafrost_qilian.tif'
PERM_DIR      = r'./data_input/permafrost_yearly'
CLIM_DIR      = r'./data_input/climate_yearly'
OUT_ROOT      = r'./results/climate_permafrost_trends/outputs'

YEARS_ALL  = list(range(1979, 2019))
YEARS_PRE  = list(range(1979, 1999))
YEARS_POST = list(range(1999, 2019))
DIVERGENCE_YEAR = 1999
ALPHA = 0.05

# Plot styling
OBS_COLOR  = '#2166AC'
TREND_COLOR = '#D6604D'
FONTSIZE_TITLE = 13
FONTSIZE_LABEL = 11
FONTSIZE_TICK  = 9
DPI = 150

os.chdir(WORK_DIR)
os.makedirs(OUT_ROOT, exist_ok=True)
for sub in ['climate', 'permafrost']:
    os.makedirs(f'{OUT_ROOT}/{sub}', exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_raster(path):
    ds = gdal.Open(path)
    if ds is None:
        return None
    band   = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    arr    = ds.ReadAsArray().astype(float)
    arr[arr < -1e10] = np.nan
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr

def load_mask():
    return load_raster(MASK_PATH).astype(bool)

def regional_mean(arr, mask):
    valid = mask & np.isfinite(arr)
    return float(np.nanmean(arr[valid])) if valid.any() else np.nan

def load_clim_annual(var_file, years, mask, agg='mean'):
    """Load climate variable — aggregate across days, return annual value per year."""
    annual = []
    for year in years:
        path = f'{CLIM_DIR}/{year}/{var_file}'
        if not os.path.exists(path):
            annual.append(np.nan)
            continue
        arr = np.load(path).astype(float)
        if arr.ndim == 3:
            arr = np.nansum(arr, axis=2) if agg == 'sum' \
                  else np.nanmean(arr, axis=2)
        arr[~mask] = np.nan
        annual.append(regional_mean(arr, mask))
    return np.array(annual)

def load_clim_spatial(var_file, years, mask, agg='mean'):
    """Load climate variable — return mean spatial map over given years."""
    stack = []
    for year in years:
        path = f'{CLIM_DIR}/{year}/{var_file}'
        if not os.path.exists(path):
            continue
        arr = np.load(path).astype(float)
        if arr.ndim == 3:
            arr = np.nansum(arr, axis=2) if agg == 'sum' \
                  else np.nanmean(arr, axis=2)
        arr[~mask] = np.nan
        stack.append(arr)
    return np.nanmean(stack, axis=0) if stack else np.full(mask.shape, np.nan)

def load_perm_annual(var_file, years, mask, agg='max'):
    """Load permafrost variable — aggregate across days, return annual value."""
    annual = []
    for year in years:
        path = f'{PERM_DIR}/{year}/{var_file}'
        if not os.path.exists(path):
            annual.append(np.nan)
            continue
        arr = np.load(path).astype(float)
        if arr.ndim == 3:
            arr = np.nanmax(arr, axis=2) if agg == 'max' \
                  else np.nanmean(arr, axis=2)
        arr[~mask] = np.nan
        annual.append(regional_mean(arr, mask))
    return np.array(annual)

def load_perm_spatial(var_file, years, mask, agg='max'):
    """Load permafrost variable — return mean spatial map over given years."""
    stack = []
    for year in years:
        path = f'{PERM_DIR}/{year}/{var_file}'
        if not os.path.exists(path):
            continue
        arr = np.load(path).astype(float)
        if arr.ndim == 3:
            arr = np.nanmax(arr, axis=2) if agg == 'max' \
                  else np.nanmean(arr, axis=2)
        arr[~mask] = np.nan
        stack.append(arr)
    return np.nanmean(stack, axis=0) if stack else np.full(mask.shape, np.nan)

def run_mk(series):
    s = np.array(series, dtype=float)
    valid = np.isfinite(s)
    if valid.sum() < 4:
        return None
    mk = mk_test(s[valid])
    line = np.full(len(s), np.nan)
    line[valid] = mk.intercept + mk.slope * np.arange(valid.sum())
    return {
        'tau': round(mk.Tau, 3), 'p': round(mk.p, 4),
        'slope': round(mk.slope, 4), 'significant': mk.p < ALPHA,
        'trend': mk.trend, 'sen_line': line,
        'intercept': round(mk.intercept, 4),   # ← add this
    }

def plot_timeseries(ax, years, series, mk_result, ylabel, title,
                    color=OBS_COLOR, vline=True):
    """Plot time series with Sen's slope line."""
    ax.plot(years, series, color=color, linewidth=2,
            marker='o', markersize=3, label='Annual mean')
    if mk_result:
        sig = '★' if mk_result['significant'] else ''
        ax.plot(years, mk_result['sen_line'], color=TREND_COLOR,
                linewidth=2, linestyle='--',
                label=f"Sen's slope: {mk_result['slope']:.4f}/yr "
                      f"(p={mk_result['p']:.3f}){sig}")
    if vline:
        ax.axvline(DIVERGENCE_YEAR, color='grey', linestyle=':',
                   linewidth=1.2, label='1999 divergence')
    ax.set_xlabel('Year', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL)
    ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xticks(years[::4])
    ax.set_xticklabels(years[::4], rotation=45, ha='right',
                       fontsize=FONTSIZE_TICK)
    ax.tick_params(axis='y', labelsize=FONTSIZE_TICK)

def plot_spatial_pair(fig, axes, mean_arr, change_arr, mask,
                      mean_title, change_title,
                      mean_cmap, mean_label,
                      change_cmap='RdBu', change_label='Change'):
    """Plot mean map and change map side by side."""
    # Mean map
    disp_mean = np.where(mask, mean_arr, np.nan)
    vmax_mean = np.nanpercentile(disp_mean[mask], 98)
    vmin_mean = np.nanpercentile(disp_mean[mask], 2)
    im0 = axes[0].imshow(disp_mean, cmap=mean_cmap,
                         vmin=vmin_mean, vmax=vmax_mean)
    axes[0].set_title(mean_title, fontsize=FONTSIZE_TITLE,
                      fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], shrink=0.75, label=mean_label)

    # Change map
    disp_chg = np.where(mask, change_arr, np.nan)
    vlim = np.nanpercentile(np.abs(disp_chg[mask & np.isfinite(disp_chg)]), 98)
    if vlim == 0 or np.isnan(vlim):
        vlim = 0.1
    im1 = axes[1].imshow(disp_chg, cmap=change_cmap,
                         vmin=-vlim, vmax=vlim)
    axes[1].set_title(change_title, fontsize=FONTSIZE_TITLE,
                      fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], shrink=0.75, label=change_label)


# ── Section 4.1: Climate Trends ───────────────────────────────────────────────

def section_climate(mask):
    print('\n[4.1] Climate trends …')
    years_arr = np.array(YEARS_ALL)
    out_dir   = f'{OUT_ROOT}/climate'

    # Load annual regional means
    print('  Loading climate data …')
    tmax_ann  = load_clim_annual('TempMax.npy', YEARS_ALL, mask, 'mean')
    tmin_ann  = load_clim_annual('TempMin.npy', YEARS_ALL, mask, 'mean')
    tmean_ann = (tmax_ann + tmin_ann) / 2
    prec_ann  = load_clim_annual('Precip.npy',  YEARS_ALL, mask, 'sum')

    # Load spatial maps
    tmax_pre   = load_clim_spatial('TempMax.npy', YEARS_PRE,  mask, 'mean')
    tmax_post  = load_clim_spatial('TempMax.npy', YEARS_POST, mask, 'mean')
    tmin_pre   = load_clim_spatial('TempMin.npy', YEARS_PRE,  mask, 'mean')
    tmin_post  = load_clim_spatial('TempMin.npy', YEARS_POST, mask, 'mean')
    tmean_pre  = (tmax_pre  + tmin_pre)  / 2
    tmean_post = (tmax_post + tmin_post) / 2
    prec_pre   = load_clim_spatial('Precip.npy',  YEARS_PRE,  mask, 'sum')
    prec_post  = load_clim_spatial('Precip.npy',  YEARS_POST, mask, 'sum')

    # MK tests
    mk_tmean = run_mk(tmean_ann)
    mk_prec  = run_mk(prec_ann)

    print(f'  TempMean: τ={mk_tmean["tau"]}, p={mk_tmean["p"]}, '
          f'slope={mk_tmean["slope"]}/yr')
    print(f'  Precip:   τ={mk_prec["tau"]},  p={mk_prec["p"]}, '
          f'slope={mk_prec["slope"]}/yr')

    # ── Figure 1: Time series — temperature and precipitation ─────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    plot_timeseries(axes[0], years_arr, tmean_ann, mk_tmean,
                    'Mean Temperature (°C)',
                    'Regional Mean Temperature (1979–2018)\n'
                    '[computed as (TempMax + TempMin) / 2]',
                    vline=False)
    plot_timeseries(axes[1], years_arr, prec_ann, mk_prec,
                    'Annual Total Precipitation (mm)',
                    'Regional Annual Total Precipitation (1979–2018)',
                    vline=False)

    fig.suptitle('Qilian Mountain Region — Climate Trends (1979–2018)',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(f'{out_dir}/climate_timeseries.png',
                dpi=DPI, bbox_inches='tight')
    plt.close()
    print('  ✓ Climate time series saved')

    # ── Figure 2: Spatial maps — temperature ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    plot_spatial_pair(fig, axes, tmean_post, tmean_post - tmean_pre, mask,
                      'Mean Temperature 1999–2018 (°C)',
                      'ΔTemp (1999–2018 minus 1979–1998)',
                      'RdYlBu_r', '°C', 'RdBu_r', 'Δ°C')

    fig.suptitle('Qilian Mountain Region — Temperature Spatial Patterns',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{out_dir}/temperature_spatial.png',
                dpi=DPI, bbox_inches='tight')
    plt.close()
    print('  ✓ Temperature spatial maps saved')

    # ── Figure 3: Spatial maps — precipitation ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    plot_spatial_pair(fig, axes, prec_post, prec_post - prec_pre, mask,
                      'Mean Annual Precip 1999–2018 (mm)',
                      'ΔPrecip (1999–2018 minus 1979–1998)',
                      'YlGnBu', 'mm', 'BrBG', 'Δmm')

    fig.suptitle('Qilian Mountain Region — Precipitation Spatial Patterns',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{out_dir}/precipitation_spatial.png',
                dpi=DPI, bbox_inches='tight')
    plt.close()
    print('  ✓ Precipitation spatial maps saved')

    # Save MK results
    pd.DataFrame([
        {'variable': 'TempMean', 'tau': mk_tmean['tau'], 'p': mk_tmean['p'],
         'slope_per_yr': mk_tmean['slope'], 'significant': mk_tmean['significant'],
         'trend': mk_tmean['trend']},
        {'variable': 'Precip',   'tau': mk_prec['tau'],  'p': mk_prec['p'],
         'slope_per_yr': mk_prec['slope'],  'significant': mk_prec['significant'],
         'trend': mk_prec['trend']},
    ]).to_csv(f'{out_dir}/climate_mk_results.csv', index=False)
    print(f'  TempMean: slope={mk_tmean["slope"]}/yr, intercept={mk_tmean["intercept"]}')
    print(f'  Precip:   slope={mk_prec["slope"]}/yr,  intercept={mk_prec["intercept"]}')


# ── Section 4.2: Permafrost Thaw Trends ──────────────────────────────────────

def section_permafrost(mask):
    print('\n[4.2] Permafrost thaw trends …')
    years_arr = np.array(YEARS_ALL)
    out_dir   = f'{OUT_ROOT}/permafrost'

    # Load annual regional means
    print('  Loading permafrost data …')
    alt_ann = load_perm_annual('active_layer_depth.npy', YEARS_ALL, mask, 'max')
    sm_ann  = load_perm_annual('avail_soil_moisture.npy', YEARS_ALL, mask, 'mean')

    # Load spatial maps
    alt_pre  = load_perm_spatial('active_layer_depth.npy', YEARS_PRE,  mask, 'max')
    alt_post = load_perm_spatial('active_layer_depth.npy', YEARS_POST, mask, 'max')
    sm_pre   = load_perm_spatial('avail_soil_moisture.npy', YEARS_PRE,  mask, 'mean')
    sm_post  = load_perm_spatial('avail_soil_moisture.npy', YEARS_POST, mask, 'mean')

    # MK tests
    mk_alt = run_mk(alt_ann)
    mk_sm  = run_mk(sm_ann)

    print(f'  ALT:          τ={mk_alt["tau"]}, p={mk_alt["p"]}, '
          f'slope={mk_alt["slope"]} m/yr')
    print(f'  Soil Moisture: τ={mk_sm["tau"]},  p={mk_sm["p"]}, '
          f'slope={mk_sm["slope"]}/yr')

    # ── Figure 4: Time series — ALT and soil moisture ─────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    plot_timeseries(axes[0], years_arr, alt_ann, mk_alt,
                    'Mean Annual Max ALT (m)',
                    'Regional Mean Active Layer Thickness (1979–2018)')
    plot_timeseries(axes[1], years_arr, sm_ann, mk_sm,
                    'Mean Available Soil Moisture (mm)',
                    'Regional Mean Available Soil Moisture (1979–2018)',
                    color='#1a9850')

    fig.suptitle('Qilian Mountain Region — Permafrost Thaw Trends (1979–2018)',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(f'{out_dir}/permafrost_timeseries.png',
                dpi=DPI, bbox_inches='tight')
    plt.close()
    print('  ✓ Permafrost time series saved')

    # ── Figure 5: Spatial maps — ALT ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    plot_spatial_pair(fig, axes, alt_post, alt_post - alt_pre, mask,
                      'Mean Annual Max ALT 1999–2018 (m)',
                      'ΔALT (1999–2018 minus 1979–1998)',
                      'YlOrRd', 'm', 'RdBu', 'Δm')

    fig.suptitle('Qilian Mountain Region — Active Layer Thickness Spatial Patterns',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{out_dir}/alt_spatial.png',
                dpi=DPI, bbox_inches='tight')
    plt.close()
    print('  ✓ ALT spatial maps saved')

    # ── Figure 6: Spatial maps — Soil moisture ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    plot_spatial_pair(fig, axes, sm_post, sm_post - sm_pre, mask,
                      'Mean Soil Moisture 1999–2018',
                      'ΔSoil Moisture (1999–2018 minus 1979–1998)',
                      'YlGnBu', 'Soil Moisture', 'BrBG', 'Δ Soil Moisture')

    fig.suptitle('Qilian Mountain Region — Soil Moisture Spatial Patterns',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{out_dir}/soil_moisture_spatial.png',
                dpi=DPI, bbox_inches='tight')
    plt.close()
    print('  ✓ Soil moisture spatial maps saved')

    # ── Figure 7: Permafrost presence/absence map ─────────────────────────────
    perm_arr = load_raster(PERM_MAP_PATH)
    perm_arr = np.where(np.isfinite(perm_arr), perm_arr, 0)

    display = np.full(mask.shape, np.nan)
    display[mask & (perm_arr == 1)] = 1   # permafrost
    display[mask & (perm_arr == 2)] = 2   # seasonally frozen

    cmap   = mcolors.ListedColormap(['#2166AC', '#808080'])
    bounds = [0.5, 1.5, 2.5]
    norm   = mcolors.BoundaryNorm(bounds, cmap.N)

    n_perm     = int(np.sum(mask & (perm_arr == 1)))
    n_seasonal = int(np.sum(mask & (perm_arr == 2)))
    n_total    = int(mask.sum())

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(display, cmap=cmap, norm=norm)
    ax.axis('off')
    ax.set_title(
        f'Permafrost Distribution — Qilian Mountain Region\n'
        f'Permafrost: {n_perm} pixels ({n_perm/n_total*100:.1f}%)  |  '
        f'Seasonally Frozen: {n_seasonal} pixels ({n_seasonal/n_total*100:.1f}%)',
        fontsize=FONTSIZE_TITLE, fontweight='bold'
    )
    patches = [
        mpatches.Patch(color='#2166AC',
                       label=f'Permafrost ({n_perm/n_total*100:.1f}%)'),
        mpatches.Patch(color='#808080',
                       label=f'Seasonally Frozen ({n_seasonal/n_total*100:.1f}%)'),
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=10,
              framealpha=0.9)
    plt.tight_layout()
    fig.savefig(f'{out_dir}/permafrost_distribution.png',
                dpi=DPI, bbox_inches='tight')
    plt.close()
    print('  ✓ Permafrost distribution map saved')
    print(f'  ALT:      slope={mk_alt["slope"]} m/yr, intercept={mk_alt["intercept"]}')
    print(f'  SM:       slope={mk_sm["slope"]}/yr,    intercept={mk_sm["intercept"]}')

    # ── Figure 8: Combined summary — 4 panel hotspot map ─────────────────────
    tmax_pre_h  = load_clim_spatial('TempMax.npy', YEARS_PRE,  mask, 'mean')
    tmax_post_h = load_clim_spatial('TempMax.npy', YEARS_POST, mask, 'mean')
    tmin_pre_h  = load_clim_spatial('TempMin.npy', YEARS_PRE,  mask, 'mean')
    tmin_post_h = load_clim_spatial('TempMin.npy', YEARS_POST, mask, 'mean')
    tmean_pre_h  = (tmax_pre_h  + tmin_pre_h)  / 2
    tmean_post_h = (tmax_post_h + tmin_post_h) / 2
    prec_pre_h  = load_clim_spatial('Precip.npy',  YEARS_PRE,  mask, 'sum')
    prec_post_h = load_clim_spatial('Precip.npy',  YEARS_POST, mask, 'sum')

    fig = plt.figure(figsize=(16, 14))
    gs  = GridSpec(3, 2, figure=fig, height_ratios=[0.6, 1, 1], hspace=0.35)
    ax_perm = fig.add_subplot(gs[0, :])   # top row spanning both columns

    # Top row — permafrost distribution spanning full width
    # ax_perm = fig.add_subplot(3, 1, 1)
    perm_display = np.full(mask.shape, np.nan)
    perm_display[mask & (perm_arr == 1)] = 1
    perm_display[mask & (perm_arr == 2)] = 2
    cmap_p  = mcolors.ListedColormap(['#2166AC', '#808080'])
    norm_p  = mcolors.BoundaryNorm([0.5, 1.5, 2.5], cmap_p.N)
    ax_perm.imshow(perm_display, cmap=cmap_p, norm=norm_p)
    ax_perm.set_title('Permafrost Distribution', fontsize=FONTSIZE_TITLE,
                    fontweight='bold')
    ax_perm.axis('off')
    patches = [mpatches.Patch(color='#2166AC', label='Permafrost'),
            mpatches.Patch(color='#808080', label='Seasonally Frozen')]
    ax_perm.legend(handles=patches, loc='lower right', fontsize=9)

    # Bottom 2x2 — change panels
    axes    = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
           fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]

    for ax, arr, title, cmap, label in [
        (axes[0], alt_post - alt_pre,          'ΔALT (m)',             'RdBu',   'Δm'),
        (axes[1], sm_post  - sm_pre,           'ΔSoil Moisture (mm)',  'BrBG',   'Δmm'),
        (axes[2], tmean_post_h - tmean_pre_h,  'ΔTemp (°C)',           'RdBu_r', 'Δ°C'),
        (axes[3], prec_post_h  - prec_pre_h,   'ΔPrecipitation (mm)',  'BrBG',   'Δmm'),
    ]:
        disp = np.where(mask, arr, np.nan)
        vals = disp[mask & np.isfinite(disp)]
        vlim = np.nanpercentile(np.abs(vals), 98) if len(vals) > 0 else 1.0
        if vlim == 0:
            vlim = 0.1
        im = ax.imshow(disp, cmap=cmap, vmin=-vlim, vmax=vlim)
        ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.75, label=label)

    fig.suptitle(
        'Qilian Mountain Region — Hotspots of Change\n'
        '(1999–2018 minus 1979–1998)',
        fontsize=15, fontweight='bold'
    )
    plt.tight_layout()
    fig.savefig(f'{out_dir}/change_hotspots.png',
                dpi=DPI, bbox_inches='tight')
    plt.close()
    print('  ✓ Change hotspot summary map saved')

    # Save MK results
    pd.DataFrame([
        {'variable': 'ALT', 'tau': mk_alt['tau'], 'p': mk_alt['p'],
         'slope_per_yr': mk_alt['slope'], 'significant': mk_alt['significant'],
         'trend': mk_alt['trend']},
        {'variable': 'Soil Moisture', 'tau': mk_sm['tau'], 'p': mk_sm['p'],
         'slope_per_yr': mk_sm['slope'], 'significant': mk_sm['significant'],
         'trend': mk_sm['trend']},
    ]).to_csv(f'{out_dir}/permafrost_mk_results.csv', index=False)

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    mask = load_mask()
    section_climate(mask)
    section_permafrost(mask)
    
    print(f'\n✓ All Chapter 4 figures saved to: {OUT_ROOT}/')