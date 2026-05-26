"""
Figure: Permafrost-Climate Interaction Heatmap
===============================================
Temperature × Precipitation bins, showing Spearman r between
permafrost variables (ALT / soil moisture) and ΔSuitability within each bin.

Two panels stacked vertically:
  Top    — Spearman r between ALT and ΔSuitability, within each climate bin
  Bottom — Spearman r between soil moisture and ΔSuitability, within each bin

Asks: in which climate regimes does the permafrost signal most strongly
drive the thaw impact on suitability?

- Cells coloured by Spearman r (RdBu, anchored at 0)
- Purple box outline where r is significant (p < 0.05)
- Pixel count annotated in each cell
- make_axes_locatable colorbar, white grid lines, Helvetica font
- Permafrost pixels only (seasonally frozen excluded)

Input:
  ./data_input/climate_yearly/
  ./data_input/permafrost_yearly/
  ./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/5_spatial/1_mean_delta/
      overall_mean_delta_suit.tif
  ./data_input/permafrost_qilian.tif
  ./data_input/qilian mask.tif

Output:
  ./results/permafrost_thaw_impact/thaw_vs_nothaw/figures/
      fig_thaw_permafrost_space.png
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import spearmanr
from osgeo import gdal

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR        = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
PERM_DIR        = r'./data_input/permafrost_yearly'
CLIM_DIR        = r'./data_input/climate_yearly'
LGP_OBS_DIR     = r'./data_output/module1'
LGP_CF_DIR      = r'./data_output/module1_nothaw'
DELTA_PATH      = (r'./results/permafrost_thaw_impact/thaw_vs_nothaw/outputs/'
                   r'5_spatial/1_mean_delta/overall_mean_delta_suit.tif')
MASK_PATH       = r'./data_input/qilian mask.tif'
PERMAFROST_PATH = r'./data_input/permafrost_qilian.tif'
OUT_DIR         = r'./results/permafrost_thaw_impact/thaw_vs_nothaw/figures'
OUT_PATH        = f'{OUT_DIR}/fig_thaw_permafrost_space.png'
DPI             = 300

FONT      = 'Helvetica'
BOLD_PATH = '/Users/ming-mayhu/Library/Fonts/Helvetica LT 75 Bold.ttf'
REG_PATH  = '/System/Library/Fonts/Helvetica.ttc'

YEARS_PRE  = list(range(1979, 1999))
YEARS_POST = list(range(1999, 2019))
N_BINS     = 5    # bins per climate axis — kept at 5 since permafrost pixels are fewer
MIN_PIX    = 8    # minimum pixels per bin to compute correlation
SIG_ALPHA  = 0.05
vmax       = 0.5  # Spearman r range

os.chdir(WORK_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# ── Font setup ────────────────────────────────────────────────────────────────
try:
    fp_bold = FontProperties(fname=BOLD_PATH)
    fp_reg  = FontProperties(fname=REG_PATH)
except Exception:
    fp_bold = FontProperties(weight='bold')
    fp_reg  = FontProperties()

# ── Seaborn theme ─────────────────────────────────────────────────────────────
sns.set_theme(
    style='ticks',
    rc={
        'font.family':        'sans-serif',
        'font.sans-serif':    [FONT],
        'axes.edgecolor':     '#000000',
        'axes.linewidth':     0.8,
    }
)

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
    arr    = load_raster(MASK_PATH)
    mask   = arr.astype(bool)
    pf_arr = load_raster(PERMAFROST_PATH)
    if pf_arr is not None:
        lake_mask = ((pf_arr == 0) | ~np.isfinite(pf_arr)) & mask
        mask[lake_mask] = False
    return mask, pf_arr

def load_mean(file_template, years, mask, agg='mean'):
    """Load .npy files for a list of years and return the temporal mean."""
    stack = []
    for year in years:
        path = file_template.format(year=year)
        if not os.path.exists(path):
            continue
        arr = np.load(path).astype(float)
        if arr.ndim == 3:
            arr = (np.nansum(arr, axis=2)  if agg == 'sum'
                   else np.nanmax(arr, axis=2) if agg == 'max'
                   else np.nanmean(arr, axis=2))
        if arr.shape != mask.shape:
            continue
        arr[~mask] = np.nan
        stack.append(arr)
    if not stack:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(np.stack(stack), axis=0)

# ── Load data ─────────────────────────────────────────────────────────────────
print('Loading mask ...')
mask, pf_arr = load_mask()

# Permafrost pixels only
pf_mask = mask & np.isin(np.round(pf_arr).astype(int), [1, 2])
print(f'  Permafrost pixels: {pf_mask.sum()} '
      f'({pf_mask.sum()/mask.sum()*100:.1f}% of study area)')

print('Loading climate means (1999-2018) ...')
temp_max = load_mean(f'{CLIM_DIR}/{{year}}/TempMax.npy', YEARS_POST, pf_mask, agg='mean')
temp_min = load_mean(f'{CLIM_DIR}/{{year}}/TempMin.npy', YEARS_POST, pf_mask, agg='mean')
precip   = load_mean(f'{CLIM_DIR}/{{year}}/Precip.npy',  YEARS_POST, pf_mask, agg='sum')
temp     = (temp_max + temp_min) / 2
print('  temp_mean derived from (TempMax + TempMin) / 2')

print('Loading permafrost variable means (pre and post) for delta computation ...')
alt_post = load_mean(f'{PERM_DIR}/{{year}}/active_layer_depth.npy',
                     YEARS_POST, pf_mask, agg='max')
alt_pre  = load_mean(f'{PERM_DIR}/{{year}}/active_layer_depth.npy',
                     YEARS_PRE,  pf_mask, agg='max')
sm_post  = load_mean(f'{PERM_DIR}/{{year}}/avail_soil_moisture.npy',
                     YEARS_POST, pf_mask, agg='mean')
sm_pre   = load_mean(f'{PERM_DIR}/{{year}}/avail_soil_moisture.npy',
                     YEARS_PRE,  pf_mask, agg='mean')

alt = alt_post - alt_pre   # ΔALT: positive = deeper active layer post-1999
sm  = sm_post  - sm_pre    # ΔSM:  positive = more soil moisture post-1999
alt[~pf_mask] = np.nan
sm[~pf_mask]  = np.nan
print(f'  ΔALT range: [{alt[pf_mask & np.isfinite(alt)].min():.3f}, '
      f'{alt[pf_mask & np.isfinite(alt)].max():.3f}] m')
print(f'  ΔSM range:  [{sm[pf_mask & np.isfinite(sm)].min():.3f}, '
      f'{sm[pf_mask & np.isfinite(sm)].max():.3f}]')

print('Loading mean ΔLGP (thaw minus no-thaw, 1999-2018) ...')
lgp_stack = []
for year in YEARS_POST:
    obs_path = f'{LGP_OBS_DIR}/{year}/LGP New.tif'
    cf_path  = f'{LGP_CF_DIR}/{year}/LGP New.tif'
    obs = load_raster(obs_path)
    cf  = load_raster(cf_path)
    if obs is None or cf is None:
        continue
    if obs.shape != pf_mask.shape or cf.shape != pf_mask.shape:
        continue
    diff = obs.astype(float) - cf.astype(float)
    diff[~pf_mask] = np.nan
    lgp_stack.append(diff)
if lgp_stack:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        lgp_delta = np.nanmean(np.stack(lgp_stack), axis=0)
    lgp_delta[~pf_mask] = np.nan
    print(f'  ΔLGP loaded: {len(lgp_stack)} years')
else:
    lgp_delta = None
    print('  WARNING: no LGP files found')

print('Loading overall mean delta ...')
delta = load_raster(DELTA_PATH)
if delta is not None:
    delta[~pf_mask] = np.nan

# ── Flatten to permafrost pixels with all variables present ───────────────────
lgp_valid = np.isfinite(lgp_delta) if lgp_delta is not None else np.zeros_like(pf_mask)

valid = (pf_mask
         & np.isfinite(temp)
         & np.isfinite(precip)
         & np.isfinite(alt)
         & np.isfinite(sm)
         & np.isfinite(delta)
         & lgp_valid)

temp_flat   = temp[valid]
precip_flat = precip[valid]
alt_flat    = alt[valid]
sm_flat     = sm[valid]
delta_flat  = delta[valid]
lgp_flat    = lgp_delta[valid] if lgp_delta is not None else np.full(valid.sum(), np.nan)

print(f'  Valid pixels: {valid.sum()}')
print(f'  Temp range:   [{temp_flat.min():.1f}, {temp_flat.max():.1f}] °C')
print(f'  Precip range: [{precip_flat.min():.0f}, {precip_flat.max():.0f}] mm')
print(f'  ΔALT range:   [{alt_flat.min():.3f}, {alt_flat.max():.3f}] m')
print(f'  ΔSM range:    [{sm_flat.min():.3f}, {sm_flat.max():.3f}]')
print(f'  ΔLGP range:   [{lgp_flat.min():.2f}, {lgp_flat.max():.2f}] days')

# ── Build climate bins ────────────────────────────────────────────────────────
temp_edges   = np.unique(np.nanpercentile(temp_flat,   np.linspace(0, 100, N_BINS + 1)))
precip_edges = np.unique(np.nanpercentile(precip_flat, np.linspace(0, 100, N_BINS + 1)))
n_temp   = len(temp_edges)   - 1
n_precip = len(precip_edges) - 1

temp_labels   = [f'{temp_edges[i]:.1f}–{temp_edges[i+1]:.1f}°C'
                 for i in range(n_temp)]
precip_labels = [f'{precip_edges[i]:.0f}–{precip_edges[i+1]:.0f} mm'
                 for i in range(n_precip)]

ti_arr = np.clip(np.digitize(temp_flat,   temp_edges)   - 1, 0, n_temp   - 1)
pi_arr = np.clip(np.digitize(precip_flat, precip_edges) - 1, 0, n_precip - 1)

# ── Build correlation grids ───────────────────────────────────────────────────
def build_corr_grid(perm_var_flat):
    """Compute Spearman r between perm_var and delta within each climate bin."""
    r_grid     = np.full((n_precip, n_temp), np.nan)
    sig_grid   = np.zeros((n_precip, n_temp), dtype=bool)
    count_grid = np.zeros((n_precip, n_temp), dtype=int)

    for pi in range(n_precip):
        for ti in range(n_temp):
            in_bin = (pi_arr == pi) & (ti_arr == ti)
            n      = in_bin.sum()
            count_grid[pi, ti] = n
            if n < MIN_PIX:
                continue
            pv = perm_var_flat[in_bin]
            dv = delta_flat[in_bin]
            # Need variance in both variables to compute correlation
            if np.std(pv) < 1e-10 or np.std(dv) < 1e-10:
                continue
            r, p = spearmanr(pv, dv)
            r_grid[pi, ti]   = float(r)
            sig_grid[pi, ti] = p < SIG_ALPHA

    return r_grid, sig_grid, count_grid

print('\nComputing ALT correlations per climate bin ...')
r_alt,  sig_alt,  count_alt  = build_corr_grid(alt_flat)
print('Computing soil moisture correlations per climate bin ...')
r_sm,   sig_sm,   count_sm   = build_corr_grid(sm_flat)
print('Computing ΔLGP correlations per climate bin ...')
r_lgp,  sig_lgp,  count_lgp  = build_corr_grid(lgp_flat)

# ── Figure ────────────────────────────────────────────────────────────────────
panels = [
    {'r_grid': r_alt,  'sig_grid': sig_alt,  'count_grid': count_alt,
     'ylabel': 'Active layer thickness (delta) \nvs suitability (delta)','letter': '(a)'},
    {'r_grid': r_sm,   'sig_grid': sig_sm,   'count_grid': count_sm,
     'ylabel': 'Available soil moisture (delta) \nvs suitability (delta)', 'letter': '(b)'},
    {'r_grid': r_lgp,  'sig_grid': sig_lgp,  'count_grid': count_lgp,
     'ylabel': 'Length of Growing Period (delta) \nvs (delta)', 'letter': '(c)'},
]

fig_w = n_temp   * 1.4 + 2.5
fig_h = n_precip * 1.0 * len(panels) + 2.5
fig, axes = plt.subplots(len(panels), 1, figsize=(fig_w, fig_h))
fig.patch.set_facecolor('white')

for ax, panel in zip(axes, panels):
    r_grid     = panel['r_grid']
    sig_grid   = panel['sig_grid']
    count_grid = panel['count_grid']

    im = ax.imshow(r_grid, cmap='RdBu', vmin=-vmax, vmax=vmax,
                   aspect='auto', origin='lower')

    # Colorbar
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes('right', size='3%', pad=0.1)
    cbar    = plt.colorbar(im, cax=cax)
    cbar.set_label('Spearman r', fontsize=12, fontproperties=fp_reg)
    cbar.ax.tick_params(labelsize=11)
    total_n = 0
    # Cell text + significance box
    for pi in range(n_precip):
        for ti in range(n_temp):
            n   = count_grid[pi, ti]
            total_n += n
            val = r_grid[pi, ti]
            sig = sig_grid[pi, ti]

            if n < MIN_PIX:
                ax.text(ti, pi, f'n={n}',
                        ha='center', va='center',
                        fontsize=9, color='#cccccc',
                        fontproperties=fp_reg)
                continue
            if not np.isfinite(val):
                # n >= MIN_PIX but correlation undefined (zero variance)
                ax.text(ti, pi, f'n={n}\n(no var)',
                        ha='center', va='center',
                        fontsize=9, color='#aaaaaa',
                        fontproperties=fp_reg)
                continue

            text_color = 'white' if abs(val) > vmax * 0.7 else 'black'
            ax.text(ti, pi, f'{val:.2f}\n(n={n})',
                    ha='center', va='center',
                    fontsize=10, color=text_color,
                    fontproperties=fp_reg)

            if sig:
                rect = mpatches.FancyBboxPatch(
                    (ti - 0.47, pi - 0.47), 0.95, 0.95,
                    boxstyle='square,pad=0',
                    linewidth=2, edgecolor='#8b1889', facecolor='none',
                    zorder=5
                )
                ax.add_patch(rect)

    print(total_n)

    # x-axis: temp labels only on bottom panel
    ax.set_xticks(range(n_temp))
    if panel is panels[-1]:
        ax.set_xticklabels(temp_labels, fontsize=11, rotation=20, ha='right')
    else:
        ax.set_xticklabels([])

    # y-axis: precip labels
    ax.set_yticks(range(n_precip))
    ax.set_yticklabels(precip_labels, fontsize=11)

    # White grid lines
    for x in np.arange(-0.5, n_temp, 1):
        ax.axvline(x, color='white', linewidth=2)
    for y in np.arange(-0.5, n_precip, 1):
        ax.axhline(y, color='white', linewidth=2)

    # Panel label on y-axis
    ax.set_ylabel("Mean Annual Precipitation (mm)", fontsize=12, labelpad=10)
    ax.set_title(panel['letter'], loc="left", fontsize=12)

# Shared x-axis label
axes[-1].set_xlabel('Mean temperature (°C)', fontsize=12,
                    fontproperties=fp_reg)

# Significance legend on top panel
sig_patch = mpatches.Patch(linewidth=2, edgecolor='#8b1889', facecolor='white')
axes[0].legend(
    handles=[sig_patch], labels=['p < 0.05'],
    loc='upper right',
    bbox_to_anchor=(0.62, 1.12),
    bbox_transform=axes[0].transAxes,
    fontsize=12, frameon=False
)

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print(f'\nSaved: {OUT_PATH}')