"""
PCF Risk Exposure by Land Cover Type — chunked processing
Processes the 30 m land cover raster in row-blocks. Risk and area rasters
are at 1 km; each 30 m pixel looks up its parent 1 km cell by coordinate
and takes a proportional share of that cell's A_adj:

    exposed_30m = A_adj_1km × (pixel_area_30m / pixel_area_1km)

This ensures A_adj is not double-counted across the ~1111 30 m pixels
that fall within each 1 km cell.
"""

import numpy as np
from osgeo import gdal
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import matplotlib.font_manager as fm
import os

gdal.UseExceptions()

# ── Config ────────────────────────────────────────────────────────────────────

RISK_PATH      = '/Users/ming-mayhu/Desktop/毕业论文/数据/thermokarst data/Potential risk/RTS_Risk_map.tif'
AREA_PATH      = '/Users/ming-mayhu/Desktop/毕业论文/数据/thermokarst data/Potential risk/RTS_area_adj.tif'
LANDCOVER_PATH = '/Users/ming-mayhu/Desktop/毕业论文/数据/agricultural land/tplcd (li et al.)/2020-2023/TPBlandcover_2023.tif'

OUT_CSV    = '/Users/ming-mayhu/Desktop/pcf_risk_by_landcover.csv'
OUT_FIG    = '/Users/ming-mayhu/Desktop/pcf_risk_by_landcover.pdf'

HELVETICA_BOLD = '/Users/ming-mayhu/Library/Fonts/HelveticaNeueLTPro-Bd.otf'

CHUNK_ROWS = 500

# ── Land cover metadata ───────────────────────────────────────────────────────

LC_LABELS = {
    1:  'Cropland',
    2:  'Forest',
    3:  'Shrubland',
    4:  'Alpine steppe',
    5:  'Alpine meadow',
    6:  'Water body',
    7:  'Bare land',
    8:  'Impervious surfaces',
    9:  'Wetlands',
    10: 'Snow/ice',
}

RISK_ORDER = ['None (0%)', 'Low (0-50%)', 'Medium (50-80%)', 'High (>80%)']
RISK_COLOURS = {
    'None (0%)':       '#d1e5f0',
    'Low (0-50%)':     '#92c5de',
    'Medium (50-80%)': '#f4a582',
    'High (>80%)':     '#ca0020',
}

LC_VALS = sorted(LC_LABELS.keys())

# ── Open datasets ─────────────────────────────────────────────────────────────

print('Opening datasets...')
ds_lc   = gdal.Open(LANDCOVER_PATH); assert ds_lc
ds_risk = gdal.Open(RISK_PATH);      assert ds_risk
ds_area = gdal.Open(AREA_PATH);      assert ds_area

gt_lc   = ds_lc.GetGeoTransform()
gt_1km  = ds_risk.GetGeoTransform()

nrows_lc   = ds_lc.RasterYSize
ncols_lc   = ds_lc.RasterXSize
nrows_1km  = ds_risk.RasterYSize
ncols_1km  = ds_risk.RasterXSize

nd_lc   = ds_lc.GetRasterBand(1).GetNoDataValue()
nd_risk = ds_risk.GetRasterBand(1).GetNoDataValue()
nd_area = ds_area.GetRasterBand(1).GetNoDataValue()

print(f'Land cover grid:  {nrows_lc} x {ncols_lc}')
print(f'Risk/area grid:   {nrows_1km} x {ncols_1km}')

# Read 1 km arrays fully (small enough)
print('Loading 1 km risk and area arrays...')
risk_1km = ds_risk.GetRasterBand(1).ReadAsArray().astype(np.float32)
area_1km = ds_area.GetRasterBand(1).ReadAsArray().astype(np.float32)

if nd_risk is not None: risk_1km[risk_1km == nd_risk] = np.nan
if nd_area is not None: area_1km[area_1km == nd_area] = np.nan
risk_1km[risk_1km < -1e10] = np.nan
area_1km[area_1km < -1e10] = np.nan
area_km2_1km = area_1km / 1_000_000

# ── Accumulators ─────────────────────────────────────────────────────────────
# [lc_idx, risk_class_idx]
acc_lc_area = np.zeros((len(LC_VALS), 4), dtype=np.float64)  # 30m pixel area
acc_exposed = np.zeros((len(LC_VALS), 4), dtype=np.float64)  # proportional A_adj

lc_val_to_idx = {v: i for i, v in enumerate(LC_VALS)}

def risk_class_idx(r):
    out = np.full(r.shape, -1, dtype=np.int8)
    out[r == 0]                = 0
    out[(r > 0) & (r < 50)]   = 1
    out[(r >= 50) & (r <= 80)] = 2
    out[r > 80]                = 3
    return out

# ── Chunked processing ────────────────────────────────────────────────────────

print('Processing chunks...')
for row_start in range(0, nrows_lc, CHUNK_ROWS):
    row_end = min(row_start + CHUNK_ROWS, nrows_lc)
    n_rows  = row_end - row_start

    if row_start % 5000 == 0:
        print(f'  Row {row_start:,} / {nrows_lc:,}')

    # Read LC chunk
    lc_chunk = ds_lc.GetRasterBand(1).ReadAsArray(
        0, row_start, ncols_lc, n_rows).astype(np.float32)
    if nd_lc is not None:
        lc_chunk[lc_chunk == nd_lc] = np.nan
    lc_chunk[lc_chunk < 0] = np.nan

    # Geographic centres of this chunk
    col_idx = np.arange(ncols_lc)
    row_idx = np.arange(row_start, row_end)
    cols_2d, rows_2d = np.meshgrid(col_idx, row_idx)

    x = gt_lc[0] + (cols_2d + 0.5) * gt_lc[1]
    y = gt_lc[3] + (rows_2d + 0.5) * gt_lc[5]

    # Parent 1 km cell indices
    c1k = np.clip(((x - gt_1km[0]) / gt_1km[1]).astype(int), 0, ncols_1km - 1)
    r1k = np.clip(((y - gt_1km[3]) / gt_1km[5]).astype(int), 0, nrows_1km - 1)

    risk_chunk     = risk_1km    [r1k, c1k]
    area_km2_chunk = area_km2_1km[r1k, c1k]

    # 30 m pixel area (latitude-corrected, km²)
    pix_area_30m = (
        (abs(gt_lc[5]) * 111.132) *
        (abs(gt_lc[1]) * 111.132 * np.cos(np.radians(y)))
    )

    # Parent 1 km pixel area (km²)
    pix_area_1km = (
        (abs(gt_1km[5]) * 111.132) *
        (abs(gt_1km[1]) * 111.132 * np.cos(np.radians(y)))
    )

    # Proportional share of A_adj for this 30 m pixel
    safe_1km = np.where(pix_area_1km > 0, pix_area_1km, np.nan)
    exposed_chunk = area_km2_chunk * (pix_area_30m / safe_1km)

    valid  = np.isfinite(lc_chunk) & np.isfinite(risk_chunk) & np.isfinite(area_km2_chunk)
    rc_idx = risk_class_idx(risk_chunk)

    for lc_val, lc_i in lc_val_to_idx.items():
        lc_mask = valid & (np.round(lc_chunk).astype(int) == lc_val)
        if not lc_mask.any():
            continue
        for rc_i in range(4):
            m = lc_mask & (rc_idx == rc_i)
            if m.any():
                acc_lc_area[lc_i, rc_i] += pix_area_30m[m].sum()
                acc_exposed[lc_i, rc_i] += exposed_chunk[m].sum()

print('Done processing.')

# ── Build table ───────────────────────────────────────────────────────────────

rows = []
for lc_i, lc_val in enumerate(LC_VALS):
    lc_name        = LC_LABELS[lc_val]
    lc_total_area  = acc_lc_area[lc_i].sum()
    lc_exposed_tot = acc_exposed[lc_i].sum()

    for rc_i, rc_name in enumerate(RISK_ORDER):
        exposed      = acc_exposed[lc_i, rc_i]
        pct_of_lc    = 100 * exposed / lc_total_area  if lc_total_area  > 0 else 0.0
        pct_of_exp   = 100 * exposed / lc_exposed_tot if lc_exposed_tot > 0 else 0.0

        rows.append({
            'Land cover':                lc_name,
            'LC value':                  lc_val,
            'Risk class':                rc_name,
            'Total LC area (km2)':       round(lc_total_area, 2),
            'Physically exposed (km2)':  round(exposed, 4),
            'Exposed (% of LC area)':    round(pct_of_lc, 2),
            'Exposed (% of LC exposed)': round(pct_of_exp, 2),
        })

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f'\nTable saved to: {OUT_CSV}')
print(df.to_string(index=False))

# ── Pivot for plotting ────────────────────────────────────────────────────────

# Panel A: absolute physically exposed km²
pivot = df[df['Risk class'] != 'None (0%)'].pivot_table(
    index='Land cover', columns='Risk class',
    values='Physically exposed (km2)', aggfunc='sum'
)[[r for r in RISK_ORDER if r != 'None (0%)']]

pivot['_total'] = pivot.sum(axis=1)
pivot = pivot.sort_values('_total', ascending=True).drop(columns='_total')
lc_order = pivot.index.tolist()

# ── Font setup ────────────────────────────────────────────────────────────────

if os.path.exists(HELVETICA_BOLD):
    prop_bold = fm.FontProperties(fname=HELVETICA_BOLD)
    matplotlib.rcParams['font.family'] = prop_bold.get_name()

import seaborn as sns
sns.set_style('ticks')
plt.rcParams.update({'font.size': 9})

# ── Figure ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={'wspace': 0.55})

risk_plot_order = ['Low (0-50%)', 'Medium (50-80%)', 'High (>80%)']

# Panel A: % of total LC area (stacked, excludes None)
ax = axes[0]
lefts = np.zeros(len(lc_order))
for rc in risk_plot_order:
    vals = pivot[rc].values
    ax.barh(lc_order, vals, left=lefts,
            color=RISK_COLOURS[rc], edgecolor='white', linewidth=0.4,
            label=rc, height=0.65)
    lefts += vals

ax.set_xlabel('Physically exposed area (km²)', fontsize=9)
ax.set_title('(a)  Exposed area as % of land cover type',
             fontsize=9, loc='left', pad=6)
ax.tick_params(labelsize=8)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
sns.despine(ax=ax, top=True, right=True)

# Panel B: risk class composition of exposed area
pivot2 = df[df['Risk class'] != 'None (0%)'].pivot_table(
    index='Land cover', columns='Risk class',
    values='Exposed (% of LC exposed)', aggfunc='sum'
)[[r for r in RISK_ORDER if r != 'None (0%)']]
pivot2 = pivot2.loc[lc_order]

ax2 = axes[1]
lefts2 = np.zeros(len(lc_order))
for rc in risk_plot_order:
    vals = pivot2[rc].values
    ax2.barh(lc_order, vals, left=lefts2,
             color=RISK_COLOURS[rc], edgecolor='white', linewidth=0.4,
             label=rc, height=0.65)
    lefts2 += vals

ax2.set_xlabel('Risk class composition (% of exposed area)', fontsize=9)
ax2.set_xlim(0, 100)
ax2.set_title('(b)  Risk class breakdown of exposed area',
              fontsize=9, loc='left', pad=6)
ax2.tick_params(labelsize=8)
ax2.set_yticklabels([])
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
sns.despine(ax=ax2, top=True, right=True)

legend_handles = [
    Patch(facecolor=RISK_COLOURS[rc], edgecolor='#555', linewidth=0.5, label=rc)
    for rc in risk_plot_order
]
fig.legend(handles=legend_handles,
           title='PCF risk class', title_fontsize=8, fontsize=8,
           loc='lower center', ncol=3,
           bbox_to_anchor=(0.5, -0.06), frameon=False)

plt.savefig(OUT_FIG, dpi=300, bbox_inches='tight')
print(f'\nFigure saved to: {OUT_FIG}')
plt.show()