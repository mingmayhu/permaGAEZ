"""
Frontier × Permafrost Regional Bar Chart
Uses only: numpy, gdal/ogr, rasterio, matplotlib, seaborn
No shapely, no geopandas.

Metrics per region:
  1. Expansion land (km²)                     — Hannah frontier binary
  2. Permafrost area (km²)                    — NIEER probability >= 50%
  3. Expansion land in permafrost (km²)       — frontier & permafrost
  4. Current agriculture in permafrost (km²)  — cropland >5% & permafrost
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
from osgeo import gdal, ogr
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────────
FRONTIER_FILE   = '/Users/ming-mayhu/Downloads/Crop_Suitability_3Method/outputs/frontier_binary.tif'
CROPLAND_FILE   = '/Users/ming-mayhu/Downloads/Global_cropland_3km_2019.tif'
PERMAFROST_FILE = '/Users/ming-mayhu/Desktop/NIEER_permafrost_dataset_released/NIEER_permafrost_dataset_released/NIEER_Probability.tif'
WORLD_MAP       = '/Applications/QGIS-LTR.app/Contents/Resources/resources/data/world_map.gpkg'
WORLD_LAYER     = 'countries'
OUT_DIR         = './results'
OUT_FIG         = os.path.join(OUT_DIR, 'frontier_permafrost_bars.png')
PERM_THRESH     = 10    # permafrost probability threshold (%)
CROP_THRESH     = 5     # cropland percentage threshold
DPI             = 300
os.makedirs(OUT_DIR, exist_ok=True)

# ── Font setup ─────────────────────────────────────────────────────────────────
REG_PATH  = '/System/Library/Fonts/Helvetica.ttc'
fp_reg    = FontProperties(fname=REG_PATH,  size=14)
fp_tick   = FontProperties(fname=REG_PATH,  size=14)

sns.set_theme(style='ticks', rc={
    'font.family':     'sans-serif',
    'font.sans-serif': ['Helvetica'],
    'axes.edgecolor':  '#000000',
    'axes.linewidth':  0.8,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
})

# ── Helper: reproject any raster to match frontier grid ───────────────────────
def reproject_to_ref(src_path, ref_transform, ref_crs, ref_height, ref_width,
                     resampling=Resampling.bilinear):
    with rasterio.open(src_path) as src:
        print(f'  Source CRS: {src.crs}, shape: {src.height}x{src.width}')
        out = np.full((ref_height, ref_width), np.nan, dtype=np.float32)
        reproject(
            source        = rasterio.band(src, 1),
            destination   = out,
            src_transform = src.transform,
            src_crs       = src.crs,
            dst_transform = ref_transform,
            dst_crs       = ref_crs,
            resampling    = resampling,
            src_nodata    = src.nodata,
            dst_nodata    = np.nan,
        )
    return out

# ── Step 1: Load frontier raster (reference grid) ─────────────────────────────
print('Step 1: Loading frontier raster...')
with rasterio.open(FRONTIER_FILE) as src:
    ref_crs       = src.crs
    ref_transform = src.transform
    ref_width     = src.width
    ref_height    = src.height
    frontier_arr  = src.read(1).astype(np.float32)
    print(f'  Grid: {ref_width} x {ref_height}, CRS: {ref_crs}')

# ── Step 2: Reproject and threshold permafrost ────────────────────────────────
print('Step 2: Aligning permafrost raster...')
perm_aligned = reproject_to_ref(PERMAFROST_FILE, ref_transform, ref_crs,
                                 ref_height, ref_width)
print(f'  Non-nan cells: {np.isfinite(perm_aligned).sum():,}')
print(f'  Value range: {np.nanmin(perm_aligned):.1f} – {np.nanmax(perm_aligned):.1f}')
perm_mask = np.where(np.isfinite(perm_aligned), perm_aligned >= PERM_THRESH, False)
print(f'  Permafrost cells (>={PERM_THRESH}%): {perm_mask.sum():,}')

# ── Step 3: Reproject and threshold cropland ──────────────────────────────────
print('Step 3: Aligning cropland raster...')
# Use bilinear for reprojection (continuous percentage values)
crop_aligned = reproject_to_ref(CROPLAND_FILE, ref_transform, ref_crs,
                                 ref_height, ref_width, Resampling.bilinear)
print(f'  Non-nan cells: {np.isfinite(crop_aligned).sum():,}')
print(f'  Value range: {np.nanmin(crop_aligned):.1f} – {np.nanmax(crop_aligned):.1f}')
crop_mask = np.where(np.isfinite(crop_aligned), crop_aligned > CROP_THRESH, False)
print(f'  Cropland cells (>{CROP_THRESH}%): {crop_mask.sum():,}')

# ── Step 4: Pixel area grid (cos-corrected km²) ───────────────────────────────
print('Step 4: Computing pixel areas...')
res_lat  = abs(ref_transform.e)
res_lon  = abs(ref_transform.a)
lats     = ref_transform.f + (np.arange(ref_height) + 0.5) * ref_transform.e
km2_row  = res_lon * 111.32 * res_lat * 111.32 * np.abs(np.cos(np.radians(lats)))
km2_grid = np.tile(km2_row[:, np.newaxis], (1, ref_width))

# ── Step 5: Build region masks using OGR ──────────────────────────────────────
print('Step 5: Building region masks...')

def clip_geom_to_bbox(geom, xmin, ymin, xmax, ymax):
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(xmin, ymin); ring.AddPoint(xmax, ymin)
    ring.AddPoint(xmax, ymax); ring.AddPoint(xmin, ymax)
    ring.AddPoint(xmin, ymin)
    bbox = ogr.Geometry(ogr.wkbPolygon)
    bbox.AddGeometry(ring)
    return geom.Intersection(bbox)

def bbox_geojson(xmin, ymin, xmax, ymax):
    return {"type": "Polygon", "coordinates": [[[xmin,ymin],[xmax,ymin],
            [xmax,ymax],[xmin,ymax],[xmin,ymin]]]}

def country_geojsons(world_path, layer_name, name_field, name_list, clip_bbox=None):
    ds  = ogr.Open(world_path)
    lyr = ds.GetLayerByName(layer_name)
    out = []
    lyr.ResetReading()
    for feat in lyr:
        if feat.GetField(name_field) in name_list:
            geom = feat.GetGeometryRef()
            if geom is None: continue
            if clip_bbox is not None:
                geom = clip_geom_to_bbox(geom, *clip_bbox)
            if geom and not geom.IsEmpty():
                out.append(json.loads(geom.ExportToJson()))
    ds = None
    return out

def geojsons_to_mask(geojson_list, height, width, transform):
    if not geojson_list:
        return np.zeros((height, width), dtype=bool)
    shapes = [(gj, 1) for gj in geojson_list]
    burned = rasterize(shapes, out_shape=(height, width),
                       transform=transform, fill=0,
                       dtype=np.uint8, all_touched=True)
    return burned.astype(bool)

# Detect name field
ds_world  = ogr.Open(WORLD_MAP)
lyr       = ds_world.GetLayerByName(WORLD_LAYER)
feat0     = lyr.GetNextFeature()
fields    = [feat0.GetFieldDefnRef(i).GetName() for i in range(feat0.GetFieldCount())]
name_field = next((f for f in ['NAME','name','ADMIN','admin','NAME_EN','sovereignt','COUNTRY']
                   if f in fields), fields[1])
lyr.ResetReading()
all_names = [feat.GetField(name_field) for feat in lyr]
ds_world  = None

print(f'  Name field: {name_field}')
print(f'  Russia:   {[n for n in all_names if n and "Russia" in n]}')
print(f'  US:       {[n for n in all_names if n and "United States" in n]}')
print(f'  Canada:   {[n for n in all_names if n and "Canada" in n]}')
print(f'  Mongolia: {[n for n in all_names if n and "Mongolia" in n]}')

# Adjust if printed names differ from these
RUSSIA_NAMES   = ['Russia', 'Russian Federation']
US_NAMES       = ['United States of America', 'United States']
CANADA_NAMES   = ['Canada']
MONGOLIA_NAMES = ['Mongolia']

print('  Rasterizing...')
alaska_mask   = geojsons_to_mask(
    country_geojsons(WORLD_MAP, WORLD_LAYER, name_field, US_NAMES,
                     clip_bbox=(-180, 54, -130, 72)),
    ref_height, ref_width, ref_transform)
canada_mask   = geojsons_to_mask(
    country_geojsons(WORLD_MAP, WORLD_LAYER, name_field, CANADA_NAMES),
    ref_height, ref_width, ref_transform)
siberia_mask  = geojsons_to_mask(
    country_geojsons(WORLD_MAP, WORLD_LAYER, name_field, RUSSIA_NAMES,
                     clip_bbox=(60, 50, 180, 80)),
    ref_height, ref_width, ref_transform)
mongolia_mask = geojsons_to_mask(
    country_geojsons(WORLD_MAP, WORLD_LAYER, name_field, MONGOLIA_NAMES),
    ref_height, ref_width, ref_transform)
qtp_mask      = geojsons_to_mask(
    [bbox_geojson(73, 26, 104, 40)],
    ref_height, ref_width, ref_transform)

named_mask = alaska_mask | canada_mask | siberia_mask | qtp_mask | mongolia_mask
# Restrict "Other" to northern latitudes (>45°N) to avoid swamping the chart
lat_grid   = np.tile(lats[:, np.newaxis], (1, ref_width))
north_mask = lat_grid > 45
other_mask = ~named_mask & north_mask

region_masks = {
    'Alaska':   alaska_mask,
    'Canada':   canada_mask,
    'Siberia':  siberia_mask,
    'Qinghai-Tibet Plateau':      qtp_mask,
    'Mongolia': mongolia_mask,
    'Other\n(>45°N)': other_mask,
}
for r, m in region_masks.items():
    print(f'  {r.replace(chr(10)," ")}: {m.sum():,} cells')

# ── Step 6: Compute metrics ────────────────────────────────────────────────────
print('\nStep 6: Computing metrics...')
frontier_bool = frontier_arr == 1

METRIC_KEYS = [
    'Agricultural frontier',
    'Permafrost area',
    'Agricultural frontier on permafrost',
]

metrics = {}
for region, rmask in region_masks.items():
    exp_in_region  = frontier_bool & rmask
    perm_in_region = perm_mask     & rmask
    exp_in_perm    = frontier_bool & perm_mask & rmask
    cur_in_perm    = crop_mask     & perm_mask & rmask
    metrics[region] = {
        METRIC_KEYS[0]: (exp_in_region  * km2_grid).sum(),
        METRIC_KEYS[1]: (perm_in_region * km2_grid).sum(),
        METRIC_KEYS[2]: (exp_in_perm    * km2_grid).sum(),
    }
    k0, k1, k2 = METRIC_KEYS
    print(f'  {region.replace(chr(10)," "):20s} '
          f'exp={metrics[region][k0]/1e6:.2f}M  '
          f'perm={metrics[region][k1]/1e6:.2f}M  '
          f'exp∩perm={metrics[region][k2]/1e6:.2f}M  ')

# ── Step 7: Plot ───────────────────────────────────────────────────────────────
print('\nStep 7: Plotting...')
COLORS  = ['#fcc50d', '#08306b', '#81a8ef']
REGIONS = list(metrics.keys())
n_r     = len(REGIONS)
n_m     = len(METRIC_KEYS)
x       = np.arange(n_r)
bw      = 0.18
offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * bw

fig, ax = plt.subplots(figsize=(13, 5.5))
fig.patch.set_facecolor('white')

for i, (key, color) in enumerate(zip(METRIC_KEYS, COLORS)):
    vals = [metrics[r][key] / 1e6 for r in REGIONS]
    ax.bar(x + offsets[i], vals, width=bw, color=color,
           alpha=0.9, edgecolor='white', linewidth=0.4,
           label=key, zorder=3)

ax.set_xticks(x)
ax.set_xticklabels([r.replace('\n', '\n') for r in REGIONS])
for lbl in ax.get_xticklabels():
    lbl.set_font_properties(fp_reg)
for lbl in ax.get_yticklabels():
    lbl.set_font_properties(fp_tick)

ax.set_ylabel('Area (million km²)')
ax.yaxis.label.set_font_properties(fp_reg)
ax.tick_params(which='major', length=4)
ax.set_xlim(-0.55, n_r - 0.45)
ax.yaxis.grid(True, color='#dddddd', linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

handles = [mpatches.Patch(color=COLORS[i], alpha=0.9,
           label=METRIC_KEYS[i].replace('\n', ' '))
           for i in range(n_m)]
leg = fig.legend(handles=handles, loc='upper center',
                 bbox_to_anchor=(0.5, 1.1),
                 ncol=n_m, frameon=False,
                 handlelength=1.5, handletextpad=0.5, columnspacing=1.2)
for text in leg.get_texts():
    text.set_font_properties(fp_reg)

# sns.despine(ax=ax)
fig.tight_layout()
fig.savefig(OUT_FIG, dpi=DPI, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f'\nSaved -> {OUT_FIG}')