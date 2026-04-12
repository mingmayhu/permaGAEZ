"""
PyAEZ Pipeline — Modules 1, 2, 4, 5
Outer loop: crops
Inner loop: years (1979–2018)

Module 1 outputs (tclimate, tzone) are computed once from the multi-year
average and reused across all crops and years.
Per-year Module 1 indicators are also computed once and cached on disk,
then read back by Modules 2, 4, and 5.
"""

# =============================================================================
# CONFIGURATION
# =============================================================================

WORK_DIR   = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
## TODO: make avg_period based on the year in the loop
YEARS      = list(range(1979, 2019))

# Geographic extents
LAT_MIN    = 35.921391739
LAT_MAX    = 39.721391739
MASK_VALUE = 0

# LGP water balance parameters
SA = 100.
D  = 1.

# Set to True to run the "no permafrost thaw" counterfactual
NO_THAW_BASELINE_RUN = False
RUN_TAG = '_nothaw' if NO_THAW_BASELINE_RUN else ''

# --- Crops -------------------------------------------------------------------
# Add or remove dicts to change which crops are processed.
# Each dict must have:
#   crop_name       : name passed to readCropandCropCycleParameters
#   soil_rain_excel : path to soil reduction xlsx (rainfed)
#   terrain_excel   : path to terrain reduction xlsx
#   no_t_climate    : list of thermal climate classes to screen out
# -----------------------------------------------------------------------------
CROPS = [
    # {
    #     'crop_name'      : 'winter_barley_59',
    #     'soil_rain_excel': r'./data_input/soil_inputs/barley_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 1',
    #     'no_t_climate'   : [1, 2, 9, 10, 11, 12],
    # },
    # {
    #     'crop_name'      : 'winter_barley_60',
    #     'soil_rain_excel': r'./data_input/soil_inputs/barley_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 1',
    #     'no_t_climate'   : [1, 2, 9, 10, 11, 12],
    # },
    # {
    #     'crop_name'      : 'winter_barley_61',
    #     'soil_rain_excel': r'./data_input/soil_inputs/barley_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 1',
    #     'no_t_climate'   : [1, 2, 9, 10, 11, 12],
    # },
    # {
    #     'crop_name'      : 'winter_barley_62',
    #     'soil_rain_excel': r'./data_input/soil_inputs/barley_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 1',
    #     'no_t_climate'   : [1, 2, 9, 10, 11, 12],
    # },
    #     {
    #     'crop_name'      : 'spring_barley_63',
    #     'soil_rain_excel': r'./data_input/soil_inputs/barley_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 1',
    #     'no_t_climate'   : [1, 2, 12],
    # },
    #         {
    #     'crop_name'      : 'spring_barley_64',
    #     'soil_rain_excel': r'./data_input/soil_inputs/barley_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 1',
    #     'no_t_climate'   : [1, 2, 12],
    # },
    #         {
    #     'crop_name'      : 'spring_barley_65',
    #     'soil_rain_excel': r'./data_input/soil_inputs/barley_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 1',
    #     'no_t_climate'   : [1, 2, 12],
    # },
    #         {
    #     'crop_name'      : 'spring_barley_66',
    #     'soil_rain_excel': r'./data_input/soil_inputs/barley_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 1',
    #     'no_t_climate'   : [1, 2, 12],
    # }
    #     {
    #     'crop_name'      : 'winter_wheat_1',
    #     'soil_rain_excel': r'./data_input/soil_inputs/wheat_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 1',
    #     'no_t_climate'   : [1, 2, 9, 10, 11, 12],
    # },
    #     {
    #     'crop_name'      : 'winter_wheat_2',
    #     'soil_rain_excel': r'./data_input/soil_inputs/wheat_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 1',
    #     'no_t_climate'   : [1, 2, 9, 10, 11, 12],
    # },
    #     {
    #     'crop_name'      : 'winter_wheat_3',
    #     'soil_rain_excel': r'./data_input/soil_inputs/wheat_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 1',
    #     'no_t_climate'   : [1, 2, 9, 10, 11, 12],
    # },
    #     {
    #     'crop_name'      : 'winter_wheat_4',
    #     'soil_rain_excel': r'./data_input/soil_inputs/wheat_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 1',
    #     'no_t_climate'   : [1, 2, 9, 10, 11, 12],
    # },
    #     {
    #     'crop_name'      : 'spring_wheat_5',
    #     'soil_rain_excel': r'./data_input/soil_inputs/wheat_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 1',
    #     'no_t_climate'   : [1, 2, 12],
    # },
    #         {
    #     'crop_name'      : 'spring_wheat_6',
    #     'soil_rain_excel': r'./data_input/soil_inputs/wheat_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 1',
    #     'no_t_climate'   : [1, 2, 12],
    # },
    #         {
    #     'crop_name'      : 'spring_wheat_7',
    #     'soil_rain_excel': r'./data_input/soil_inputs/wheat_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 1',
    #     'no_t_climate'   : [1, 2, 12],
    # },
    #         {
    #     'crop_name'      : 'spring_wheat_8',
    #     'soil_rain_excel': r'./data_input/soil_inputs/wheat_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 1',
    #     'no_t_climate'   : [1, 2, 12],
    # },
    #         {
    #     'crop_name'      : 'spring_wheat_9',
    #     'soil_rain_excel': r'./data_input/soil_inputs/wheat_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 1',
    #     'no_t_climate'   : [1, 2, 12],
    # },
    #             {
    #     'crop_name'      : 'silage_maize_53',
    #     'soil_rain_excel': r'./data_input/soil_inputs/silage_maize_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 2',
    #     'no_t_climate'   : [9, 10, 11, 12],
    # },
    #                 {
    #     'crop_name'      : 'silage_maize_54',
    #     'soil_rain_excel': r'./data_input/soil_inputs/silage_maize_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 2',
    #     'no_t_climate'   : [9, 10, 11, 12],
    # },
    #                 {
    #     'crop_name'      : 'silage_maize_55',
    #     'soil_rain_excel': r'./data_input/soil_inputs/silage_maize_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 2',
    #     'no_t_climate'   : [9, 10, 11, 12],
    # },
    #                 {
    #     'crop_name'      : 'silage_maize_56',
    #     'soil_rain_excel': r'./data_input/soil_inputs/silage_maize_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 2',
    #     'no_t_climate'   : [1, 2, 9, 10, 11, 12],
    # },
    #                     {
    #     'crop_name'      : 'silage_maize_57',
    #     'soil_rain_excel': r'./data_input/soil_inputs/silage_maize_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 2',
    #     'no_t_climate'   : [1, 2, 9, 10, 11, 12],
    # },
    #                     {
    #     'crop_name'      : 'silage_maize_58',
    #     'soil_rain_excel': r'./data_input/soil_inputs/silage_maize_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 2',
    #     'no_t_climate'   : [1, 2, 9, 10, 11, 12],
    # },
    # {
    #     'crop_name'      : 'white_potato_135',
    #     'soil_rain_excel': r'./data_input/soil_inputs/white_potato_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 2',
    #     'no_t_climate'   : [12],
    # },
    #     {
    #     'crop_name'      : 'white_potato_136',
    #     'soil_rain_excel': r'./data_input/soil_inputs/white_potato_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 2',
    #     'no_t_climate'   : [12],
    # },
    #     {
    #     'crop_name'      : 'white_potato_137',
    #     'soil_rain_excel': r'./data_input/soil_inputs/white_potato_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 2',
    #     'no_t_climate'   : [12],
    # },
    #         {
    #     'crop_name'      : 'white_potato_138',
    #     'soil_rain_excel': r'./data_input/soil_inputs/white_potato_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 2',
    #     'no_t_climate'   : [1, 12],
    # },
    #             {
    #     'crop_name'      : 'white_potato_139',
    #     'soil_rain_excel': r'./data_input/soil_inputs/white_potato_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 2',
    #     'no_t_climate'   : [1, 12],
    # },
    #             {
    #     'crop_name'      : 'white_potato_140',
    #     'soil_rain_excel': r'./data_input/soil_inputs/white_potato_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 2',
    #     'no_t_climate'   : [1, 12],
    # },
    #             {
    #     'crop_name'      : 'white_potato_141',
    #     'soil_rain_excel': r'./data_input/soil_inputs/white_potato_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 2',
    #     'no_t_climate'   : [1, 12],
    # },
    # {
    #     'crop_name'      : 'spring_oat_128',
    #     'soil_rain_excel': r'./data_input/soil_inputs/oat_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 1',
    #     'no_t_climate'   : [1, 2, 12],
    # },
    #     {
    #     'crop_name'      : 'spring_oat_129',
    #     'soil_rain_excel': r'./data_input/soil_inputs/oat_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 1',
    #     'no_t_climate'   : [1, 2, 12],
    # },
    #     {
    #     'crop_name'      : 'spring_oat_130',
    #     'soil_rain_excel': r'./data_input/soil_inputs/oat_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 1',
    #     'no_t_climate'   : [1, 2, 12],
    # },
    #         {
    #     'crop_name'      : 'dry_pea_189',
    #     'soil_rain_excel': r'./data_input/soil_inputs/dry_pea_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 2',
    #     'no_t_climate'   : [1, 9, 10, 11, 12],
    # },
    #             {
    #     'crop_name'      : 'dry_pea_190',
    #     'soil_rain_excel': r'./data_input/soil_inputs/dry_pea_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 2',
    #     'no_t_climate'   : [1, 9, 10, 11, 12],
    # },
    #             {
    #     'crop_name'      : 'dry_pea_191',
    #     'soil_rain_excel': r'./data_input/soil_inputs/dry_pea_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 2',
    #     'no_t_climate'   : [1, 9, 10, 11, 12],
    # },
    #                 {
    #     'crop_name'      : 'winter_rape_216',
    #     'soil_rain_excel': r'./data_input/soil_inputs/rapeseed_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 2',
    #     'no_t_climate'   : [1, 2, 9, 10, 11, 12],
    # },
    #                     {
    #     'crop_name'      : 'winter_rape_217',
    #     'soil_rain_excel': r'./data_input/soil_inputs/rapeseed_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 2',
    #     'no_t_climate'   : [1, 2, 9, 10, 11, 12],
    # },
    #                     {
    #     'crop_name'      : 'winter_rape_218',
    #     'soil_rain_excel': r'./data_input/soil_inputs/rapeseed_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 2',
    #     'no_t_climate'   : [1, 2, 9, 10, 11, 12],
    # },
    #                     {
    #     'crop_name'      : 'winter_rape_219',
    #     'soil_rain_excel': r'./data_input/soil_inputs/rapeseed_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 2',
    #     'no_t_climate'   : [1, 2, 9, 10, 11, 12],
    # },
    #                         {
    #     'crop_name'      : 'spring_rape_220',
    #     'soil_rain_excel': r'./data_input/soil_inputs/rapeseed_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 2',
    #     'no_t_climate'   : [1, 2, 9, 10, 11, 12],
    # },
    #                             {
    #     'crop_name'      : 'spring_rape_221',
    #     'soil_rain_excel': r'./data_input/soil_inputs/rapeseed_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 2',
    #     'no_t_climate'   : [1, 2, 9, 10, 11, 12],
    # },
                                {
        'crop_name'      : 'spring_rape_222',
        'soil_rain_excel': r'./data_input/soil_inputs/rapeseed_soil_reduction.xlsx',
        'terrain_crop_group': 'annuals 2',
        'no_t_climate'   : [1, 2, 9, 10, 11, 12],
    },
    #                             {
    #     'crop_name'      : 'spring_rape_223',
    #     'soil_rain_excel': r'./data_input/soil_inputs/rapeseed_soil_reduction.xlsx',
    #     'terrain_crop_group': 'annuals 2',
    #     'no_t_climate'   : [1, 2, 9, 10, 11, 12],
    # },
]

# --- Shared input paths ------------------------------------------------------
MASK_PATH   = r'./data_input/qilian mask.tif'
BASEPATH    = r'./data_input/qilian mask.tif'
SOIL_MAP    = r'./data_input/soil_inputs/hwsd.tif'
SLOPE_PATH  = r'./data_input/terrain/slope.tif'
SOIL_TOPSOIL = r'./data_input/soil_inputs/qtp_topsoil.xlsx'
SOIL_SUBSOIL = r'./data_input/soil_inputs/qtp_subsoil.xlsx'
SOIL_INPUT_LEVEL = 'L'   # L: Low, I: Intermediate, H: High
TERRAIN_EXCEL = r'./data_input/terrain/terrain_reduction.xlsx' 
CROP_EXCEL = r'./data_input/crop_inputs/input_crop_tsum_parameters.xlsx'
CROP_RULE_EXCEL = r'./data_input/crop_inputs/crop_specific_rule.xlsx'

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

try:
    from osgeo import gdal
except ImportError:
    import gdal

os.chdir(WORK_DIR)
sys.path.insert(0, os.path.dirname(WORK_DIR))

from pyaez import ClimateRegime, CropSimulation, SoilConstraints, TerrainConstraints, UtilitiesCalc

obj_util = UtilitiesCalc.UtilitiesCalc()


# =============================================================================
# HELPERS
# =============================================================================

def make_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def load_climate(year, permafrost_year=None):
    import calendar
    if permafrost_year is None:
        permafrost_year = year
        
    d = f'data_input/climate_yearly/{year}'
    p = f'data_input/permafrost_yearly/{permafrost_year}'  # use permafrost_year here

    data = {
        'max_temp'  : np.load(f'{d}/TempMax.npy'),
        'min_temp'  : np.load(f'{d}/TempMin.npy'),
        'precip'    : np.load(f'{d}/Precip.npy'),
        'rh'        : np.load(f'{d}/RH.npy'),
        'wind'      : np.load(f'{d}/Wind.npy'),
        'rad'       : np.load(f'{d}/Radiation.npy'),
        'eta'       : np.load(f'{p}/ET.npy'),
        'alt'       : np.load(f'{p}/active_layer_depth.npy'),
        'soil_moist': np.load(f'{p}/avail_soil_moisture.npy'),
    }

    if calendar.isleap(year):
        data = {k: np.delete(v, 59, axis=2) if v.ndim == 3 else v 
                for k, v in data.items()}

    return data

# =============================================================================
# MODULE 1 — PER-YEAR INDICATORS
# =============================================================================

def run_module1_year(year, mask, elevation, tclimate, tzone, clim):
    """
    Compute and save all per-year Module 1 indicators.
    Returns dict with lgp, lgpt5, lgpt10, tsum0, tsum10 arrays (needed downstream).
    """
    print(f"  [Module 1 | {year}] Computing per-year indicators …")
    out_dir = f'./data_output/original/module1{RUN_TAG}/{year}'
    make_dirs(out_dir)

    tile_list = ['A9','A8','A7','A6','A5','A4','A3','A2',
                 'A1','B1','B2','B3','B4','B5','B6','B7','B8','B9']

    cr = ClimateRegime.ClimateRegime()
    cr.setStudyAreaMask(mask, MASK_VALUE)
    cr.setLocationTerrainData(LAT_MIN, LAT_MAX, elevation)
    cr.setDailyClimateData(
        clim['min_temp'], clim['max_temp'], clim['precip'],
        clim['rad'], clim['wind'], clim['rh']
    )
    cr.setDailyPermafrostData(clim['eta'])

    # -- Thermal LGPs --
    lgpt0  = cr.getThermalLGP0()
    lgpt5  = cr.getThermalLGP5()
    lgpt10 = cr.getThermalLGP10()

    plt.figure(figsize=(24, 8))
    for i, (arr, lbl) in enumerate(zip([lgpt0, lgpt5, lgpt10],
                                        ['LGPt 0', 'LGPt 5', 'LGPt 10']), start=1):
        plt.subplot(1, 3, i)
        plt.imshow(arr, vmin=0, vmax=366)
        plt.title(lbl); plt.colorbar(shrink=0.8)
    plt.savefig(f'{out_dir}/thermalLGPs.png', bbox_inches='tight', dpi=150)
    plt.close()
    obj_util.saveRaster(MASK_PATH, f'{out_dir}/LGPt0.tif',  lgpt0)
    obj_util.saveRaster(MASK_PATH, f'{out_dir}/LGPt5.tif',  lgpt5)
    obj_util.saveRaster(MASK_PATH, f'{out_dir}/LGPt10.tif', lgpt10)

    # -- Temperature Sums --
    tsum0  = cr.getTemperatureSum0()
    tsum5  = cr.getTemperatureSum5()
    tsum10 = cr.getTemperatureSum10()

    plt.figure(figsize=(24, 8))
    for i, (arr, lbl) in enumerate(zip([tsum0, tsum5, tsum10],
                                        ['T-sum 0', 'T-sum 5', 'T-sum 10']), start=1):
        plt.subplot(1, 3, i)
        plt.imshow(arr, cmap='hot_r', vmin=0, vmax=11000)
        plt.title(lbl); plt.colorbar(shrink=0.8)
    plt.savefig(f'{out_dir}/Tsum.png', bbox_inches='tight', dpi=150)
    plt.close()
    obj_util.saveRaster(MASK_PATH, f'{out_dir}/tsum0.tif',  tsum0)
    obj_util.saveRaster(MASK_PATH, f'{out_dir}/tsum5.tif',  tsum5)
    obj_util.saveRaster(MASK_PATH, f'{out_dir}/tsum10.tif', tsum10)

    # -- Temperature Profile --
    tprofile = cr.getTemperatureProfile()
    fig = plt.figure(figsize=(12, 20))
    for i in range(1, 19):
        plt.subplot(6, 3, i)
        plt.imshow(tprofile[i - 1])
        plt.title(tile_list[i - 1]); plt.colorbar(shrink=0.9)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/Tprofiles.png', bbox_inches='tight', dpi=150)
    plt.close()
    for i in range(18):
        obj_util.saveRaster(MASK_PATH, f'{out_dir}/TProfile_{tile_list[i]}.tif', tprofile[i])

    # -- LGP (water-balance) --
    lgp      = cr.getLGP(Sa=SA, D=D)
    lgp_equv = cr.getLGPEquivalent()

    for arr, fname, title in [
        (lgp,      'LGP New',      'LGP [days]'),
        (lgp_equv, 'LGP_Equv New', 'LGP Equivalent [days]'),
    ]:
        plt.imshow(arr, cmap='viridis', vmin=0, vmax=366)
        plt.title(title); plt.colorbar()
        plt.savefig(f'{out_dir}/{fname}.png', bbox_inches='tight', dpi=150)
        plt.close()
        obj_util.saveRaster(MASK_PATH, f'{out_dir}/{fname}.tif', arr)

    # -- Permafrost Evaluation --
    permafrost_eval = cr.AirFrostIndexandPermafrostEvaluation()
    frost_index = permafrost_eval[0]
    permafrost = permafrost_eval[1]
    print(permafrost_eval)
    obj_util.saveRaster(MASK_PATH, f'{out_dir}/frost_index.tif', frost_index)
    obj_util.saveRaster(MASK_PATH, f'{out_dir}/permafrost.tif', permafrost)

    # -- Multi-Cropping Zone --
    multi_crop        = cr.getMultiCroppingZones(tclimate, lgp, lgpt5, lgpt10, tsum0, tsum10)
    multi_crop_rain   = multi_crop[0]
    plt.imshow(multi_crop_rain, cmap=plt.get_cmap('gist_ncar_r', 9), vmin=-0.2, vmax=8.4)
    plt.title('Multi Cropping Zone - RAINFED'); plt.colorbar()
    plt.savefig(f'{out_dir}/multicrop_rain.png', bbox_inches='tight', dpi=150)
    plt.close()
    obj_util.saveRaster(MASK_PATH, f'{out_dir}/multicrop_rain.tif', multi_crop_rain)

    # -- Fallow Requirement --
    tzone_fallow = cr.TZoneFallowRequirement(tzone)
    plt.imshow(tzone_fallow, cmap=plt.get_cmap('tab10', 7), vmin=-0.5, vmax=6.3)
    plt.title('Fallow Requirement'); plt.colorbar()
    plt.savefig(f'{out_dir}/fallow.png', bbox_inches='tight', dpi=150)
    plt.close()
    obj_util.saveRaster(MASK_PATH, f'{out_dir}/fallow.tif', tzone_fallow)

    print(f"    ✓ Module 1 outputs saved to {out_dir}")
    return {'lgp': lgp, 'lgpt5': lgpt5, 'lgpt10': lgpt10,
            'tsum0': tsum0, 'tsum10': tsum10}


# =============================================================================
# MODULE 2 — CROP SIMULATION
# =============================================================================

def run_module2(year, crop, mask, elevation, clim, tclimate, m1, permafrost_class):
    """
    Run crop simulation for one crop and one year.
    Returns (yield_map_rain, starting_date_rain).
    m1: dict returned by run_module1_year (lgp, lgpt5, lgpt10, tsum0, tsum10)
    """
    print(f"  [Module 2 | {year} | {crop['crop_name']}] Crop simulation …")
    out_dir = f'./data_output/original/module2{RUN_TAG}/{crop["crop_name"]}/{year}'
    make_dirs(out_dir)

    aez = CropSimulation.CropSimulation()
    aez.setStudyAreaMask(mask, MASK_VALUE)
    aez.setLocationTerrainData(LAT_MIN, LAT_MAX, elevation)
    aez.setDailyClimateData(
        clim['min_temp'], clim['max_temp'], clim['precip'],
        clim['rad'], clim['wind'], clim['rh']
    )
    aez.setPermafrostData(clim['alt'], clim['soil_moist'])

    aez.readCropandCropCycleParameters(
        file_path=CROP_EXCEL,
        crop_name=crop['crop_name']
    )
    aez.setSoilWaterParameters(Sa=100 * np.ones(mask.shape), pc=0.5)
    aez.setThermalClimateScreening(tclimate, no_t_climate=crop['no_t_climate'])
    aez.setPermafrostScreening(permafrost_class=permafrost_class)
    aez.setCropSpecificRule(
        file_path=CROP_RULE_EXCEL,
        crop_name=crop['crop_name']
    )
    aez.ImportLGPandLGPT(lgp=m1['lgp'], lgpt5=m1['lgpt5'], lgpt10=m1['lgpt10'])

    # TODO: change this to account for leap years
    aez.simulateCropCycle(start_doy=1, end_doy=365, step_doy=1, leap_year=False)

    yield_map_rain    = aez.getEstimatedYieldRainfed()
    starting_date_rain = aez.getOptimumCycleStartDateRainfed()
    fc1_rain          = aez.getThermalReductionFactor()[0]
    fc2               = aez.getMoistureReductionFactor()

    # -- Plots --
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im0 = axes[0].imshow(yield_map_rain, vmin=0, vmax=np.max(yield_map_rain))
    axes[0].set_title('Rainfed Yield'); plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(starting_date_rain, vmin=0, vmax=366)
    axes[1].set_title('Starting Date Rainfed'); plt.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    plt.savefig(f'{out_dir}/yield_and_start.png', bbox_inches='tight', dpi=150)
    plt.close()

    plt.imshow(fc1_rain, vmin=0, vmax=1); plt.colorbar()
    plt.title('Fc1 Rainfed')
    plt.savefig(f'{out_dir}/fc1_rain.png', bbox_inches='tight', dpi=150)
    plt.close()

    plt.imshow(fc2, vmin=0, vmax=1); plt.colorbar()
    plt.title('Fc2 Moisture Reduction')
    plt.savefig(f'{out_dir}/fc2_rain.png', bbox_inches='tight', dpi=150)
    plt.close()

    # -- Save rasters --
    obj_util.saveRaster(MASK_PATH, f'{out_dir}/yield_map_rain.tif',    yield_map_rain)
    obj_util.saveRaster(MASK_PATH, f'{out_dir}/starting_date_rain.tif', starting_date_rain)
    obj_util.saveRaster(MASK_PATH, f'{out_dir}/fc1_rain.tif',          fc1_rain)
    obj_util.saveRaster(MASK_PATH, f'{out_dir}/fc2_rain.tif',          fc2)

    print(f"    ✓ Module 2 outputs saved to {out_dir}")
    return yield_map_rain, starting_date_rain


# =============================================================================
# MODULE 4 — SOIL CONSTRAINTS
# =============================================================================

def run_module4(year, crop, yield_map_rain):
    """
    Apply soil constraints to rainfed yield.
    Returns yield_map_rain_m4.
    """
    print(f"  [Module 4 | {year} | {crop['crop_name']}] Soil constraints …")
    out_dir = f'./data_output/original/module4{RUN_TAG}/{crop["crop_name"]}/{year}'
    make_dirs(out_dir)

    soil_map = gdal.Open(SOIL_MAP).ReadAsArray()

    # soc_np = np.stack([
    #     np.load(f'./data_input/soc/soc_pct_d{i}.npy') for i in range(1, 8)
    # ], axis=2)

    sc = SoilConstraints.SoilConstraints()
    sc.importSoilReductionSheet(
        rain_sheet_path=crop['soil_rain_excel'],
        irr_sheet_path=crop['soil_rain_excel']
    )
    sc.calculateSoilQualities(
        irr_or_rain='R',
        topsoil_path=SOIL_TOPSOIL,
        subsoil_path=SOIL_SUBSOIL,
    )
    sc.calculateSoilRatings(SOIL_INPUT_LEVEL)

    yield_map_rain_m4 = sc.applySoilConstraints(soil_map, yield_map_rain)
    yield_map_class   = obj_util.classifyFinalYield(yield_map_rain_m4)
    fc4_rain          = sc.getSoilSuitabilityMap()

    # Guard against all-zero yield maps
    if not np.any(yield_map_rain_m4 > 0):
        print(f"    ⚠ No valid yield after soil constraints for {crop['crop_name']} in {year} — saving zero map.")
        obj_util.saveRaster(BASEPATH, f'{out_dir}/yield_soil.tif', yield_map_rain_m4)
        obj_util.saveRaster(BASEPATH, f'{out_dir}/fc4_rain.tif', fc4_rain)
        return yield_map_rain_m4

    # -- Plots --
    fig, axes = plt.subplots(1, 3, figsize=(25, 9))
    for ax, arr, title, vmax in zip(
        axes,
        [yield_map_rain, yield_map_rain_m4, fc4_rain],
        ['Original Rainfed Yield', 'Soil Constrained Yield', 'Soil Fc4 Factor'],
        [2000, 2000, 1]
    ):
        im = ax.imshow(arr, vmin=0, vmax=vmax)
        ax.set_title(title); plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/soil_constraints.png', bbox_inches='tight', dpi=150)
    plt.close()

    # -- Save rasters --
    obj_util.saveRaster(BASEPATH, f'{out_dir}/yield_soil.tif',       yield_map_rain_m4)
    obj_util.saveRaster(BASEPATH, f'{out_dir}/yield_soil_class.tif', yield_map_class)
    obj_util.saveRaster(BASEPATH, f'{out_dir}/fc4_rain.tif',         fc4_rain)

    print(f"    ✓ Module 4 outputs saved to {out_dir}")
    return yield_map_rain_m4


# =============================================================================
# MODULE 5 — TERRAIN CONSTRAINTS
# =============================================================================

def run_module5(year, crop, yield_map_rain_m4, precip):
    """
    Apply terrain constraints to soil-constrained yield.
    Returns yield_map_rain_m5.
    """
    print(f"  [Module 5 | {year} | {crop['crop_name']}] Terrain constraints …")
    out_dir = f'./data_output/original/module5{RUN_TAG}/{crop["crop_name"]}/{year}'
    make_dirs(out_dir)

    slope_map = gdal.Open(SLOPE_PATH).ReadAsArray()

    tc = TerrainConstraints.TerrainConstraints()
    tc.importTerrainReductionSheet(
        irr_file_path=TERRAIN_EXCEL,
        rain_file_path=TERRAIN_EXCEL,
        sheet_name=crop['terrain_crop_group']
    )
    tc.setClimateTerrainData(precip, slope_map)
    tc.calculateFI()

    fi       = tc.getFI()
    yield_m5 = tc.applyTerrainConstraints(yield_map_rain_m4, 'R')
    fc5_rain = tc.getTerrainReductionFactor()

    # -- Plots --
    plt.imshow(fi); plt.colorbar(); plt.title('Fournier Index')
    plt.savefig(f'{out_dir}/fournier_index.png', bbox_inches='tight', dpi=150)
    plt.close()

    vmax = np.max([yield_map_rain_m4, yield_m5])
    fig, axes = plt.subplots(1, 3, figsize=(18, 9))
    for ax, arr, title in zip(
        axes,
        [yield_map_rain_m4, yield_m5, fc5_rain],
        ['Soil Constrained Yield', 'Terrain Constrained Yield', 'Terrain Fc5 Factor']
    ):
        im = ax.imshow(arr, vmax=vmax if 'Yield' in title else 1,
                       vmin=0)
        ax.set_title(title); plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/terrain_constraints.png', bbox_inches='tight', dpi=150)
    plt.close()

    # -- Save rasters --
    obj_util.saveRaster(BASEPATH, f'{out_dir}/yield_terrain.tif', yield_m5)
    obj_util.saveRaster(BASEPATH, f'{out_dir}/fi.tif',            fi)
    obj_util.saveRaster(BASEPATH, f'{out_dir}/fc5_rain.tif',      fc5_rain)

    print(f"    ✓ Module 5 outputs saved to {out_dir}")
    return yield_m5

def final_yield_classification(year, crop, yield_map_rain_m5):
    """
    Classify final yield into discrete classes.
    Returns yield_map_class.
    """
    print(f"  [Final Classification | {year} | {crop['crop_name']}] Classifying final yield …")
    out_dir = f'./data_output/original/final_classification{RUN_TAG}/{crop["crop_name"]}'
    make_dirs(out_dir)

        # Check if any valid yield exists
    if not np.any(yield_map_rain_m5 > 0):
        print(f"    ⚠ No valid yield for {crop['crop_name']} in {year} — saving zero map.")
        yield_map_class = np.zeros(yield_map_rain_m5.shape)
        obj_util.saveRaster(BASEPATH, f'{out_dir}/{year}_final_yield_class{RUN_TAG}.tif', yield_map_class)
        return yield_map_class

    yield_map_class = obj_util.classifyFinalYield(yield_map_rain_m5)

    plt.imshow(yield_map_class, vmin=0, vmax=5, cmap=plt.get_cmap('tab10', 6))
    plt.title('Final Yield Class'); plt.colorbar()
    plt.savefig(f'{out_dir}/{year}_final_yield_class{RUN_TAG}.png', bbox_inches='tight', dpi=150)
    plt.close()

    obj_util.saveRaster(BASEPATH, f'{out_dir}/{year}_final_yield_class{RUN_TAG}.tif', yield_map_class)
    print(f"    ✓ Final classified yield saved to {out_dir}")
    return yield_map_class

def plot_final_classification(crop):
    n_cols = 8
    n_rows = -(-len(YEARS) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
    axes = axes.flatten()
    TARGET_EXTENT = [93.663346096, 102.963346096, 35.821391739, 39.721391739]


    for i, year in enumerate(YEARS):
        tag = '' if (NO_THAW_BASELINE_RUN and year < 1999) else RUN_TAG
        path = f'./data_output/original/final_classification{tag}/{crop}/{year}_final_yield_class.tif'
        ds = gdal.Open(path)
        if ds is None:
            axes[i].set_title(f'{year}\n(missing)', fontsize=8)
            axes[i].axis('off')
            continue
        arr = ds.ReadAsArray().astype(float)
        mask_ds = gdal.Open(MASK_PATH)
        mask = mask_ds.ReadAsArray().astype(float)
        arr[arr < 0] = 0          # treat negatives as class 0
        arr[mask == 0] = np.nan   # mask outside region as NaN
        im = axes[i].imshow(arr, cmap=plt.get_cmap('tab10', 6), vmin=0, vmax=5, extent=TARGET_EXTENT)
        axes[i].set_title(str(year), fontsize=9)
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], shrink=0.8)


    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f'Final Yield Classification {RUN_TAG} — {crop}', fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = f'./data_output/original/final_classification{RUN_TAG}/{crop}/all_years_classification{RUN_TAG}.png'
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  ✓ All-years classification plot saved to {out_path}')

def combine_crop_maps(year, varieties, output_tag=None):
    out_dir = f'./data_output/original/final_classification{RUN_TAG}/{output_tag}'
    os.makedirs(out_dir, exist_ok=True)

    # tag = RUN_TAG
    # tag = '' if (NO_THAW_BASELINE_RUN and year < 1999) else RUN_TAG
    if NO_THAW_BASELINE_RUN and year < 1999:
        print(f"Skipping year {year} for combined map since it's not included in the no-thaw baseline run.")
        return

    mask_arr = gdal.Open(MASK_PATH).ReadAsArray().astype(bool)

    stacked = []
    for variety in varieties:
        path = f'./data_output/original/module5{RUN_TAG}/{variety}/{year}/yield_terrain.tif'
        if not os.path.exists(path):
            print(f"    ⚠ Missing yield map for {variety} in {year} — skipping.")
            continue
        arr = gdal.Open(path).ReadAsArray().astype(float)
        arr[arr == -999] = 0        # screened-out pixels → 0, not nodata
        arr[~mask_arr] = np.nan     # outside mask → NaN for plotting only
        stacked.append(arr)

    stacked = np.stack(stacked, axis=0)       # (n_varieties, rows, cols)
    best_raw = np.nanmax(stacked, axis=0)     # best raw yield per cell

    # Classify ONCE across all varieties together
    best_class = obj_util.classifyFinalYield(best_raw)

    # -- Plot --
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im0 = axes[0].imshow(best_raw, cmap='YlGn')
    axes[0].set_title(f'Best Raw Yield — {output_tag} {year}')
    plt.colorbar(im0, ax=axes[0], shrink=0.8, label='kg/ha')

    im1 = axes[1].imshow(best_class, cmap=plt.get_cmap('tab10', 6), vmin=0, vmax=5)
    axes[1].set_title(f'Yield Suitability Class — {output_tag} {year}')
    plt.colorbar(im1, ax=axes[1], shrink=0.8, ticks=[0,1,2,3,4,5],
                 label='0=none 1=not suitable … 5=very suitable')

    plt.tight_layout()
    plt.savefig(f'{out_dir}/{year}_combined.png', bbox_inches='tight', dpi=150)
    plt.close()

    obj_util.saveRaster(BASEPATH, f'{out_dir}/{year}_raw_yield.tif', best_raw)
    obj_util.saveRaster(BASEPATH, f'{out_dir}/{year}_final_yield_class.tif', best_class)

    return best_raw, best_class

# =============================================================================
# MAIN
# =============================================================================

def run_all_module1():
    """Run once. Re-run only if climate data changes."""
    mask      = gdal.Open(MASK_PATH).ReadAsArray()
    elevation = np.load(r'./data_input/terrain/elevation.npy')

    for year in YEARS:

         # Load thermal climate and zones based on average period that includes this year
        if year in range(1979, 1999):
            avg_period= '1979-1998'
        else:                
            avg_period= '1999-2018'

        tclimate = gdal.Open(f'./data_output/module1/{avg_period}/thermalClimate.tif').ReadAsArray()
        tzone    = gdal.Open(f'./data_output/module1/{avg_period}/thermalZone.tif').ReadAsArray()

        if NO_THAW_BASELINE_RUN and year in range(1999, 2019):
            permafrost_year = year - 20  # 1999→1979, 2000→1980, ..., 2018→1998
            print(f"    [NO THAW] Using permafrost from {permafrost_year} (instead of {year})")
        else:
            permafrost_year = year
        try: 
            clim = load_climate(year, permafrost_year=permafrost_year)
            run_module1_year(year, mask, elevation, tclimate, tzone, clim)
            del clim
        except Exception as e:
            raise RuntimeError(f"Failed on year {year}") from e

def main():
    # -- Load static data (once) --
    print("Loading static data …")
    mask      = gdal.Open(MASK_PATH).ReadAsArray()
    elevation = np.load(r'./data_input/terrain/elevation.npy')

    # -- Outer loop: crops --
    for crop in CROPS:
        print(f"\n{'='*60}")
        print(f"  CROP: {crop['crop_name']}")
        print(f"{'='*60}")

        # -- Inner loop: years --
        for year in YEARS:
            print(f"\n  --- Year: {year} ---")

            # After:
            if NO_THAW_BASELINE_RUN and year in range(1999, 2019):
                permafrost_year = year - 20  # 1999→1979, 2000→1980, ..., 2018→1998
                print(f"    [NO THAW] Using permafrost from {permafrost_year} (instead of {year})")
            else:
                permafrost_year = year

            # Load climate data for this year
            clim = load_climate(year, permafrost_year=permafrost_year)

            # Load permafrost classification (produced by Module 1 in prior run,
            # or by the permafrost standalone script)
            
            permafrost_class = gdal.Open(
                f'./data_output/original/module1/{year}/permafrost.tif').ReadAsArray()

             # Load thermal climate and zones based on average period that includes this year
            if year in range(1979, 1999):
                avg_period= '1979-1998'
            else:                
                avg_period= '1999-2018'
            
            tclimate = gdal.Open(f'./data_output/module1/{avg_period}/thermalClimate.tif').ReadAsArray()
            lgp = gdal.Open(f'./data_output/original/module1{RUN_TAG}/{year}/LGP New.tif').ReadAsArray()
            lgpt5 = gdal.Open(f'./data_output/original/module1{RUN_TAG}/{year}/LGPt5.tif').ReadAsArray()
            lgpt10 = gdal.Open(f'./data_output/original/module1{RUN_TAG}/{year}/LGPt10.tif').ReadAsArray()
            m1 = {'lgp': lgp, 'lgpt5': lgpt5, 'lgpt10': lgpt10}

            try:
                # Module 2: crop simulation
                yield_rain, start_date = run_module2(
                    year, crop, mask, elevation, clim, tclimate, m1, permafrost_class
                )

                # Module 4: soil constraints
                yield_rain_m4 = run_module4(year, crop, yield_rain)

                # Module 5: terrain constraints
                yield_rain_m5 = run_module5(year, crop, yield_rain_m4, clim['precip'])

                # Final classification            
                final_class = final_yield_classification(year, crop, yield_rain_m5)
                # Free year's climate data
                del clim
            except Exception as e:
                raise RuntimeError(f"Failed on year {year} for crop {crop['crop_name']}") from e

            print(f"  ✓ Year {year} complete for {crop['crop_name']}")
        
        print(f"\n  ✓ All years complete for crop: {crop['crop_name']}")

    print("\n\nAll crops and years complete.")

def run_from_module4():
    print("Loading static data …")

    for crop in CROPS:
        print(f"\n{'='*60}")
        print(f"  CROP: {crop['crop_name']}")
        print(f"{'='*60}")

        for year in YEARS:
            print(f"\n  --- Year: {year} ---")

            # Load saved module 2 yield output
            m2_path = f'./data_output/original/module2{RUN_TAG}/{crop["crop_name"]}/{year}/yield_map_rain.tif'
            ds = gdal.Open(m2_path)
            if ds is None:
                print(f"    ⚠ Module 2 output not found for {crop['crop_name']} {year} — skipping.")
                continue
            yield_rain = ds.ReadAsArray().astype(float)
            yield_rain[yield_rain < 0] = 0  # clean nodata values

            if not np.any(yield_rain > 0):
                print(f"    ⚠ No valid yield in Module 2 output for {crop['crop_name']} {year} — skipping.")
                continue

            # Load precip for module 5
            clim_precip = np.load(f'data_input/climate_yearly/{year}/Precip.npy')

            try:
                yield_rain_m4 = run_module4(year, crop, yield_rain)
                yield_rain_m5 = run_module5(year, crop, yield_rain_m4, clim_precip)
                final_class   = final_yield_classification(year, crop, yield_rain_m5)
            except Exception as e:
                raise RuntimeError(f"Failed on year {year} for crop {crop['crop_name']}") from e

            print(f"  ✓ Year {year} complete for {crop['crop_name']}")

        print(f"\n  ✓ All years complete for crop: {crop['crop_name']}")

    print("\n\nAll crops and years complete.")


if __name__ == '__main__':
    # run_from_module4()
    # main()
    # run_all_module1()
    varieties = ["spring_rape_220", "spring_rape_221", "spring_rape_222", "spring_rape_223"]
    for year in YEARS:
        combine_crop_maps(year, varieties, output_tag='combined_spring_rape')
    plot_final_classification("combined_spring_rape")