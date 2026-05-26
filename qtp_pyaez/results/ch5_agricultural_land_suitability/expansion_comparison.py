"""
Expansion vs Intensification — Count-based vs Magnitude-based comparison
=========================================================================
Adds count-based metrics alongside the existing magnitude-based metrics
so both approaches can be compared directly.

Count-based approach (NEW):
  - Expansion:       number of pixels crossing ≤1 → ≥2
  - Loss:            number of pixels crossing ≥2 → ≤1
  - Intensification: number of pixels already suitable (≥2) in both periods
  - % contribution based on pixel counts

Magnitude-based approach (ORIGINAL):
  - Expansion:       sum of (post - pre) across newly-suitable pixels
  - Intensification: sum of (post - pre) across already-suitable pixels
  - Loss:            sum of (post - pre) across lost pixels
  - % contribution based on summed class differences

Outputs:
  ./results/agricultural_land_suitability/outputs/expansion_count_based.csv
  ./results/agricultural_land_suitability/outputs/expansion_magnitude_based.csv
  ./results/agricultural_land_suitability/outputs/expansion_comparison.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from osgeo import gdal

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR  = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH = r'./data_input/qilian_mask_new.tif'
OUT_ROOT  = r'./results/agricultural_land_suitability/outputs'
PERMAFROST_PATH = r'./data_input/permafrost_qilian.tif'

YEARS_ALL  = list(range(1979, 2019))
YEARS_PRE  = list(range(1979, 1999))
YEARS_POST = list(range(1999, 2019))

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

FONTSIZE_TITLE = 12
FONTSIZE_LABEL = 10
FONTSIZE_TICK  = 8
DPI = 150

os.chdir(WORK_DIR)
os.makedirs(OUT_ROOT, exist_ok=True)


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
    mask = load_raster(MASK_PATH).astype(bool)
    pf_arr = load_raster(PERMAFROST_PATH)
    if pf_arr is not None:
        lake_mask = ((pf_arr == 0) | ~np.isfinite(pf_arr)) & mask
        mask[lake_mask] = False
        print(f'  Excluded {lake_mask.sum()} lake/nodata pixels from mask')
    return mask

def obs_suit_path(tag, year):
    return f'./data_output/final_classification_fixed/{tag}/{year}_suitability_class.tif'

def remap(arr_int):
    out = arr_int.copy()
    out[out == 0] = 1
    return out

def build_period_means(tag, mask):
    """Build pre and post period mean suitability maps for one crop."""
    pre_stack, post_stack = [], []
    for year in YEARS_ALL:
        arr = load_raster(obs_suit_path(tag, year))
        if arr is None:
            continue
        arr_int = np.where(mask & np.isfinite(arr), arr, 0).astype(int)
        arr_int = np.clip(arr_int, 0, 5)
        arr_int = remap(arr_int)
        arr_flt = arr_int.astype(float)
        arr_flt[~mask] = np.nan
        if year in YEARS_PRE:
            pre_stack.append(arr_flt)
        else:
            post_stack.append(arr_flt)

    pre  = np.nanmean(pre_stack,  axis=0) if pre_stack  else None
    post = np.nanmean(post_stack, axis=0) if post_stack else None
    return pre, post


def compute_magnitude_based(pre, post, mask):
    """Original magnitude-based approach."""
    pre_cls  = np.round(pre).astype(int)
    post_cls = np.round(post).astype(int)
    valid    = mask & np.isfinite(pre) & np.isfinite(post)

    newly_suitable   = valid & (pre_cls <= 1) & (post_cls >= 2)
    lost_suitable    = valid & (pre_cls >= 2) & (post_cls <= 1)
    already_suitable = valid & (pre_cls >= 2) & (post_cls >= 2)

    expansion_units       = float(np.nansum(post[newly_suitable] - pre[newly_suitable])) \
                            if newly_suitable.sum() > 0 else 0.0
    intensification_units = float(np.nansum(post[already_suitable] - pre[already_suitable])) \
                            if already_suitable.sum() > 0 else 0.0
    lost_units            = float(np.nansum(post[lost_suitable] - pre[lost_suitable])) \
                            if lost_suitable.sum() > 0 else 0.0

    total = expansion_units + intensification_units + lost_units

    return {
        'expansion_units':          round(expansion_units, 2),
        'intensification_units':    round(intensification_units, 2),
        'lost_units':               round(lost_units, 2),
        'total_units':              round(total, 2),
        'pct_expansion':            round(100 * expansion_units / total, 1) if total > 0 else np.nan,
        'pct_intensification':      round(100 * intensification_units / total, 1) if total > 0 else np.nan,
        'pct_loss':                 round(100 * lost_units / total, 1) if total > 0 else np.nan,
        'newly_suitable_px':        int(newly_suitable.sum()),
        'lost_suitable_px':         int(lost_suitable.sum()),
        'already_suitable_px':      int(already_suitable.sum()),
    }


def compute_count_based(pre, post, mask):
    """New count-based approach — each pixel counted once."""
    pre_cls  = np.round(pre).astype(int)
    post_cls = np.round(post).astype(int)
    valid    = mask & np.isfinite(pre) & np.isfinite(post)

    newly_suitable   = valid & (pre_cls <= 1) & (post_cls >= 2)
    lost_suitable    = valid & (pre_cls >= 2) & (post_cls <= 1)
    already_suitable = valid & (pre_cls >= 2) & (post_cls >= 2)
    # pixels not suitable in either period — excluded
    # never_suitable = valid & (pre_cls <= 1) & (post_cls <= 1)

    n_expansion       = int(newly_suitable.sum())
    n_loss            = int(lost_suitable.sum())
    n_intensification = int(already_suitable.sum())

    # For % contribution: use net change pixels only (expansion + loss + intensification)
    # Intensification only counts pixels where there was actual improvement
    intensification_improved = already_suitable & (post_cls > pre_cls)
    intensification_declined = already_suitable & (post_cls < pre_cls)
    n_intens_improved = int(intensification_improved.sum())
    n_intens_declined = int(intensification_declined.sum())

    # % of pixels that changed suitability status or class
    n_changed = n_expansion + n_loss + n_intens_improved + n_intens_declined
    total_suitable_pixels = n_expansion + n_loss + n_intensification

    return {
        'newly_suitable_px':             n_expansion,
        'lost_suitable_px':              n_loss,
        'net_expansion_px':              n_expansion - n_loss,
        'already_suitable_px':           n_intensification,
        'intensification_improved_px':   n_intens_improved,
        'intensification_declined_px':   n_intens_declined,
        'total_suitable_px':             total_suitable_pixels,
        'pct_expansion_of_suitable':     round(100 * n_expansion / total_suitable_pixels, 1)
                                         if total_suitable_pixels > 0 else np.nan,
        'pct_loss_of_suitable':          round(100 * n_loss / total_suitable_pixels, 1)
                                         if total_suitable_pixels > 0 else np.nan,
        'pct_intensification_of_suitable': round(100 * n_intensification / total_suitable_pixels, 1)
                                           if total_suitable_pixels > 0 else np.nan,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    mask = load_mask()

    rows_mag   = []
    rows_count = []

    for crop in CROPS:
        print(f'  Processing {crop["label"]} …')
        pre, post = build_period_means(crop['tag'], mask)
        if pre is None or post is None:
            print(f'    ✗ Missing data for {crop["label"]}')
            continue

        mag   = compute_magnitude_based(pre, post, mask)
        count = compute_count_based(pre, post, mask)

        rows_mag.append({'crop': crop['label'], **mag})
        rows_count.append({'crop': crop['label'], **count})

    df_mag   = pd.DataFrame(rows_mag)
    df_count = pd.DataFrame(rows_count)

    df_mag.to_csv(f'{OUT_ROOT}/expansion_magnitude_based.csv',   index=False)
    df_count.to_csv(f'{OUT_ROOT}/expansion_count_based.csv', index=False)

    print('\nMagnitude-based:')
    print(df_mag[['crop', 'pct_expansion', 'pct_intensification', 'pct_loss']].to_string(index=False))
    print('\nCount-based:')
    print(df_count[['crop', 'newly_suitable_px', 'lost_suitable_px',
                    'already_suitable_px', 'pct_expansion_of_suitable',
                    'pct_intensification_of_suitable']].to_string(index=False))

    # ── Comparison figure ─────────────────────────────────────────────────────
    crop_labels = df_mag['crop'].tolist()
    x = np.arange(len(crop_labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Magnitude-based
    axes[0].bar(x, df_mag['pct_intensification'], color='#2166AC', alpha=0.85,
                label='Intensification')
    axes[0].bar(x, df_mag['pct_expansion'],
                bottom=df_mag['pct_intensification'],
                color='#92C5DE', alpha=0.85, label='Expansion')
    axes[0].bar(x, df_mag['pct_loss'].abs() if 'pct_loss' in df_mag else 0,
                bottom=df_mag['pct_intensification'] + df_mag['pct_expansion'],
                color='#D6604D', alpha=0.85, label='Loss')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(crop_labels, rotation=30, ha='right', fontsize=FONTSIZE_TICK)
    axes[0].set_ylabel('% of Total Suitability Change (magnitude)', fontsize=FONTSIZE_LABEL)
    axes[0].set_title('Magnitude-Based\n(sum of class differences)',
                      fontsize=FONTSIZE_TITLE, fontweight='bold')
    axes[0].axhline(50, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(0, 110)

    # Count-based
    axes[1].bar(x, df_count['pct_intensification_of_suitable'], color='#2166AC', alpha=0.85,
                label='Intensification (already suitable)')
    axes[1].bar(x, df_count['pct_expansion_of_suitable'],
                bottom=df_count['pct_intensification_of_suitable'],
                color='#92C5DE', alpha=0.85, label='Expansion (newly suitable)')
    axes[1].bar(x, df_count['pct_loss_of_suitable'],
                bottom=df_count['pct_intensification_of_suitable'] + df_count['pct_expansion_of_suitable'],
                color='#D6604D', alpha=0.85, label='Loss')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(crop_labels, rotation=30, ha='right', fontsize=FONTSIZE_TICK)
    axes[1].set_ylabel('% of Total Suitable Pixels', fontsize=FONTSIZE_LABEL)
    axes[1].set_title('Count-Based\n(pixel counts)',
                      fontsize=FONTSIZE_TITLE, fontweight='bold')
    axes[1].axhline(50, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(0, 110)

    fig.suptitle('Expansion vs Intensification: Magnitude-Based vs Count-Based\n'
                 '(1979–1998 to 1999–2018)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f'{OUT_ROOT}/expansion_comparison.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f'\n✓ Comparison figure saved to {OUT_ROOT}/expansion_comparison.png')
    print(f'✓ CSVs saved to {OUT_ROOT}/')


if __name__ == '__main__':
    main()