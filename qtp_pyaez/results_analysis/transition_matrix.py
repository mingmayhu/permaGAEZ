"""
Suitability Class Transition Matrix Analysis
=============================================
For each crop, computes the transition matrix between counterfactual and
observed suitability classes in the same year (1999–2018).

For each pixel in each year:
  CF class → Observed class

Aggregated across all years to give mean annual pixel counts and area.

This reveals WHERE in the suitability distribution thaw is having its effect:
  - 0→1: frontier expansion (thaw opens new land)
  - 1→2, 2→3 etc: within-suitable improvement
  - 1→0, 2→1 etc: degradation (thaw hurts suitability)
  - Diagonal: no change

Outputs:
  - Per-crop transition matrix heatmap
  - Per-crop transition matrix CSV
  - Summary figure: net transitions by type across all crops
  - Summary of positive vs negative transitions by crop
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from osgeo import gdal

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR  = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
MASK_PATH = r'./data_input/qilian mask.tif'
OUT_ROOT  = './results_analysis/outputs/5_transition_matrix'

YEARS_CF = list(range(1999, 2019))
PIXEL_AREA_KM2 = 78.0

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

CLASSES     = list(range(6))   # 0–5
CLASS_NAMES = ['0\n(none)', '1\n(not suit.)', '2\n(marginal)',
               '3\n(moderate)', '4\n(suitable)', '5\n(very suit.)']

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
    return load_raster(MASK_PATH).astype(bool)

def obs_suit_path(tag, year):
    return f'./data_output/final_classification_fixed/{tag}/{year}_suitability_class.tif'

def cf_suit_path(tag, year):
    return f'./data_output/final_classification_nothaw_fixed/{tag}/{year}_suitability_class.tif'

def clean_class(arr, mask):
    """Convert raster to integer class array, masking invalid pixels."""
    out = arr.copy()
    out[~mask] = -1
    out[~np.isfinite(out)] = -1
    out[out < 0] = -1
    return out.astype(int)

def transition_label(cf_cls, obs_cls):
    """Return human-readable label for a transition."""
    diff = obs_cls - cf_cls
    if diff == 0:
        return 'no change'
    elif diff > 0:
        return f'+{diff} (improvement)'
    else:
        return f'{diff} (degradation)'


# ── Main ──────────────────────────────────────────────────────────────────────

def run(mask):
    all_summary = []
    all_net     = []

    for crop in CROPS:
        tag, label = crop['tag'], crop['label']
        print(f'\n── {label} ──')

        # Accumulate pixel counts across all years
        # matrix[cf_class, obs_class] = total pixel-years
        matrix = np.zeros((6, 6), dtype=float)
        n_years = 0

        for year in YEARS_CF:
            obs = load_raster(obs_suit_path(tag, year))
            cf  = load_raster(cf_suit_path(tag, year))

            if obs is None or cf is None:
                print(f'  ⚠ missing {year}')
                continue

            obs_cls = clean_class(obs, mask)
            cf_cls  = clean_class(cf,  mask)

            valid = mask & (obs_cls >= 0) & (cf_cls >= 0)

            for r in CLASSES:
                for c in CLASSES:
                    matrix[r, c] += np.sum((cf_cls == r) & (obs_cls == c) & valid)

            n_years += 1

        if n_years == 0:
            print(f'  ⚠ No valid years for {label}')
            continue

        # Convert to mean annual pixel counts and km²
        matrix_annual = matrix / n_years
        matrix_km2    = matrix_annual * PIXEL_AREA_KM2

        # ── Transition summary ────────────────────────────────────────────────
        # Categorize transitions
        diagonal    = np.sum(matrix_annual[i, i] for i in CLASSES)
        improved    = np.sum(matrix_annual[r, c]
                             for r in CLASSES for c in CLASSES if c > r)
        degraded    = np.sum(matrix_annual[r, c]
                             for r in CLASSES for c in CLASSES if c < r)
        total_valid = diagonal + improved + degraded

        pct_no_change  = 100 * diagonal / total_valid if total_valid > 0 else 0
        pct_improved   = 100 * improved  / total_valid if total_valid > 0 else 0
        pct_degraded   = 100 * degraded  / total_valid if total_valid > 0 else 0

        print(f'  No change: {pct_no_change:.1f}% | '
              f'Improved: {pct_improved:.1f}% | '
              f'Degraded: {pct_degraded:.1f}%')

        all_summary.append({
            'crop'          : label,
            'pct_no_change' : round(pct_no_change, 2),
            'pct_improved'  : round(pct_improved, 2),
            'pct_degraded'  : round(pct_degraded, 2),
            'area_improved_km2' : round(improved  * PIXEL_AREA_KM2, 2),
            'area_degraded_km2' : round(degraded  * PIXEL_AREA_KM2, 2),
            'net_area_km2'      : round((improved - degraded) * PIXEL_AREA_KM2, 2),
        })

        # Net transitions by step size
        for step in range(-5, 6):
            if step == 0:
                continue
            px = sum(matrix_annual[r, c]
                     for r in CLASSES for c in CLASSES
                     if c - r == step and 0 <= r <= 5 and 0 <= c <= 5)
            all_net.append({
                'crop'       : label,
                'step'       : step,
                'pixels_annual': round(px, 2),
                'area_km2'   : round(px * PIXEL_AREA_KM2, 2),
                'direction'  : 'improvement' if step > 0 else 'degradation',
            })

        # ── Save CSV ──────────────────────────────────────────────────────────
        df_mat = pd.DataFrame(
            matrix_km2,
            index  =[f'CF_{c}' for c in CLASSES],
            columns=[f'Obs_{c}' for c in CLASSES]
        )
        df_mat.to_csv(f'{OUT_ROOT}/{tag}_transition_matrix.csv')

        # ── Heatmap ───────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Left: full matrix heatmap
        ax = axes[0]
        # Mask diagonal for colour scale so off-diagonal transitions are visible
        off_diag = matrix_km2.copy()
        for i in CLASSES:
            off_diag[i, i] = np.nan
        vmax = np.nanpercentile(off_diag, 98) if np.any(np.isfinite(off_diag)) else 1

        im = ax.imshow(matrix_km2, cmap='YlOrRd', aspect='auto',
                       vmin=0, vmax=vmax)
        ax.set_xticks(CLASSES)
        ax.set_xticklabels(CLASS_NAMES, fontsize=9)
        ax.set_yticks(CLASSES)
        ax.set_yticklabels(CLASS_NAMES, fontsize=9)
        ax.set_xlabel('Observed Class (with thaw)', fontsize=11)
        ax.set_ylabel('Counterfactual Class (no thaw)', fontsize=11)
        ax.set_title(f'Mean Annual Transition Area (km²)\n{label}',
                     fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Mean Annual Area (km²)')

        # Annotate cells
        for r in CLASSES:
            for c in CLASSES:
                val = matrix_km2[r, c]
                if val > 0:
                    color = 'white' if val > vmax * 0.6 else 'black'
                    ax.text(c, r, f'{val:.1f}', ha='center', va='center',
                            fontsize=7, color=color)

        # Highlight diagonal
        for i in CLASSES:
            ax.add_patch(mpatches.Rectangle(
                (i - 0.5, i - 0.5), 1, 1,
                fill=False, edgecolor='blue', linewidth=1.5
            ))

        # Right: net transition bar chart by step
        ax2 = axes[1]
        net_df = pd.DataFrame([r for r in all_net if r['crop'] == label])
        if not net_df.empty:
            colors = ['#2166AC' if s > 0 else '#D6604D'
                      for s in net_df['step']]
            ax2.bar(net_df['step'], net_df['area_km2'],
                    color=colors, edgecolor='white', width=0.7)
            ax2.axhline(0, color='black', linewidth=0.8)
            ax2.set_xlabel('Class Change (Observed − Counterfactual)', fontsize=11)
            ax2.set_ylabel('Mean Annual Area (km²)', fontsize=11)
            ax2.set_title('Net Transitions by Step Size\nBlue = improvement, Red = degradation',
                          fontsize=12, fontweight='bold')
            ax2.set_xticks(range(-5, 6))
            ax2.set_xticklabels([f'{s:+d}' for s in range(-5, 6)], fontsize=9)

        fig.suptitle(f'{label} — Suitability Class Transitions (CF → Observed)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        fig.savefig(f'{OUT_ROOT}/{tag}_transitions.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  ✓ {label} saved')

    # ── Summary across all crops ───────────────────────────────────────────────
    df_summary = pd.DataFrame(all_summary)
    df_summary.to_csv(f'{OUT_ROOT}/transition_summary.csv', index=False)

    df_net = pd.DataFrame(all_net)
    df_net.to_csv(f'{OUT_ROOT}/net_transitions_by_step.csv', index=False)

    # Summary figure: improved vs degraded % per crop
    df_s = df_summary.sort_values('pct_improved', ascending=True)
    x     = np.arange(len(df_s))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: % improved vs degraded
    axes[0].barh(df_s['crop'], df_s['pct_improved'],
                 color='#2166AC', alpha=0.85, label='Improved')
    axes[0].barh(df_s['crop'], -df_s['pct_degraded'],
                 color='#D6604D', alpha=0.85, label='Degraded')
    axes[0].axvline(0, color='black', linewidth=0.8)
    axes[0].set_xlabel('% of Pixels per Year', fontsize=11)
    axes[0].set_title('% Pixels Improved vs Degraded\nby Thaw (mean 1999–2018)',
                      fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)

    # Right: net area
    colors = ['#2166AC' if v >= 0 else '#D6604D'
              for v in df_s['net_area_km2']]
    axes[1].barh(df_s['crop'], df_s['net_area_km2'],
                 color=colors, edgecolor='white')
    axes[1].axvline(0, color='black', linewidth=0.8)
    axes[1].set_xlabel('Net Area (km²)', fontsize=11)
    axes[1].set_title('Net Area Changed by Thaw\n(Improved − Degraded, mean 1999–2018)',
                      fontsize=12, fontweight='bold')

    fig.suptitle('Suitability Class Transitions: Impact of Permafrost Thaw\n'
                 'Observed vs. No-Thaw Counterfactual (1999–2018)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{OUT_ROOT}/transition_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Also plot net transitions aggregated across all crops
    df_net_agg = df_net.groupby('step')['area_km2'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#2166AC' if s > 0 else '#D6604D' for s in df_net_agg['step']]
    ax.bar(df_net_agg['step'], df_net_agg['area_km2'],
           color=colors, edgecolor='white', width=0.7)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Class Change (Observed − Counterfactual)', fontsize=11)
    ax.set_ylabel('Total Area Across All Crops (km²)', fontsize=11)
    ax.set_title('Net Suitability Transitions — All Crops Combined\n'
                 'Blue = improvement, Red = degradation (mean 1999–2018)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(range(-5, 6))
    ax.set_xticklabels([f'{s:+d}' for s in range(-5, 6)], fontsize=9)
    plt.tight_layout()
    fig.savefig(f'{OUT_ROOT}/net_transitions_all_crops.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f'\n✓ All outputs saved to: {OUT_ROOT}/')
    print('\nTransition Summary:')
    print(df_summary.to_string(index=False))


if __name__ == '__main__':
    mask = load_mask()
    run(mask)