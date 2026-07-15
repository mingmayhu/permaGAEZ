"""
Reference Production → Suitability Class Comparison
=====================================================
Reclassifies reference production rasters (FAO GAEZ v5, Monfreda) into
suitability classes using the same fixed-boundary percentile binning as
the PermaGAEZ reclassification script, then compares against model mean
suitability class using Spearman rank correlation.

Class definitions (matching PermaGAEZ fixed-boundary approach):
  1 = not suitable       (0% of max, i.e. zero production)
  1 = lowest             (0–20% of per-crop max within study area)
  2 = marginally suitable(20–40%)
  3 = moderately suitable(40–60%)
  4 = suitable           (60–80%)
  5 = very suitable      (>80%)

Zero production pixels → class 1 (not suitable / lowest)

Reference sources and model periods:
  FAO GAEZ v5 production  → model mean 1999–2018
  Monfreda (2000)         → model mean 1979–1998

Outputs:
  results/validation/outputs/suitability_class_comparison.csv
  results/validation/outputs/suitability_class_comparison/{crop}_{source}.png
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from osgeo import gdal
from scipy import stats

# ── Config ────────────────────────────────────────────────────────────────────

BASE         = "/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez"
FAO_PROD_DIR = os.path.join(BASE, "data_input/fao_actual_prod")
MRF_BASE     = "/Users/ming-mayhu/Downloads/HarvestedAreaYield175Crops_Geotiff/GeoTiff"
OUTPUT_DIR   = os.path.join(BASE, "results/validation/outputs/suitability_class_comparison")

REFERENCE_TIF = os.path.join(
    BASE, "data_output/final_classification_fixed/combined_barley/1999_suitability_class.tif")

# (crop_label, model_folder, fao_prod_file_or_None, monfreda_subfolder_or_None)
CROPS = [
    ("Barley",       "combined_barley",       "barley.tif",    "barley"),
    ("Spring Oat",   "combined_oat",          None,            "oats"),
    ("Dry Pea",      "combined_dry_pea",      None,            "pea"),
    ("Rapeseed",     "combined_rape",         "rapeseed.tif",  "rapeseed"),
    ("Wheat",        "combined_wheat",        "wheat.tif",     "wheat"),
    ("White Potato", "combined_white_potato", None,            "potato"),
]

MODELS = [
    ("Permafrost Model", "data_output/final_classification_fixed"),
    ("No-Thaw Model",    "data_output/final_classification_nothaw_fixed"),
    ("Original Model",   "data_output/original/final_classification_fixed"),
]

AVG_YEARS_RECENT   = list(range(1999, 2019))  # FAO → 1999–2018
AVG_YEARS_BASELINE = list(range(1979, 1999))  # Monfreda → 1979–1998

# No-thaw model lacks 1979–1998 files
MONFREDA_SKIP_MODELS = {"No-Thaw Model"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_ref_info():
    ds  = gdal.Open(REFERENCE_TIF)
    geo = ds.GetGeoTransform()
    shp = (ds.RasterYSize, ds.RasterXSize)
    ds  = None
    return geo, shp


def warp_to_ref(path, ref_geo, ref_shape, resample=gdal.GRA_Average):
    rows, cols = ref_shape
    x_min = ref_geo[0];  y_max = ref_geo[3]
    x_res = ref_geo[1];  y_res = abs(ref_geo[5])
    ds = gdal.Warp("", path, format="MEM",
                   outputBounds=(x_min, y_max - rows*y_res,
                                 x_min + cols*x_res, y_max),
                   xRes=x_res, yRes=y_res, resampleAlg=resample)
    band   = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    arr    = band.ReadAsArray().astype(float)
    arr[arr < -1e10] = np.nan
    if nodata is not None:
        arr[np.abs(arr - nodata) < 1e-6] = np.nan
    ds = None
    return arr


def build_lake_mask(ref_geo, ref_shape):
    """Load the combined study area mask (excludes lakes, outside area)."""
    arr = warp_to_ref(
        os.path.join(BASE, "data_input/qilian_mask_new.tif"),
        ref_geo, ref_shape, resample=gdal.GRA_NearestNeighbour)
    return np.isfinite(arr) & (arr > 0)


def get_pixel_ha(ref_geo, ref_shape):
    x_res_deg  = abs(ref_geo[1])
    y_res_deg  = abs(ref_geo[5])
    centre_lat = ref_geo[3] + (ref_shape[0] / 2) * ref_geo[5]
    return (x_res_deg * 111.32 * math.cos(math.radians(centre_lat))
            * y_res_deg * 110.574 * 100)


def production_to_output_density(prod_path, scale_kg_per_raw, pixel_ha,
                                  ref_geo, ref_shape):
    """
    Load production raster and convert to kg/ha of total pixel area.
    Pixels outside raster extent → 0 (no production, not missing).
    """
    arr = warp_to_ref(prod_path, ref_geo, ref_shape)
    arr = np.where(np.isfinite(arr) & (arr >= 0), arr, 0.0)
    return arr * scale_kg_per_raw / pixel_ha


def classify_production(density_arr, keep):
    """
    Reclassify production output density into suitability classes 1–5
    using fixed percentile boundaries computed within the study area.

    Zero-production pixels → class 1 (not suitable / lowest).
    Non-zero pixels binned at 20th, 40th, 60th, 80th percentile of
    non-zero values (equivalent to 0–20%, 20–40%, … of per-crop max).
    """
    out = np.full(density_arr.shape, np.nan)

    # Study area pixels only
    study = density_arr.copy()
    study[~keep] = np.nan

    nonzero = study[np.isfinite(study) & (study > 0)]
    if len(nonzero) == 0:
        print("  WARNING: no non-zero production pixels in study area")
        out[keep] = 1.0
        return out, None

    y_max = float(np.max(nonzero))
    b20 = 0.20 * y_max
    b40 = 0.40 * y_max
    b60 = 0.60 * y_max
    b80 = 0.80 * y_max

    boundaries = {"y_max": y_max, "b20": b20, "b40": b40, "b60": b60, "b80": b80}
    print(f"  Production boundaries: max={y_max:.3f}, "
          f"b20={b20:.3f}, b40={b40:.3f}, b60={b60:.3f}, b80={b80:.3f} kg/ha")

    # Assign classes
    out[keep] = 1.0  # default: class 1 (zero production or lowest)
    valid_nonzero = keep & np.isfinite(study) & (study > 0)
    out[valid_nonzero & (study <= b20)] = 1
    out[valid_nonzero & (study > b20) & (study <= b40)] = 2
    out[valid_nonzero & (study > b40) & (study <= b60)] = 3
    out[valid_nonzero & (study > b60) & (study <= b80)] = 4
    out[valid_nonzero & (study > b80)] = 5

    return out, boundaries


def load_mean_suitability(model_dir, crop_folder, years, ref_geo, ref_shape):
    """Average annual suitability class rasters across a period."""
    stacked = []
    missing = []
    corrupt = []
    for year in years:
        if model_dir == "data_output/original/final_classification_nothaw_fixed" and crop_folder == "combined_oat":
            crop_folder = "combined_spring_oat_NEW"
        path = os.path.join(BASE, model_dir, crop_folder,
                            f"{year}_suitability_class.tif")
        if not os.path.exists(path):
            missing.append(year)
            continue
        test = gdal.Open(path)
        if test is None:
            corrupt.append(year)
            continue
        test = None
        arr = warp_to_ref(path, ref_geo, ref_shape,
                          resample=gdal.GRA_NearestNeighbour)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        stacked.append(arr)
    if missing:
        print(f"  WARNING: missing years {missing[:5]}{'...' if len(missing)>5 else ''}")
    if corrupt:
        print(f"  WARNING: corrupt files skipped: {corrupt}")
    if not stacked:
        raise FileNotFoundError(f"No files found: {model_dir}/{crop_folder}")
    return np.nanmean(np.array(stacked), axis=0)


def _draw_crop_panels(axes_list, all_crop_data, source_label,
                      model_colors, model_markers, ref_class_vals):
    """Draw one source's data into a list of axes (one per crop).

    Spearman r is the focal point:
      - Line thickness proportional to r value
      - r annotated prominently at end of each line
      - No regression line — r tells the discrimination story
    """
    for ax, crop_data in zip(axes_list, all_crop_data):
        crop_label   = crop_data["crop_label"]
        sources_data = crop_data["sources_data"]
        keep         = crop_data["keep"]

        if source_label not in sources_data:
            ax.set_visible(False)
            continue

        source_info = sources_data[source_label]
        ref_masked  = source_info["ref_masked"]
        models_dict = source_info["models_dict"]

        # Pixel counts
        valid0 = keep & np.isfinite(ref_masked)
        a0     = ref_masked[valid0]
        ns_ref = [int(np.sum(np.round(a0).astype(int) == c))
                  for c in ref_class_vals]

        # Get r range for thickness scaling
        r_vals = [t[1] for t in models_dict.values()]
        r_min  = min(r_vals) if r_vals else 0
        r_max  = max(r_vals) if r_vals else 1
        r_range = max(r_max - r_min, 0.01)

        legend_handles = []

        for model_label, (model_arr, ri, pi, prec, rec, f1) in models_dict.items():
            color  = model_colors.get(model_label, "grey")
            marker = model_markers.get(model_label, "o")

            # Line thickness proportional to Spearman r (1.0–3.5 range)
            lw = 1.0 + 2.5 * (ri - r_min) / r_range

            valid = keep & np.isfinite(ref_masked) & np.isfinite(model_arr)
            a = ref_masked[valid]
            m = model_arr[valid]
            means, sds = [], []
            for c in ref_class_vals:
                subset = m[np.round(a).astype(int) == c]
                means.append(np.mean(subset) if len(subset) > 0 else np.nan)
                sds.append(np.std(subset)    if len(subset) > 0 else np.nan)

            means = np.array(means, dtype=float)
            sds   = np.array(sds,   dtype=float)
            xs    = np.array(ref_class_vals, dtype=float)
            vp    = np.isfinite(means) & np.isfinite(sds)

            # Shaded band
            if vp.sum() > 1:
                ax.fill_between(xs[vp], (means-sds)[vp], (means+sds)[vp],
                                color=color, alpha=0.10, linewidth=0)

            short = model_label.replace(" Model", "")
            line, = ax.plot(xs[vp], means[vp], color=color, linestyle="-",
                            marker=marker, markersize=5,
                            linewidth=lw, label=short)
            legend_handles.append(line)

            # Annotate r at end of line
            last_valid = np.where(vp)[0][-1]
            x_end = xs[last_valid]
            y_end = means[last_valid]
            ax.annotate(f"r={ri:.3f}",
                        xy=(x_end, y_end),
                        xytext=(6, 0), textcoords="offset points",
                        fontsize=8.5, fontweight="bold",
                        color=color, va="center")

        # Suitability threshold
        ax.axhline(y=2, color="grey", linestyle="--",
                   alpha=0.4, linewidth=0.8)

        ax.set_title(crop_label, fontsize=11, fontweight="bold", pad=5)
        ax.set_xlim(0.5, 5.9)  # slightly wider to fit r annotation
        ax.set_ylim(0, 5.5)
        ax.set_xticks(ref_class_vals)
        ax.set_xticklabels([f"{c}\n(n={n})" for c, n in
                             zip(ref_class_vals, ns_ref)], fontsize=7.5)
        ax.set_xlabel("Reference Production Class", fontsize=9)
        ax.set_ylabel("Mean Model Suitability\nClass (±1 SD)", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0.2, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Legend: model name + line thickness note
        legend_handles.append(
            plt.Line2D([0], [0], color="grey", linewidth=0.8,
                       alpha=0.5, label="thicker = higher r"))
        ax.legend(handles=legend_handles, fontsize=7.5,
                  loc="upper left", framealpha=0.85, edgecolor="none")


def _make_axes_centered(fig, n_crops, ncols, top, bottom, hspace, wspace):
    """Build axes list with last row centered if n_crops % ncols != 0."""
    import matplotlib.gridspec as gridspec
    nrows      = math.ceil(n_crops / ncols)
    n_last     = n_crops % ncols or ncols
    axes       = []

    if nrows == 1:
        gs = gridspec.GridSpec(1, n_crops, figure=fig,
                               top=top, bottom=bottom,
                               hspace=hspace, wspace=wspace)
        for i in range(n_crops):
            axes.append(fig.add_subplot(gs[0, i]))
    else:
        mid = (top + bottom) / 2 + 0.02
        gs_top = gridspec.GridSpec(nrows - 1, ncols, figure=fig,
                                   top=top, bottom=mid,
                                   hspace=hspace, wspace=wspace)
        for i in range(n_crops - n_last):
            axes.append(fig.add_subplot(gs_top[i // ncols, i % ncols]))
        offset = (ncols - n_last) / 2
        gs_bot = gridspec.GridSpec(1, ncols * 10, figure=fig,
                                   top=mid - 0.04, bottom=bottom,
                                   hspace=0, wspace=0)
        for j in range(n_last):
            start = int((offset + j) * 10)
            axes.append(fig.add_subplot(gs_bot[0, start:start + 9]))
    return axes


def plot_all_crops_figure(all_crop_data, out_path):
    """
    Two-row layout: top row = Monfreda, bottom row = FAO.
    One column per crop (up to ncols=5 for 5 crops).
    Shorter panel height than before.
    """
    model_colors  = {"Permafrost Model": "#2166ac",
                     "No-Thaw Model":    "#4dac26",
                     "Original Model":   "#d6604d"}
    model_markers = {"Permafrost Model": "o",
                     "No-Thaw Model":    "s",
                     "Original Model":   "^"}
    ref_class_vals = [1, 2, 3, 4, 5]

    # Determine which sources exist
    sources_present = []
    for src in ["Monfreda (2000)", "FAO Production"]:
        if any(src in d["sources_data"] for d in all_crop_data):
            sources_present.append(src)

    n_crops  = len(all_crop_data)
    ncols    = min(n_crops, 5)
    n_src    = len(sources_present)
    # Shorter panels: 3.5 inches tall each
    fig_h    = 3.5 * n_src + 0.6
    fig_w    = 4.5 * ncols

    fig = plt.figure(figsize=(fig_w, fig_h))

    src_labels = {"Monfreda (2000)": "Monfreda et al. (2000)  ·  1979–1998 model period",
                  "FAO Production":  "FAO GAEZ v5 Production  ·  1999–2018 model period\n"
                                     "(note: partial circularity — see text)"}

    slice_h   = 1.0 / n_src
    top_pad   = 0.06
    bot_pad   = 0.10
    inner_gap = 0.12

    for s_idx, source_label in enumerate(sources_present):
        top    = 1.0 - top_pad - s_idx * slice_h
        bottom = 1.0 - top_pad - (s_idx + 1) * slice_h + bot_pad

        axes = _make_axes_centered(fig, n_crops, ncols,
                                   top=top, bottom=bottom,
                                   hspace=0.4, wspace=0.38)

        # Source label as figure text on the left
        fig.text(0.01, (top + bottom) / 2,
                 src_labels[source_label],
                 va="center", ha="left", fontsize=9,
                 color="#444444", rotation=90)

        _draw_crop_panels(axes, all_crop_data, source_label,
                          model_colors, model_markers, ref_class_vals)

    fig.suptitle("Model Suitability Class vs Reference Production Class",
                 fontsize=13, fontweight="bold", y=0.99)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved combined figure: {os.path.basename(out_path)}")


def plot_performance_heatmap(all_crop_data, metric, out_path):
    """
    Option 4: Heatmap of model performance across all crops and sources.
    Rows = crops, columns = models, cell = Spearman r or F1.
    Two sub-tables side by side: Monfreda and FAO.

    metric: 'r' for Spearman r, 'f1' for F1 score
    """
    sources   = ["Monfreda (2000)", "FAO Production"]
    src_labels = {"Monfreda (2000)": "Monfreda (1979–1998)",
                  "FAO Production":  "FAO Production (1999–2018)†"}
    models    = ["Permafrost Model", "No-Thaw Model", "Original Model"]
    mod_short = {"Permafrost Model": "Permafrost",
                 "No-Thaw Model":    "No-Thaw",
                 "Original Model":   "Original"}
    crops     = [d["crop_label"] for d in all_crop_data]

    # Build data matrices: shape (n_crops, n_models) per source
    matrices = {}
    for src in sources:
        mat = np.full((len(crops), len(models)), np.nan)
        for ci, crop_data in enumerate(all_crop_data):
            if src not in crop_data["sources_data"]:
                continue
            models_dict = crop_data["sources_data"][src]["models_dict"]
            for mi, model_label in enumerate(models):
                if model_label not in models_dict:
                    continue
                _, ri, pi, prec, rec, f1 = models_dict[model_label]
                mat[ci, mi] = ri if metric == "r" else f1
        matrices[src] = mat

    # Figure: two heatmaps side by side
    n_src_with_data = sum(1 for s in sources
                          if not np.all(np.isnan(matrices[s])))
    fig, axes = plt.subplots(1, n_src_with_data,
                             figsize=(3.5 * n_src_with_data + 0.5, 0.55 * len(crops) + 1.2),
                             squeeze=False)

    metric_label = "Spearman r" if metric == "r" else "F1 Score"
    cmap  = "RdYlGn"
    vmin, vmax = (0, 0.5) if metric == "r" else (0, 0.5)

    ax_idx = 0
    for src in sources:
        mat = matrices[src]
        if np.all(np.isnan(mat)):
            continue
        ax = axes[0, ax_idx]

        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect="auto")

        # Cell annotations
        for ci in range(len(crops)):
            for mi in range(len(models)):
                val = mat[ci, mi]
                if not np.isnan(val):
                    # Bold the best model per crop (highest value)
                    row_vals = mat[ci, ~np.isnan(mat[ci])]
                    is_best  = (val == np.max(row_vals)) if len(row_vals) > 0 else False
                    ax.text(mi, ci, f"{val:.3f}",
                            ha="center", va="center",
                            fontsize=9,
                            fontweight="bold" if is_best else "normal",
                            color="white" if val > (vmax * 0.65) else "black")
                else:
                    ax.text(mi, ci, "—", ha="center", va="center",
                            fontsize=9, color="grey")

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([mod_short[m] for m in models],
                           fontsize=10, fontweight="bold")
        ax.set_yticks(range(len(crops)))
        ax.set_yticklabels(crops, fontsize=10)
        ax.set_title(src_labels[src], fontsize=11, fontweight="bold", pad=8)
        ax.tick_params(left=False, bottom=False)

        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add subtle grid
        ax.set_xticks(np.arange(-0.5, len(models), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(crops), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", length=0)

        plt.colorbar(im, ax=ax, shrink=0.6, label=metric_label,
                     fraction=0.04, pad=0.02)
        ax_idx += 1

    metric_title = "Spearman r" if metric == "r" else "F1 Score"
    fig.suptitle(f"Model Validation Performance — {metric_title}\n"
                 f"(bold = best per crop per source)",
                 fontsize=12, fontweight="bold")
    if "FAO" in "".join(sources):
        fig.text(0.5, -0.02,
                 "†FAO Production note: partial circularity with GAEZ methodology — see text",
                 ha="center", fontsize=8, color="grey", style="italic")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved heatmap: {os.path.basename(out_path)}")


def plot_difference_figure(all_crop_data, out_path):
    """
    Option 2: Difference plot — model suitability minus reference class.
    Values near 0 = good. Positive = model overpredicts. Negative = underpredicts.
    One panel per crop × source combination.
    Rows = sources, columns = crops.
    """
    import matplotlib.gridspec as gridspec

    sources    = ["Monfreda (2000)", "FAO Production"]
    src_labels = {"Monfreda (2000)": "Monfreda (1979–1998)",
                  "FAO Production":  "FAO Production (1999–2018)†"}
    model_colors  = {"Permafrost Model": "#2166ac",
                     "No-Thaw Model":    "#4dac26",
                     "Original Model":   "#d6604d"}
    model_markers = {"Permafrost Model": "o",
                     "No-Thaw Model":    "s",
                     "Original Model":   "^"}
    ref_class_vals = [1, 2, 3, 4, 5]

    crops      = [d["crop_label"] for d in all_crop_data]
    n_crops    = len(crops)
    src_active = [s for s in sources
                  if any(s in d["sources_data"] for d in all_crop_data)]
    n_src      = len(src_active)

    fig, axes = plt.subplots(n_src, n_crops,
                             figsize=(3.8 * n_crops, 3.2 * n_src),
                             squeeze=False)
    fig.suptitle("Model Suitability Class − Reference Production Class\n"
                 "(0 = perfect, positive = overprediction, negative = underprediction)",
                 fontsize=12, fontweight="bold")

    for s_idx, source_label in enumerate(src_active):
        for c_idx, crop_data in enumerate(all_crop_data):
            ax         = axes[s_idx, c_idx]
            crop_label = crop_data["crop_label"]
            keep       = crop_data["keep"]

            if source_label not in crop_data["sources_data"]:
                ax.set_visible(False)
                continue

            source_info = crop_data["sources_data"][source_label]
            ref_masked  = source_info["ref_masked"]
            models_dict = source_info["models_dict"]

            for model_label, (model_arr, ri, pi, prec, rec, f1) in models_dict.items():
                color  = model_colors.get(model_label, "grey")
                marker = model_markers.get(model_label, "o")

                valid = keep & np.isfinite(ref_masked) & np.isfinite(model_arr)
                a = ref_masked[valid]
                m = model_arr[valid]

                diffs, sds = [], []
                for c in ref_class_vals:
                    subset_diff = m[np.round(a).astype(int) == c] - c
                    diffs.append(np.mean(subset_diff) if len(subset_diff) > 0
                                 else np.nan)
                    sds.append(np.std(subset_diff) if len(subset_diff) > 0
                               else np.nan)

                diffs = np.array(diffs, dtype=float)
                sds   = np.array(sds,   dtype=float)
                xs    = np.array(ref_class_vals, dtype=float)
                vp    = np.isfinite(diffs) & np.isfinite(sds)

                if vp.sum() > 1:
                    ax.fill_between(xs[vp],
                                    (diffs - sds)[vp],
                                    (diffs + sds)[vp],
                                    color=color, alpha=0.10, linewidth=0)

                short = model_label.replace(" Model", "")
                ax.plot(xs[vp], diffs[vp], color=color, linestyle="-",
                        marker=marker, markersize=5, linewidth=1.6,
                        label=f"{short} (r={ri:.3f})")

            # Zero line = perfect agreement
            ax.axhline(y=0, color="black", linestyle="-",
                       alpha=0.4, linewidth=1.0)
            # Light shading for acceptable range ±1
            ax.axhspan(-1, 1, color="grey", alpha=0.06)

            if c_idx == 0:
                ax.set_ylabel(f"{src_labels[source_label]}\nModel − Ref Class",
                              fontsize=8)
            if s_idx == 0:
                ax.set_title(crop_label, fontsize=10,
                             fontweight="bold", pad=5)

            ax.set_xlim(0.5, 5.5)
            ax.set_ylim(-5.5, 5.5)
            ax.set_xticks(ref_class_vals)
            ax.set_xticklabels(ref_class_vals, fontsize=8)
            ax.set_xlabel("Reference Class", fontsize=8)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="y", alpha=0.2, linewidth=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(fontsize=6.5, loc="upper left",
                      framealpha=0.8, edgecolor="none")

    if "FAO" in " ".join(src_active):
        fig.text(0.5, -0.01,
                 "†FAO Production: partial circularity with GAEZ methodology — see text",
                 ha="center", fontsize=8, color="grey", style="italic")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved difference plot: {os.path.basename(out_path)}")


def plot_scatter_comparison(all_crop_data, metric, out_path):
    """
    Single panel scatter. Y-axis = Permafrost r.
    Blue (■/●) = Original r on x-axis
    Red  (■/●) = No-Thaw r on x-axis
    Shape: circle = Monfreda, square = FAO.
    Points above diagonal = Permafrost wins.
    Smart annotation offsets to reduce overlap.
    """
    sources    = ["Monfreda (2000)", "FAO Production"]
    src_shapes = {"Monfreda (2000)": "^", "FAO Production": "o"}
    src_labels = {"Monfreda (2000)": "EarthSTAT (1979–1998)",
                  "FAO Production":  "FAO (1999–2018)"}

    cmp_models = [
        ("Original Model", "#d6604d",  "PyAEZ"),
        ("No-Thaw Model",  "#2166ac", "No-thaw"),
    ]
    metric_label = "Spearman r" if metric == "r" else "F1 Score"

    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    points   = []
    all_vals = []

    for cmp_model, color, short_name in cmp_models:
        for crop_data in all_crop_data:
            crop_label   = crop_data["crop_label"]
            sources_data = crop_data["sources_data"]
            for source_label in sources:
                if source_label not in sources_data:
                    continue
                models_dict = sources_data[source_label]["models_dict"]
                if ("Permafrost Model" not in models_dict or
                        cmp_model not in models_dict):
                    continue
                _, ri_perm, _, _, _, f1_p = models_dict["Permafrost Model"]
                _, ri_cmp,  _, _, _, f1_c = models_dict[cmp_model]
                x = ri_cmp  if metric == "r" else f1_c
                y = ri_perm if metric == "r" else f1_p
                all_vals.extend([x, y])
                points.append((x, y, color, src_shapes[source_label],
                                crop_label, short_name))

    if not all_vals:
        plt.close()
        return

    # Plot dots
    for x, y, color, marker, *_ in points:
        ax.scatter(x, y, color=color, marker=marker,
                   s=95, zorder=3, linewidths=0.8, edgecolors="white")

    # Individual label for every dot — alternate offset by color to reduce overlap
    label_offsets = {"PyAEZ": (-24, 8), "No-thaw": (-5, -14)}
    for x, y, color, marker, label, short_name in points:
        dx, dy = label_offsets.get(short_name, (6, 5))
        ax.annotate(label, (x, y),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=10, color=color, alpha=0.9)

    vmin = 0
    vmax = 0.35
    ax.plot([vmin, vmax], [vmin, vmax], "k--",
            alpha=0.35, linewidth=1.0, zorder=1)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.fill_between(xlim, xlim, [ylim[1], ylim[1]],
                    color="#2166ac", alpha=0.04, zorder=0)
    ax.text(xlim[0] + 0.003, ylim[1] - 0.003,
            "PermaGAEZ (thaw) wins ↑", fontsize=12,
            color="#2166ac", alpha=0.7, va="top")
    ax.text(xlim[1] - 0.003, ylim[0] + 0.003,
            "Comparison model wins →",
            fontsize=12, color="grey", alpha=0.6, ha="right")

    from matplotlib.lines import Line2D
    legend_elements = []
    # Model comparison — squares
    for _, color, short_name in cmp_models:
        legend_elements.append(
            Line2D([0], [0], marker="s", color="w",
                   markerfacecolor=color, markersize=9,
                   label=f"vs {short_name}"))
    # Source — triangle and circle
    for source_label in sources:
        if any(source_label in d["sources_data"] for d in all_crop_data):
            legend_elements.append(
                Line2D([0], [0], marker=src_shapes[source_label],
                       color="w", markerfacecolor="grey",
                       markersize=9, label=src_labels[source_label]))
    ax.legend(handles=legend_elements, fontsize=12,
              loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=2, framealpha=0.85, edgecolor="none")

    ax.set_xlabel(f"Comparison model Spearman r", fontsize=12)
    ax.set_ylabel(f"PermaGAEZ (thaw) Spearman r", fontsize=12)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved scatter comparison: {os.path.basename(out_path)}")


def plot_scatter_comparison_single(all_crop_data, metric, out_path):
    """
    Original single-panel scatter: Permafrost vs Original.
    No-Thaw r annotated in italic grey below each crop point (Option D).
    """
    sources      = ["Monfreda (2000)", "FAO Production"]
    src_colors   = {"Monfreda (2000)": "#2166ac",
                    "FAO Production":  "#d6604d"}
    src_labels   = {"Monfreda (2000)": "EarthSTAT (1979–1998)",
                    "FAO Production":  "FAO (1999–2018)"}
    crop_markers = ["o", "s", "^", "D", "v", "P"]
    metric_label = "Spearman r" if metric == "r" else "F1 Score"

    fig, ax = plt.subplots(figsize=(6.0, 5.5))

    all_vals = []
    for c_idx, crop_data in enumerate(all_crop_data):
        crop_label   = crop_data["crop_label"]
        sources_data = crop_data["sources_data"]
        marker       = crop_markers[c_idx % len(crop_markers)]

        for source_label in sources:
            if source_label not in sources_data:
                continue
            models_dict = sources_data[source_label]["models_dict"]
            if ("Permafrost Model" not in models_dict or
                    "Original Model" not in models_dict):
                continue

            _, ri_perm, _, _, _, f1_p = models_dict["Permafrost Model"]
            _, ri_orig, _, _, _, f1_o = models_dict["Original Model"]
            x = ri_orig if metric == "r" else f1_o
            y = ri_perm if metric == "r" else f1_p
            all_vals.extend([x, y])

            color = src_colors[source_label]
            ax.scatter(x, y, color=color, marker=marker,
                       s=90, zorder=3, linewidths=0.8,
                       edgecolors="white")

            # Crop name
            ax.annotate(crop_label, (x, y),
                        xytext=(5, 4), textcoords="offset points",
                        fontsize=7.5, color=color, alpha=0.9)

            # Option D: No-Thaw r in italic grey below crop name
            if "No-Thaw Model" in models_dict:
                _, ri_nt, _, _, _, f1_nt = models_dict["No-Thaw Model"]
                r_nt = ri_nt if metric == "r" else f1_nt
                ax.annotate(f"No-Thaw: {r_nt:.3f}", (x, y),
                            xytext=(5, -9), textcoords="offset points",
                            fontsize=6.5, color="grey",
                            alpha=0.75, style="italic")

    if not all_vals:
        plt.close()
        return

    vmin = min(all_vals) - 0.02
    vmax = max(all_vals) + 0.02
    ax.plot([vmin, vmax], [vmin, vmax], "k--",
            alpha=0.35, linewidth=1.0, zorder=1)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.fill_between(xlim, xlim, [ylim[1], ylim[1]],
                    color="#2166ac", alpha=0.04, zorder=0)
    ax.text(xlim[0] + 0.003, ylim[1] - 0.003,
            "Permafrost wins ↑", fontsize=8.5,
            color="#2166ac", alpha=0.7, va="top")
    ax.text(xlim[1] - 0.003, ylim[0] + 0.003,
            "Original wins →", fontsize=8.5,
            color="#d6604d", alpha=0.7, ha="right")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=src_colors[s], label=src_labels[s])
        for s in sources if any(s in d["sources_data"] for d in all_crop_data)
    ]
    # Add No-Thaw annotation note
    from matplotlib.lines import Line2D
    legend_elements.append(
        Line2D([0], [0], color="none",
               label="italic = No-Thaw r"))
    ax.legend(handles=legend_elements, fontsize=8,
              loc="lower right", framealpha=0.85, edgecolor="none")

    ax.set_xlabel(f"Original Model — {metric_label}", fontsize=11)
    ax.set_ylabel(f"Permafrost Model — {metric_label}", fontsize=11)
    ax.set_title(f"Permafrost vs Original — {metric_label}\n"
                 f"(points above diagonal = Permafrost performs better;\n"
                 f"italic annotation = No-Thaw r)",
                 fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.text(0.5, -0.02,
             "†FAO Production: partial circularity with GAEZ methodology — see text",
             ha="center", fontsize=8, color="grey", style="italic")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved single scatter: {os.path.basename(out_path)}")


def plot_dot_plot(all_crop_data, metric, out_path):
    """
    Option A: Cleveland dot plot.
    Rows = crops × sources. Three dots per row (one per model).
    Color = model. Makes it easy to see which model wins per crop.
    """
    models      = ["Permafrost Model", "No-Thaw Model", "Original Model"]
    mod_colors  = {"Permafrost Model": "#2166ac",
                   "No-Thaw Model":    "#4dac26",
                   "Original Model":   "#d6604d"}
    mod_short   = {"Permafrost Model": "Permafrost",
                   "No-Thaw Model":    "No-Thaw",
                   "Original Model":   "Original"}
    sources     = ["Monfreda (2000)", "FAO Production"]
    src_labels  = {"Monfreda (2000)": "Monfreda",
                   "FAO Production":  "FAO†"}
    metric_label = "Spearman r" if metric == "r" else "F1 Score"

    # Build rows: (row_label, {model: value})
    rows = []
    for crop_data in all_crop_data:
        crop_label   = crop_data["crop_label"]
        sources_data = crop_data["sources_data"]
        for source_label in sources:
            if source_label not in sources_data:
                continue
            models_dict = sources_data[source_label]["models_dict"]
            row_vals = {}
            for model_label in models:
                if model_label not in models_dict:
                    continue
                _, ri, _, prec, rec, f1 = models_dict[model_label]
                row_vals[model_label] = ri if metric == "r" else f1
            if row_vals:
                rows.append({
                    "label":  f"{crop_label}\n({src_labels[source_label]})",
                    "vals":   row_vals,
                })

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(6.5, 0.55 * n_rows + 1.5))

    all_vals = []
    for r_idx, row in enumerate(rows):
        y = n_rows - r_idx - 1  # top-to-bottom
        vals = row["vals"]

        # Horizontal connector line
        if len(vals) >= 2:
            v_list = list(vals.values())
            ax.hlines(y, min(v_list), max(v_list),
                      color="grey", linewidth=1.0, alpha=0.4, zorder=1)

        for model_label, val in vals.items():
            color = mod_colors.get(model_label, "grey")
            ax.scatter(val, y, color=color, s=80, zorder=3,
                       linewidths=0.5, edgecolors="white")
            all_vals.append(val)

        # Best model annotation
        if vals:
            best_model = max(vals, key=vals.get)
            best_val   = vals[best_model]
            ax.annotate(f"{best_val:.3f}",
                        (best_val, y),
                        xytext=(4, 0), textcoords="offset points",
                        fontsize=7, color=mod_colors.get(best_model, "grey"),
                        va="center", fontweight="bold")

    # Y-axis labels
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([r["label"] for r in reversed(rows)],
                       fontsize=9)
    ax.set_ylim(-0.5, n_rows - 0.5)

    if all_vals:
        margin = (max(all_vals) - min(all_vals)) * 0.15 or 0.05
        ax.set_xlim(min(all_vals) - margin, max(all_vals) + margin * 3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=mod_colors[m],
               markersize=8, label=mod_short[m])
        for m in models
    ]
    ax.legend(handles=legend_elements, fontsize=9,
              loc="lower right", framealpha=0.85, edgecolor="none")

    ax.set_xlabel(metric_label, fontsize=11)
    ax.set_title(f"Model Validation — {metric_label} by Crop and Source\n"
                 f"(annotated value = best model per row)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    if any("FAO" in r["label"] for r in rows):
        fig.text(0.5, -0.02,
                 "†FAO Production: partial circularity with GAEZ methodology — see text",
                 ha="center", fontsize=8, color="grey", style="italic")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved dot plot: {os.path.basename(out_path)}")


def plot_pixel_scatter(all_crop_data, source_label, out_path):
    """
    Scatter plot: reference production class (x) vs model suitability class (y).
    One point per pixel. Alpha handles overplotting.
    One panel per crop, one figure per source.
    1:1 diagonal = perfect agreement.
    Color = model.
    """
    model_colors = {"Permafrost Model": "#2166ac",
                    "No-Thaw Model":    "#4dac26",
                    "Original Model":   "#d6604d"}
    mod_short    = {"Permafrost Model": "Permafrost",
                    "No-Thaw Model":    "No-Thaw",
                    "Original Model":   "Original"}
    src_title    = {"Monfreda (2000)": "Monfreda et al. (2000)  ·  1979–1998 model period",
                    "FAO Production":  "FAO GAEZ v5 Production  ·  1999–2018 model period\n"
                                       "(note: partial circularity — see text)"}

    # Filter to crops that have this source
    crop_list = [d for d in all_crop_data
                 if source_label in d["sources_data"]]
    if not crop_list:
        print(f"  No data for {source_label} — skipping pixel scatter")
        return

    n_crops = len(crop_list)
    ncols   = min(n_crops, 3)
    nrows   = math.ceil(n_crops / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.0 * ncols, 3.8 * nrows),
                             squeeze=False)
    fig.suptitle(f"Reference Production Class vs Model Suitability Class\n"
                 f"{src_title.get(source_label, source_label)}",
                 fontsize=11, fontweight="bold")

    ref_class_vals = [1, 2, 3, 4, 5]

    for idx, crop_data in enumerate(crop_list):
        row = idx // ncols
        col = idx % ncols
        ax  = axes[row][col]

        crop_label   = crop_data["crop_label"]
        keep         = crop_data["keep"]
        source_info  = crop_data["sources_data"][source_label]
        ref_masked   = source_info["ref_masked"]
        models_dict  = source_info["models_dict"]

        # Jitter for visibility (small random offset)
        rng = np.random.default_rng(seed=42)

        for model_label, (model_arr, ri, pi, prec, rec, f1) in models_dict.items():
            color = model_colors.get(model_label, "grey")
            short = mod_short.get(model_label, model_label)

            valid = keep & np.isfinite(ref_masked) & np.isfinite(model_arr)
            x = ref_masked[valid]
            y = model_arr[valid]

            # Jitter so overlapping points are visible
            jitter_x = rng.uniform(-0.18, 0.18, size=len(x))
            jitter_y = rng.uniform(-0.18, 0.18, size=len(y))

            # Compute pixel counts per class for alpha scaling
            # More pixels = more transparent to avoid solid blobs
            n_total = len(x)
            alpha   = max(0.03, min(0.25, 200 / n_total))

            ax.scatter(x + jitter_x, y + jitter_y,
                       color=color, alpha=alpha, s=8,
                       linewidths=0, rasterized=True)
            # Invisible point for legend
            ax.scatter([], [], color=color, s=30, alpha=0.9,
                       label=f"{short} (r={ri:.3f})")

        # 1:1 diagonal
        ax.plot([0.5, 5.5], [0.5, 5.5], "k--",
                alpha=0.4, linewidth=1.2, zorder=5)

        # Mean per reference class (overlay on scatter)
        for model_label, (model_arr, ri, pi, prec, rec, f1) in models_dict.items():
            color = model_colors.get(model_label, "grey")
            valid = keep & np.isfinite(ref_masked) & np.isfinite(model_arr)
            x = ref_masked[valid]
            y = model_arr[valid]
            means = [np.mean(y[np.round(x).astype(int) == c])
                     if np.sum(np.round(x).astype(int) == c) > 0 else np.nan
                     for c in ref_class_vals]
            vp = [i for i, m in enumerate(means) if not np.isnan(m)]
            if vp:
                ax.plot([ref_class_vals[i] for i in vp],
                        [means[i] for i in vp],
                        color=color, linewidth=2.0,
                        marker="o", markersize=5, zorder=6,
                        alpha=0.9)

        # Suitability threshold
        ax.axhline(y=2, color="grey", linestyle="--",
                   alpha=0.35, linewidth=0.8)

        ax.set_title(crop_label, fontsize=11, fontweight="bold", pad=5)
        ax.set_xlim(0.5, 5.5)
        ax.set_ylim(0, 5.5)
        ax.set_xticks(ref_class_vals)
        ax.set_yticks(ref_class_vals)
        ax.set_xlabel("Reference Production Class", fontsize=9)
        ax.set_ylabel("Model Suitability Class", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=7.5, loc="upper left",
                  framealpha=0.85, edgecolor="none")

    # Hide unused panels
    for idx in range(len(crop_list), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved pixel scatter: {os.path.basename(out_path)}")



def plot_validation_figure(ref_masked, keep, crop_label, source_label,
                           models_dict, out_path):
    """
    Validation figure: mean model suitability class per reference production
    class, one line per model with error bars (±1 SD).

    If reference class 1 dominates, also plots a zoomed inset for classes 2–5
    to make the upper-class trend visible.

    models_dict: {model_label: (model_class_array, spearman_r, spearman_p,
                                 precision, recall, f1)}
    """
    colors     = {"Permafrost Model": "#2166ac",
                  "No-Thaw Model":    "#4dac26",
                  "Original Model":   "#d6604d"}
    markers    = {"Permafrost Model": "o",
                  "No-Thaw Model":    "s",
                  "Original Model":   "^"}

    ref_class_vals = [1, 2, 3, 4, 5]

    # Compute mean ± SD of model suitability per reference class
    model_stats = {}
    for model_label, (model_arr, ri, pi, prec, rec, f1) in models_dict.items():
        valid = keep & np.isfinite(ref_masked) & np.isfinite(model_arr)
        a = ref_masked[valid]
        m = model_arr[valid]
        means, sds, ns = [], [], []
        for c in ref_class_vals:
            subset = m[np.round(a).astype(int) == c]
            means.append(np.mean(subset) if len(subset) > 0 else np.nan)
            sds.append(np.std(subset)   if len(subset) > 0 else np.nan)
            ns.append(len(subset))
        model_stats[model_label] = {"means": means, "sds": sds,
                                     "ns": ns, "r": ri, "p": pi,
                                     "f1": f1, "prec": prec, "rec": rec}

    # Count pixels per reference class (same across all models)
    valid0 = keep & np.isfinite(ref_masked)
    a0     = ref_masked[valid0]
    ns_ref = [int(np.sum(np.round(a0).astype(int) == c))
              for c in ref_class_vals]

    # Determine if class 1 dominates (>80% of pixels)
    class1_frac  = ns_ref[0] / sum(ns_ref) if sum(ns_ref) > 0 else 0
    has_upper    = any(ns_ref[i] > 0 for i in range(1, 5))
    show_inset   = class1_frac > 0.8 and has_upper

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f"{crop_label} — {source_label}\n"
                 f"Mean Model Suitability Class by Reference Production Class",
                 fontsize=12, fontweight="bold")

    for model_label, stats in model_stats.items():
        color  = colors.get(model_label, "grey")
        marker = markers.get(model_label, "o")
        means  = np.array(stats["means"], dtype=float)
        sds    = np.array(stats["sds"],   dtype=float)
        r, p   = stats["r"], stats["p"]
        f1_val = stats["f1"]

        ax.errorbar(ref_class_vals, means, yerr=sds,
                    label=f"{model_label.replace(' Model','')} "
                          f"(r={r:.3f}, F1={f1_val:.3f})",
                    color=color, marker=marker, markersize=7,
                    linewidth=1.8, capsize=4, capthick=1.5,
                    linestyle="-")

    # Reference line at class 2 (suitability threshold)
    ax.axhline(y=2, color="grey", linestyle="--", alpha=0.6,
               linewidth=1, label="Suitable threshold (class 2)")

    # Diagonal reference (perfect agreement)
    ax.plot(ref_class_vals, ref_class_vals, color="black",
            linestyle=":", alpha=0.4, linewidth=1,
            label="Perfect agreement")

    # Pixel count annotations on x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ref_class_vals)
    ax2.set_xticklabels([f"n={n}" for n in ns_ref], fontsize=8,
                         color="grey")
    ax2.set_xlabel("Pixels per reference class", fontsize=9, color="grey")

    ax.set_xlabel("Reference Production Class", fontsize=11)
    ax.set_ylabel("Mean Model Suitability Class (± 1 SD)", fontsize=11)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0, 5.5)
    ax.set_xticks(ref_class_vals)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # Inset for classes 2–5 if class 1 dominates
    if show_inset:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ax_inset = inset_axes(ax, width="45%", height="45%", loc="lower right")
        for model_label, stats in model_stats.items():
            color  = colors.get(model_label, "grey")
            marker = markers.get(model_label, "o")
            means  = np.array(stats["means"][1:], dtype=float)  # classes 2–5
            sds    = np.array(stats["sds"][1:],   dtype=float)
            ax_inset.errorbar([2, 3, 4, 5], means, yerr=sds,
                              color=color, marker=marker, markersize=5,
                              linewidth=1.5, capsize=3, linestyle="-")
        ax_inset.axhline(y=2, color="grey", linestyle="--",
                         alpha=0.6, linewidth=0.8)
        ax_inset.set_xlim(1.5, 5.5)
        ax_inset.set_ylim(0, 5.5)
        ax_inset.set_xticks([2, 3, 4, 5])
        ax_inset.set_xlabel("Ref class 2–5", fontsize=7)
        ax_inset.set_ylabel("Model class", fontsize=7)
        ax_inset.tick_params(labelsize=7)
        ax_inset.set_title("Classes 2–5 (zoomed)", fontsize=7)
        ax_inset.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {os.path.basename(out_path)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ref_geo, ref_shape = get_ref_info()
    print("Building study area mask...")
    keep     = build_lake_mask(ref_geo, ref_shape)
    pixel_ha = get_pixel_ha(ref_geo, ref_shape)
    print(f"  Valid pixels: {keep.sum()}, pixel area: {pixel_ha:.0f} ha")

    all_rows      = []
    all_crop_data = []  # for combined figure

    for crop_label, model_folder, fao_file, mrf_folder in CROPS:
        print(f"\n{'='*60}\n  {crop_label}\n{'='*60}")

        sources = []

        if fao_file is not None:
            fao_path = os.path.join(FAO_PROD_DIR, fao_file)
            if os.path.exists(fao_path):
                density = production_to_output_density(
                    fao_path, 1_000_000.0, pixel_ha, ref_geo, ref_shape)
                sources.append(("FAO Production", density,
                                 AVG_YEARS_RECENT, set()))
            else:
                print(f"  SKIP FAO: not found — {fao_path}")

        if mrf_folder is not None:
            mrf_path = os.path.join(MRF_BASE, mrf_folder,
                                    f"{mrf_folder}_Production.tif")
            if os.path.exists(mrf_path):
                density = production_to_output_density(
                    mrf_path, 1_000.0, pixel_ha, ref_geo, ref_shape)
                sources.append(("Monfreda (2000)", density,
                                 AVG_YEARS_BASELINE, MONFREDA_SKIP_MODELS))
            else:
                print(f"  SKIP Monfreda: not found — {mrf_path}")

        for source_label, density, avg_years, skip_models in sources:
            period = f"{avg_years[0]}–{avg_years[-1]}"
            print(f"\n  [{source_label}] period={period}")

            ref_class, boundaries = classify_production(density, keep)
            if boundaries is None:
                continue

            n_nonzero  = int(np.sum(keep & (density > 0)))
            ref_masked = np.where(keep, ref_class, np.nan)
            print(f"  Non-zero production pixels: {n_nonzero}")

            models_dict = {}

            for model_label, model_dir in MODELS:
                if model_label in skip_models:
                    print(f"  [{model_label}] skipped")
                    continue

                try:
                    model_class = load_mean_suitability(
                        model_dir, model_folder, avg_years, ref_geo, ref_shape)
                except FileNotFoundError as e:
                    print(f"  [{model_label}] ERROR: {e}")
                    continue

                model_masked = np.where(keep, model_class, np.nan)
                valid = keep & np.isfinite(ref_masked) & np.isfinite(model_masked)
                a = ref_masked[valid]
                m = model_masked[valid]

                if len(a) < 2:
                    print(f"  [{model_label}] insufficient pixels")
                    continue

                # Spearman r
                r, p = stats.spearmanr(a, m)
                n    = int(len(a))

                # Binary metrics: class >= 2 = suitable
                # Reference class >= 2 = actual positive
                # Model class >= 2 = predicted positive
                ref_bin   = (a >= 2).astype(int)
                model_bin = (m >= 2).astype(int)
                tp = int(np.sum((ref_bin == 1) & (model_bin == 1)))
                fp = int(np.sum((ref_bin == 0) & (model_bin == 1)))
                fn = int(np.sum((ref_bin == 1) & (model_bin == 0)))
                tn = int(np.sum((ref_bin == 0) & (model_bin == 0)))
                precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
                recall    = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
                f1        = round(2 * precision * recall / (precision + recall), 4) \
                            if (precision + recall) > 0 else 0.0

                print(f"  [{model_label}] Spearman r={r:.4f}, p={p:.4f}, n={n} | "
                      f"P={precision:.3f} R={recall:.3f} F1={f1:.3f} "
                      f"TP={tp} FP={fp} FN={fn} TN={tn}")

                models_dict[model_label] = (model_masked, r, p,
                                             precision, recall, f1)

                all_rows.append({
                    "Crop":            crop_label,
                    "Source":          source_label,
                    "Model Period":    period,
                    "Model":           model_label,
                    "N pixels":        n,
                    "Spearman r":      round(r, 4),
                    "Spearman p":      round(p, 4),
                    "Precision":       precision,
                    "Recall":          recall,
                    "F1":              f1,
                    "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                    "Ref max (kg/ha)": round(boundaries["y_max"], 3),
                    "N non-zero ref":  n_nonzero,
                })

            # Collect data for combined figure (both sources)
                crop_entry = next(
                    (d for d in all_crop_data
                     if d["crop_label"] == crop_label), None)
                if crop_entry is None:
                    crop_entry = {"crop_label": crop_label,
                                  "keep": keep,
                                  "sources_data": {}}
                    all_crop_data.append(crop_entry)
                crop_entry["sources_data"][source_label] = {
                    "ref_masked":  ref_masked,
                    "models_dict": models_dict,
                }

    # Scatter comparison figure
    if all_crop_data:
        plot_scatter_comparison(
            all_crop_data, metric="r",
            out_path=os.path.join(OUTPUT_DIR, "scatter_comparison_r.png"))

    # Save CSV
    if all_rows:
        df   = pd.DataFrame(all_rows)
        cols = ["Crop", "Source", "Model Period", "Model", "N pixels",
                "Spearman r", "Spearman p",
                "Precision", "Recall", "F1",
                "TP", "FP", "FN", "TN",
                "Ref max (kg/ha)", "N non-zero ref"]
        df[cols].to_csv(
            os.path.join(OUTPUT_DIR, "suitability_class_comparison.csv"),
            index=False)
        print(f"\n  Saved: {OUTPUT_DIR}/suitability_class_comparison.csv")
        print("\n" + df[["Crop","Source","Model",
                          "Spearman r","Spearman p",
                          "Precision","Recall","F1"]].to_string(index=False))

    print(f"\n{'='*60}\n  Complete.\n{'='*60}\n")


if __name__ == "__main__":
    main()