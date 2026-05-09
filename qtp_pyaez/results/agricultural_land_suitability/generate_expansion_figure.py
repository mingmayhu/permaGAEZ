"""
Plot Expansion vs Intensification — Chapter 5
==============================================
Reads pre-computed CSVs and produces the expansion vs intensification figure:
  - Stacked % bar chart (intensification + expansion contribution)
  - Overall average bar shown darker, individual crops lighter

Input:
  ./results/agricultural_land_suitability/outputs/csv/expansion_vs_intensification.csv

Output:
  ./results/agricultural_land_suitability/outputs/expansion_vs_intensification.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
import seaborn as sns

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR = r'/Users/ming-mayhu/Desktop/毕业论文/qtp-pyaez/qtp_pyaez'
CSV_PATH = r'./results/agricultural_land_suitability/outputs/csv/expansion_vs_intensification.csv'
OUT_PATH = r'./results/agricultural_land_suitability/outputs/expansion_vs_intensification.png'

DPI = 300

os.chdir(WORK_DIR)

# ── Font setup ────────────────────────────────────────────────────────────────
FONT      = 'Helvetica'
BOLD_PATH = '/Users/ming-mayhu/Library/Fonts/Helvetica LT 75 Bold.ttf'
REG_PATH  = '/System/Library/Fonts/Helvetica.ttc'

# ── Seaborn theme ─────────────────────────────────────────────────────────────
sns.set_theme(
    style='ticks',
    rc={
        'font.family':       'sans-serif',
        'font.sans-serif':   [FONT],
        'xtick.direction':   'out',
        'ytick.direction':   'out',
        'xtick.major.size':  4,
        'ytick.major.size':  4,
        'axes.edgecolor':    '#000000',
        'axes.linewidth':    0.8,
    }
)

# Colours — light for individual crops, dark for overall
EXPAND_LIGHT  = "#fdd1d1ff"   # light pink — expansion, individual crops
INTENS_LIGHT  = "#e6cbeeff"   # light purple — intensification, individual crops
EXPAND_DARK   = "#ff9c9f"   # dark pink — expansion, overall
INTENS_DARK   = "#cca1d7"   # dark purple — intensification, overall

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

# Compute overall row
overall_row = pd.DataFrame([{
    'crop':                     'Overall',
    'pct_from_expansion':       df['pct_from_expansion'].mean(),
    'pct_from_intensification': df['pct_from_intensification'].mean(),
}])
df_plot = pd.concat([df[['crop', 'pct_from_expansion', 'pct_from_intensification']],
                     overall_row], ignore_index=True)

crop_labels = df_plot['crop'].tolist()
x           = np.arange(len(crop_labels))
is_overall  = np.array([c == 'Overall' for c in crop_labels])

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor('white')

fp_bold   = FontProperties(fname=BOLD_PATH, size=16)
fp_normal = FontProperties(fname=REG_PATH,  size=14)
fp_legend = FontProperties(fname=REG_PATH,  size=14)


# Expansion bars
expand_colors = [EXPAND_DARK if o else EXPAND_LIGHT for o in is_overall]
for i, (xi, val, col) in enumerate(zip(x, df_plot['pct_from_expansion'], expand_colors)):
    ax.bar(xi, val, color=col, alpha=0.9,
           edgecolor='white',
           linewidth=0.5)

# Intensification bars stacked on top
intens_colors = [INTENS_DARK if o else INTENS_LIGHT for o in is_overall]
for i, (xi, val, bot, col) in enumerate(zip(
        x, df_plot['pct_from_intensification'],
        df_plot['pct_from_expansion'], intens_colors)):
    ax.bar(xi, val, bottom=bot, color=col, alpha=0.9,
           edgecolor='white',
           linewidth=0.5)

# 50% reference line
ax.axhline(50, color='black', linewidth=0.8, linestyle='--', alpha=0.4)

# Axis formatting
ax.set_xticks(x)
ax.set_xticklabels(crop_labels, rotation=30, ha='right', fontsize=14)
ax.set_ylabel('% of Total Suitability Gain', fontsize=14, labelpad=6)
ax.set_ylim(0, 100)
ax.tick_params(which='major', labelsize=14, length=4,
               color='#000000', width=0.8)
ax.spines['left'].set_color('#000000')
ax.spines['bottom'].set_color('#000000')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# Legend — use light colours to represent all crops, dark for overall
handles = [
    Patch(color=EXPAND_DARK,   alpha=0.9, label='Expansion'),
    Patch(color=INTENS_DARK,   alpha=0.9, label='Intensification'),
]
leg = ax.legend(handles=handles,
                loc='upper center',
                bbox_to_anchor=(0.5, -0.25),
                bbox_transform=ax.transAxes,
                framealpha=0, edgecolor='none',
                borderpad=0.5, ncol=4)
for text in leg.get_texts():
    text.set_fontproperties(fp_legend)

ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['top'].set_color('#000000')
ax.spines['right'].set_color('#000000')

plt.tight_layout()
plt.subplots_adjust(bottom=0.25)
fig.savefig(OUT_PATH, dpi=DPI, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f'Saved to {OUT_PATH}')