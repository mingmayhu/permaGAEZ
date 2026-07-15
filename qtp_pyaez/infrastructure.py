"""
PCF Land Cover Risk Figure
Reads pcf_risk_by_landcover.csv and plots:
  Panel A: physically exposed km² (stacked by risk class) with total km² + % of LC area label
  Panel B: risk class composition (% of exposed area)
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import matplotlib.font_manager as fm
import os

IN_CSV  = '/Users/ming-mayhu/Desktop/pcf_risk_by_landcover.csv'
OUT_FIG = '/Users/ming-mayhu/Desktop/pcf_risk_by_landcover.png'

HELVETICA_BOLD = '/Users/ming-mayhu/Library/Fonts/HelveticaNeueLTPro-Bd.otf'

RISK_PLOT = ['Low (0-50%)', 'Medium (50-80%)', 'High (>80%)']
RISK_COLOURS = {
    'Low (0-50%)':     '#92c5de',
    'Medium (50-80%)': '#f4a582',
    'High (>80%)':     '#ca0020',
}

# ── Load ──────────────────────────────────────────────────────────────────────

df = pd.read_csv(IN_CSV)
df = df[df['Risk class'] != 'None (0%)']

# Panel A: absolute km²
pivot_km2 = df.pivot_table(
    index='Land cover', columns='Risk class',
    values='Physically exposed (km2)', aggfunc='sum'
)[RISK_PLOT]

# Total exposed and % of LC area per land cover type
lc_totals = df.groupby('Land cover').agg(
    total_lc_area=('Total LC area (km2)', 'first'),
    total_exposed=('Physically exposed (km2)', 'sum')
).reset_index()
lc_totals['pct_of_lc'] = 100 * lc_totals['total_exposed'] / lc_totals['total_lc_area']

# Sort by total exposed ascending
pivot_km2['_total'] = pivot_km2.sum(axis=1)
pivot_km2 = pivot_km2.sort_values('_total', ascending=True).drop(columns='_total')
lc_order = pivot_km2.index.tolist()

# Panel B: % of exposed area composition
pivot_pct = df.pivot_table(
    index='Land cover', columns='Risk class',
    values='Exposed (% of LC exposed)', aggfunc='sum'
)[RISK_PLOT]
pivot_pct = pivot_pct.loc[lc_order]

# ── Font ──────────────────────────────────────────────────────────────────────

if os.path.exists(HELVETICA_BOLD):
    prop = fm.FontProperties(fname=HELVETICA_BOLD)
    matplotlib.rcParams['font.family'] = prop.get_name()

import seaborn as sns
sns.set_style('ticks')
plt.rcParams.update({'font.size': 9})

# ── Figure ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'wspace': 0.5})

# ── Panel A: km² with end labels ─────────────────────────────────────────────

ax = axes[0]
lefts = np.zeros(len(lc_order))
for rc in RISK_PLOT:
    vals = pivot_km2[rc].values
    ax.barh(lc_order, vals, left=lefts,
            color=RISK_COLOURS[rc], edgecolor='white', linewidth=0.4,
            height=0.65)
    lefts += vals

# End label: total km² and % of LC area
lc_totals_idx = lc_totals.set_index('Land cover')
x_max = lefts.max()
for i, lc in enumerate(lc_order):
    total_km2 = lefts[i]
    pct       = lc_totals_idx.loc[lc, 'pct_of_lc']
    ax.text(total_km2 + x_max * 0.01, i,
            f'{total_km2:,.0f} km²  ({pct:.1f}%)',
            ha='left', va='center', fontsize=7.5)

ax.set_xlabel('Physically exposed area (km²)', fontsize=9)
ax.set_title('(a)  Physically exposed area by land cover',
             fontsize=9, loc='left', pad=6)
ax.set_xlim(0, x_max * 1.45)
ax.tick_params(labelsize=8)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
sns.despine(ax=ax, top=True, right=True)

# ── Panel B: % composition ────────────────────────────────────────────────────

ax2 = axes[1]
lefts2 = np.zeros(len(lc_order))
for rc in RISK_PLOT:
    vals = pivot_pct[rc].fillna(0).values
    ax2.barh(lc_order, vals, left=lefts2,
             color=RISK_COLOURS[rc], edgecolor='white', linewidth=0.4,
             height=0.65)
    lefts2 += vals

ax2.set_xlabel('Risk class composition (% of exposed area)', fontsize=9)
ax2.set_xlim(0, 100)
ax2.set_title('(b)  Risk class breakdown of exposed area',
              fontsize=9, loc='left', pad=6)
ax2.tick_params(labelsize=8)
ax2.set_yticklabels([])
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
sns.despine(ax=ax2, top=True, right=True)

# ── Legend ────────────────────────────────────────────────────────────────────

legend_handles = [
    Patch(facecolor=RISK_COLOURS[rc], edgecolor='#888', linewidth=0.5, label=rc)
    for rc in RISK_PLOT
]
fig.legend(handles=legend_handles,
           title='PCF risk class', title_fontsize=8, fontsize=8,
           loc='lower center', ncol=3,
           bbox_to_anchor=(0.5, -0.06), frameon=False)

plt.savefig(OUT_FIG, dpi=300, bbox_inches='tight')
print(f'Figure saved to: {OUT_FIG}')
plt.show()