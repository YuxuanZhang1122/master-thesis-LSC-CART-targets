#!/usr/bin/env python3
"""
Four-quadrant decision plot for gene pair analysis results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
RESULTS_PATH = BASE_DIR / "Pair_search/statistical_interaction/results/positive_interactions.csv"
OUTPUT_PATH = BASE_DIR / "Pair_search/statistical_interaction/figures/decision_quadrant_plot.pdf"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(RESULTS_PATH)

# Use columns directly from CSV (already calculated in the analysis script)
# X-axis: Specificity gain (pair specificity ratio - anchor specificity)
df['x_axis'] = df['specificity_gain']

# Y-axis: LSPC coverage retention (pair / anchor) in percentage
df['y_axis'] = df['efficacy_retention'] * 100

# Color: Specificity ratio
df['color_val'] = df['specificity_ratio']

# Size: LSPC coverage percentage
df['size_val'] = df['lspc_coverage'] * 100

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))

# Create scatter plot
scatter = ax.scatter(
    df['x_axis'],
    df['y_axis'],
    c=df['color_val'],
    s=df['size_val'] * 5,
    cmap='plasma',
    alpha=0.7,
    edgecolors='black',
    linewidth=0.5
)

# Add colorbars and legends
cbar = plt.colorbar(scatter, ax=ax, label='Pair Specificity (LSPC/HSPC)', location='left')

# Create size legend
sizes = [20, 30, 40]
size_labels = ['20%', '30%', '40%']
legend_elements = [
    plt.scatter([], [], s=s*5, c='gray', alpha=0.7, edgecolors='black', linewidth=0.5)
    for s in sizes
]
legend1 = ax.legend(
    legend_elements,
    size_labels,
    title='Pair LSPC Coverage (%)',
    loc='lower left',
    frameon=True,
    fontsize=14
)
ax.add_artist(legend1)

# Add reference lines at x=2.0, y=50%
ref_x = 2.0
ref_y = 50.0

ax.axvline(ref_x, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)
ax.axhline(ref_y, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)

# Categorize pairs into quadrants
top_right = df[(df['x_axis'] >= ref_x) & (df['y_axis'] >= ref_y)]
bottom_left = df[(df['x_axis'] < ref_x) & (df['y_axis'] < ref_y)]
top_left = df[(df['x_axis'] < ref_x) & (df['y_axis'] >= ref_y)]
bottom_right = df[(df['x_axis'] >= ref_x) & (df['y_axis'] < ref_y)]

# Filter for specific pairs to highlight in other quadrants
highlight_pairs = [('EMB', 'HCST'), ('EMB', 'CD47'), ('CD9', 'CD99')]

def filter_highlight_pairs(quadrant_df):
    mask = pd.Series([False] * len(quadrant_df), index=quadrant_df.index)
    for g1, g2 in highlight_pairs:
        mask |= ((quadrant_df['gene1'] == g1) & (quadrant_df['gene2'] == g2))
        mask |= ((quadrant_df['gene1'] == g2) & (quadrant_df['gene2'] == g1))
    return quadrant_df[mask]

bottom_left_filtered = filter_highlight_pairs(bottom_left)
top_left_filtered = filter_highlight_pairs(top_left)
bottom_right_filtered = filter_highlight_pairs(bottom_right)

# Calculate y-offset based on axis range (about 2% of range)
ylim = ax.get_ylim()
y_offset = (ylim[1] - ylim[0]) * 0.02

# Top right: ALL pairs in BOLD
for _, row in top_right.iterrows():
    label = f"{row['gene1']} + {row['gene2']}"
    ax.text(
        row['x_axis'],
        row['y_axis'] + y_offset,
        label,
        fontsize=15,
        ha='center',
        va='bottom',
        fontweight='bold'
    )

# Bottom left: filtered pairs only
for _, row in bottom_left_filtered.iterrows():
    label = f"{row['gene1']} + {row['gene2']}"
    ax.text(
        row['x_axis'],
        row['y_axis'] + y_offset,
        label,
        fontsize=12,
        ha='center',
        va='bottom'
    )

# Top left: filtered pairs only
for _, row in top_left_filtered.iterrows():
    label = f"{row['gene1']} + {row['gene2']}"
    ax.text(
        row['x_axis'],
        row['y_axis'] + y_offset,
        label,
        fontsize=12,
        ha='center',
        va='bottom'
    )

# Bottom right: filtered pairs only
for _, row in bottom_right_filtered.iterrows():
    label = f"{row['gene1']} + {row['gene2']}"
    ax.text(
        row['x_axis'],
        row['y_axis'] + y_offset,
        label,
        fontsize=12,
        ha='center',
        va='bottom'
    )

# Labels and title
ax.set_xlabel('Specificity Improvement (Δ LSPC/HSPC Ratio)', fontsize=18)
ax.set_ylabel('Efficacy Retention (% of Anchor Coverage)', fontsize=18)
ax.set_title('Gene Pair Decision Plot: Efficacy vs Specificity Trade-offs', fontsize=21, fontweight='bold')

# Add quadrant labels
xlim = ax.get_xlim()
ylim = ax.get_ylim()

ax.text(
    ref_x + (xlim[1] - ref_x) * 0.5,
    ref_y + (ylim[1] - ref_y) * 0.5,
    'High Retention\nHigh Specificity',
    ha='center',
    va='center',
    fontsize=15,
    alpha=0.3,
    style='italic'
)

ax.text(
    ref_x - (ref_x - xlim[0]) * 0.5,
    ref_y + (ylim[1] - ref_y) * 0.5,
    'High Retention\nLow Specificity',
    ha='center',
    va='center',
    fontsize=15,
    alpha=0.3,
    style='italic'
)

ax.text(
    ref_x + (xlim[1] - ref_x) * 0.5,
    ref_y - (ref_y - ylim[0]) * 0.5,
    'Low Retention\nHigh Specificity',
    ha='center',
    va='center',
    fontsize=15,
    alpha=0.3,
    style='italic'
)

ax.text(
    ref_x - (ref_x - xlim[0]) * 0.5,
    ref_y - (ref_y - ylim[0]) * 0.5,
    'Low Retention\nLow Specificity',
    ha='center',
    va='center',
    fontsize=15,
    alpha=0.3,
    style='italic'
)


plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_PATH.with_suffix('.png'), dpi=300, bbox_inches='tight', transparent=True)

print(f"Saved: {OUTPUT_PATH}")
print(f"Saved: {OUTPUT_PATH.with_suffix('.png')}")
print(f"\nPlotted {len(df)} gene pairs")
print(f"Top right (all annotated, bold): {len(top_right)}")
print(f"Bottom left (filtered): {len(bottom_left_filtered)} / {len(bottom_left)}")
print(f"Top left (filtered): {len(top_left_filtered)} / {len(top_left)}")
print(f"Bottom right (filtered): {len(bottom_right_filtered)} / {len(bottom_right)}")
