"""
LSC-specific GSEA Lollipop Plot with User-Selected Pathways

Creates publication-ready themed lollipop plot with:
- User-selected pathways organized by biological themes
- X-axis: -log10(FDR) for significance
- Dot size: abs(NES) for effect size magnitude
- Shortened pathway names for readability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Configuration
RESULTS_DIR = 'Outputs/gsea_results'
OUTPUT_FILE = 'Outputs/gsea_results/lsc_themed_lollipop.png'
FDR_THRESHOLD = 0.25

# User-selected pathways organized by theme
SELECTED_PATHWAYS = {
    'Cell Cycle & Proliferation': {
        'pathways': [
            'G2-M Checkpoint',
            'DNA-templated DNA Replication Maintenance Of Fidelity (GO:0045005)',
            'Negative Regulation Of Cell Cycle (GO:0045786)',
        ],
        'color': '#F39C12'
    },
    'Apoptosis & Survival': {
        'pathways': [
            'p53 Pathway',
            'Apoptosis',
            'Necroptosis',
            'Negative Regulation Of Cell Death (GO:0060548)',
        ],
        'color': '#8E44AD'
    },
    'Differentiation': {
        'pathways': [
            'Leukocyte transendothelial migration',
            'Epithelial Cell Differentiation (GO:0030855)',
            'Myeloid Leukocyte Differentiation (GO:0002573)',
        ],
        'color': '#16A085'
    },
    'Metabolism': {
        'pathways': [
            'Oxidative Phosphorylation',  # Hallmark
            'Oxidative phosphorylation',  # KEGG
            'Cellular Respiration (GO:0045333)',
        ],
        'color': '#3498DB'
    },
    'Oncogenic Signaling': {
        'pathways': [
            'Myc Targets V1',
            'Proteasome',
            'PI3K/AKT/mTOR  Signaling',
            'Ras signaling pathway',
            'KRAS Signaling Dn',
        ],
        'color': '#E67E22'
    },
}

# Shortened pathway names for display
PATHWAY_NAME_MAP = {
    # Cell Cycle
    'G2-M Checkpoint': 'G2-M Checkpoint',
    'DNA-templated DNA Replication Maintenance Of Fidelity (GO:0045005)': 'DNA Replication Fidelity',
    'Negative Regulation Of Cell Cycle (GO:0045786)': 'Cell Cycle Inhibition',

    # Apoptosis
    'p53 Pathway': 'p53 Pathway',
    'Apoptosis': 'Apoptosis',
    'Necroptosis': 'Necroptosis',
    'Negative Regulation Of Cell Death (GO:0060548)': 'Cell Death Inhibition',

    # Differentiation
    'Leukocyte transendothelial migration': 'Leukocyte Migration',
    'Epithelial Cell Differentiation (GO:0030855)': 'Epithelial Differentiation',
    'Myeloid Leukocyte Differentiation (GO:0002573)': 'Myeloid Differentiation',

    # Metabolism
    'Oxidative Phosphorylation': 'OXPHOS (Hallmark)',
    'Oxidative phosphorylation': 'OXPHOS (KEGG)',
    'Cellular Respiration (GO:0045333)': 'Cellular Respiration',

    # Oncogenic Signaling
    'Myc Targets V1': 'MYC Targets',
    'Proteasome': 'Proteasome',
    'PI3K/AKT/mTOR  Signaling': 'PI3K/AKT/mTOR',
    'Ras signaling pathway': 'RAS Signaling',
    'KRAS Signaling Dn': 'KRAS Signaling Down',
}


def load_gsea_results(results_dir):
    """Load all GSEA results from CSV files."""
    results_path = Path(results_dir)
    all_results = []

    for db_file, db_name in [('gsea_hallmark_results.csv', 'Hallmark'),
                              ('gsea_gobp_results.csv', 'GO_BP'),
                              ('gsea_kegg_results.csv', 'KEGG')]:
        file_path = results_path / db_file
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['FDR q-val'] = pd.to_numeric(df['FDR q-val'], errors='coerce')
            df['Database'] = db_name
            all_results.append(df)

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()


def select_pathways(results_df):
    """Select user-specified pathways and organize by theme."""
    theme_data = {}

    for theme, info in SELECTED_PATHWAYS.items():
        theme_pathways = []

        for pathway_name in info['pathways']:
            # Find this pathway in results
            match = results_df[results_df['Term'] == pathway_name]

            if len(match) > 0:
                row = match.iloc[0]

                # Get shortened name
                short_name = PATHWAY_NAME_MAP.get(pathway_name, pathway_name)

                theme_pathways.append({
                    'Pathway': short_name,
                    'Original': pathway_name,
                    'NES': row['NES'],
                    'FDR': row['FDR q-val'],
                    'Database': row['Database'],
                    'abs_NES': abs(row['NES'])
                })
            else:
                print(f"WARNING: Pathway not found: {pathway_name}")

        if len(theme_pathways) > 0:
            theme_data[theme] = {
                'pathways': theme_pathways,
                'color': info['color']
            }

    return theme_data


def create_lollipop_plot(theme_data, output_file):
    """Create publication-ready lollipop plot."""

    # Prepare plot data
    plot_data = []
    theme_positions = {}
    current_y = 0

    for theme, info in theme_data.items():
        pathways = info['pathways']
        color = info['color']

        # Sort by NES within theme
        pathways_sorted = sorted(pathways, key=lambda x: x['NES'])

        theme_start = current_y
        for pathway_info in pathways_sorted:
            # Calculate -log10(FDR) and clip to max 4
            fdr = pathway_info['FDR']
            neg_log_fdr = -np.log10(max(fdr, 1e-10))
            neg_log_fdr_clipped = min(neg_log_fdr, 4.0)

            plot_data.append({
                'y': current_y,
                'NES': pathway_info['NES'],
                'neg_log_fdr': neg_log_fdr_clipped,
                'Pathway': pathway_info['Pathway'],
                'Theme': theme,
                'Color': color,
                'FDR': fdr,
                'Database': pathway_info['Database']
            })
            current_y += 1

        theme_positions[theme] = (theme_start, current_y - 1, color)
        current_y += 0.5  # Gap between themes

    if len(plot_data) == 0:
        print("No pathways to plot!")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(8, max(10, len(plot_data) * 0.45)))

    # Plot lollipops
    for item in plot_data:
        y = item['y']
        nes = item['NES']
        sig = item['neg_log_fdr']
        color = item['Color']

        # X-position: positive NES to right, negative to left
        x_pos = sig if nes > 0 else -sig

        # Line color based on direction
        line_color = '#E74C3C' if nes > 0 else '#3498DB'
        ax.plot([0, x_pos], [y, y], color=line_color, linewidth=2.5, alpha=0.5)

        # Dot size based on abs(NES)
        size = abs(nes) * 250
        ax.scatter(x_pos, y, s=size, color=color, alpha=0.85,
                  edgecolors='white', linewidths=2, zorder=3)

        # Add NES value as text label
        offset = -0.12 if nes > 0 else 0.15
        ha = 'left' if nes > 0 else 'right'
        ax.text(x_pos + offset, y, f'{nes:.2f}',
               va='center', ha=ha, fontsize=6, fontweight='bold')

    # Set pathway labels
    ax.set_yticks([item['y'] for item in plot_data])
    pathway_labels = [item['Pathway'] for item in plot_data]
    ax.set_yticklabels(pathway_labels, fontsize=10)

    # Vertical line at x=0
    ax.axvline(0, color='grey', linewidth=1.5, linestyle='--', alpha=0.6)

    # Add theme separators
    for theme, (start, end, color) in theme_positions.items():
        if end < len(plot_data) - 1:
            ax.axhline(end + 0.5, color='gray', linewidth=0.8,
                      linestyle='--', alpha=0.4)

    # Styling
    ax.set_xlabel('-log10(FDR q-value, clipped at max 4)', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle=':')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nLollipop plot saved to: {output_file}")


def print_summary(theme_data):
    """Print summary of selected pathways."""
    print("\n" + "="*100)
    print("SELECTED PATHWAYS FOR LOLLIPOP PLOT")
    print("="*100)

    total_pathways = 0
    for theme, info in theme_data.items():
        pathways = info['pathways']
        n_enriched = sum(1 for p in pathways if p['NES'] > 0)
        n_depleted = len(pathways) - n_enriched

        print(f"\n{theme} ({len(pathways)} pathways: {n_enriched}↑, {n_depleted}↓)")
        print("─" * 100)

        for p in sorted(pathways, key=lambda x: -x['NES']):
            direction = "↑" if p['NES'] > 0 else "↓"
            print(f"  {direction} {p['Pathway']:<45} | NES={p['NES']:>6.2f} | FDR={p['FDR']:.2e} | {p['Database']}")

        total_pathways += len(pathways)

    print("\n" + "="*100)
    print(f"TOTAL: {total_pathways} pathways selected")
    print("="*100 + "\n")


if __name__ == '__main__':
    # Load results
    results_df = load_gsea_results(RESULTS_DIR)
    print(f"Loaded {len(results_df)} total pathway results\n")

    # Select user-specified pathways
    theme_data = select_pathways(results_df)

    # Print summary
    print_summary(theme_data)

    # Create plot
    create_lollipop_plot(theme_data, OUTPUT_FILE)
