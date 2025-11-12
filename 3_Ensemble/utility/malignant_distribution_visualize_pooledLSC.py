import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

sns.set_style('whitegrid')
plt.rcParams['font.size'] = 11

CLASSIFIED_DIR = Path('../pooledLSC/4visualization')
OUTPUT_DIR = CLASSIFIED_DIR / 'figures'
label_key = 'consensus_label_6votes' # consensus_label_5votes, _6votes
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

CELL_TYPES = ['HSC MPP', 'LMPP', 'Early GMP']

def order_datasets(dataset_list):
    """Order datasets"""
    # Define priority order
    priority_order = {
        'Ennis': 2,
        'Henrik': 7,
        'Naldini_V02': 4,
        'Naldini_V03': 3,
        'Petti': 5,
        'vanGalen':6,
        #'Setty': 1,
        'Mende': 1
    }

    def get_sort_key(dataset):
        # Extract base name (before first underscore if it's a compound ID)
        base = dataset.split('_')[0]

        # Handle Naldini_V0X cases
        if base == 'Naldini' and len(dataset.split('_')) > 1:
            if dataset.split('_')[1].startswith('V02'):
                base = 'Naldini_V02'
            elif dataset.split('_')[1].startswith('V03'):
                base = 'Naldini_V03'

        priority = priority_order.get(base, 50)
        return (priority, dataset)

    return sorted(dataset_list, key=get_sort_key)

def load_classified_data():
    """Load ensemble results from all datasets and concatenate"""
    all_adatas = []

    for dataset_dir in sorted(CLASSIFIED_DIR.glob('*')):
        if not dataset_dir.is_dir() or dataset_dir.name == 'figures':
            continue

        results_path = dataset_dir / 'ensemble_results.h5ad'
        if results_path.exists():
            adata = sc.read_h5ad(results_path)
            all_adatas.append(adata)

    if not all_adatas:
        return None

    combined = sc.concat(all_adatas, join='outer', index_unique='-')
    return combined

def calculate_malignancy_stats(adata, label_key):
    """Calculate malignancy distribution statistics by cell type and compound dataset ID"""
    rows = []

    consensus = adata.obs[label_key].values
    cell_types = adata.obs['predicted_CellType_Broad'].values
    sources = adata.obs['source_dataset'].values
    time_points = adata.obs['time_point'].values

    # Create compound dataset IDs
    compound_ids = sources

    # Overall stats per cell type
    for cell_type in CELL_TYPES:
        mask_ct = cell_types == cell_type
        n_total = mask_ct.sum()

        if n_total == 0:
            continue

        n_lspc = ((consensus == 'LSPC') & mask_ct).sum()
        n_hspc = ((consensus == 'HSPC') & mask_ct).sum()
        n_uncertain = ((consensus == 'uncertain') & mask_ct).sum()
        n_labeled = n_lspc + n_hspc
        pct_lspc = (n_lspc / n_labeled * 100) if n_labeled > 0 else 0

        rows.append({
            'CellType': cell_type,
            'Dataset': 'Overall',
            'Total': n_total,
            'LSPC': n_lspc,
            'HSPC': n_hspc,
            'Uncertain': n_uncertain,
            'Pct_LSPC': pct_lspc,
            'Pct_HSPC': 100 - pct_lspc
        })

        # Per compound dataset for this cell type
        for dataset_id in np.unique(compound_ids):
            mask = (cell_types == cell_type) & (compound_ids == dataset_id)
            n_total_ds = mask.sum()

            if n_total_ds == 0:
                continue

            n_lspc_ds = ((consensus == 'LSPC') & mask).sum()
            n_hspc_ds = ((consensus == 'HSPC') & mask).sum()
            n_uncertain_ds = ((consensus == 'uncertain') & mask).sum()
            n_labeled_ds = n_lspc_ds + n_hspc_ds
            pct_lspc_ds = (n_lspc_ds / n_labeled_ds * 100) if n_labeled_ds > 0 else 0

            rows.append({
                'CellType': cell_type,
                'Dataset': dataset_id,
                'Total': n_total_ds,
                'LSPC': n_lspc_ds,
                'HSPC': n_hspc_ds,
                'Uncertain': n_uncertain_ds,
                'Pct_LSPC': pct_lspc_ds,
                'Pct_HSPC': 100 - pct_lspc_ds
            })

    return pd.DataFrame(rows)

def plot_malignancy_by_dataset(stats_df, figsize=(20, 6)):
    """Create malignancy proportion by source dataset (wide figure)"""

    fig, ax = plt.subplots(figsize=figsize)

    dataset_stats = stats_df[stats_df['Dataset'] != 'Overall'].copy()

    if len(dataset_stats) > 0:
        # Get unique datasets and order them
        datasets = order_datasets(list(dataset_stats['Dataset'].unique()))
        n_datasets = len(datasets)

        # Position bars
        x = np.arange(n_datasets)
        width = 0.25
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

        for i, ct in enumerate(CELL_TYPES):
            ct_data = dataset_stats[dataset_stats['CellType'] == ct].set_index('Dataset')

            if len(ct_data) > 0:
                pct_values = [ct_data.loc[ds, 'Pct_LSPC'] if ds in ct_data.index else 0
                             for ds in datasets]

                bars = ax.bar(x + i * width, pct_values, width,
                             label=ct, alpha=0.85, color=colors[i])

                # Add percentage labels on bars
                for j, (bar, pct) in enumerate(zip(bars, pct_values)):
                    if pct > 0:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                               f'{pct:.1f}%', ha='center', va='bottom',
                               fontsize=8, fontweight='bold')

        ax.set_ylabel('Malignant Proportion (%)', fontweight='bold', fontsize=12)
        ax.set_xlabel('Source Dataset', fontweight='bold', fontsize=12)
        ax.set_title('Malignancy Rate by Source Dataset',
                    fontweight='bold', fontsize=13, pad=12)
        ax.set_xticks(x + width)
        ax.set_xticklabels(datasets, rotation=40, ha='right', fontsize=10)
        ax.legend(title='Cell Type', framealpha=0.5, fontsize=11, loc='upper left')
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No source dataset metadata available',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Malignancy by Source Dataset', fontweight='bold', fontsize=13)

    plt.tight_layout()
    return fig

def plot_celltype_distribution(stats_df, figsize=(20, 6)):
    """Create cell type distribution by source dataset (wide figure)"""

    fig, ax = plt.subplots(figsize=figsize)

    dataset_stats = stats_df[stats_df['Dataset'] != 'Overall'].copy()

    if len(dataset_stats) > 0:
        # Get unique datasets and order them
        datasets = order_datasets(list(dataset_stats['Dataset'].unique()))
        n_datasets = len(datasets)

        # Position bars
        x = np.arange(n_datasets)
        width = 0.25
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

        for i, ct in enumerate(CELL_TYPES):
            ct_data = dataset_stats[dataset_stats['CellType'] == ct].set_index('Dataset')

            if len(ct_data) > 0:
                count_values = [int(ct_data.loc[ds, 'Total']) if ds in ct_data.index else 0
                               for ds in datasets]

                bars = ax.bar(x + i * width, count_values, width,
                             label=ct, alpha=0.85, color=colors[i])

                # Add count labels on bars
                for j, (bar, count) in enumerate(zip(bars, count_values)):
                    if count > 0:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2, height + height*0.02,
                               f'{count:,}', ha='center', va='bottom',
                               fontsize=8, fontweight='bold', rotation=0)

        ax.set_ylabel('Cell Count', fontweight='bold', fontsize=12)
        ax.set_xlabel('Source Dataset', fontweight='bold', fontsize=12)
        ax.set_title('Cell Type Distribution Across Source Datasets',
                    fontweight='bold', fontsize=13, pad=12)
        ax.set_xticks(x + width)
        ax.set_xticklabels(datasets, rotation=40, ha='right', fontsize=10)
        ax.legend(title='Cell Type', framealpha=0.95, fontsize=11, loc='upper left')
        ax.grid(axis='y', alpha=0.3)

        # Format y-axis with thousands separator
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y):,}'))
    else:
        ax.text(0.5, 0.5, 'No source dataset metadata available',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Cell Type Distribution by Source Dataset',
                    fontweight='bold', fontsize=13)

    plt.tight_layout()
    return fig

def get_dataset_colors(datasets):
    """Assign colors to datasets by study group"""
    colors = {}

    # HC controls - Blue shades (light to dark)
    hc_colors = {
        'Setty': '#6baed6',   # light blue
        'Mende': '#31a354'    # darker blue
    }

    # Ennis - Red/Pink shades (light to dark)
    ennis_colors = {
        'Ennis_DG': '#fc9272',   # light red
        'Ennis_MRD': '#de2d26',  # medium red
        'Ennis_REL': '#a50f15'   # dark red
    }

    # Naldini - Orange/Yellow shades (light to dark)
    naldini_colors = {
        'Naldini_V03_DG': '#fec44f',   # light orange
        'Naldini_V03_MRD': '#fe9929',  # medium orange
        'Naldini_V03_REL': '#d95f0e',  # dark orange
        'Naldini_V02_DG': '#feb24c',   # light orange-yellow
        'Naldini_V02_MRD': '#fd8d3c'   # medium orange
    }

    # Henrik - Purple
    henrik_colors = {
        'Henrik_DG': '#756bb1'
    }

    # Petti - Green
    petti_colors = {
        'Petti_DG': '#6baed6'
    }

    # vanGalen - Blue
    vanGalen_colors = {
        'vanGalen': '#1E90FF'
    }

    # Combine all colors
    all_colors = {**hc_colors, **ennis_colors, **naldini_colors,
                  **henrik_colors, **petti_colors, **vanGalen_colors}

    # Assign colors to datasets
    for ds in datasets:
        if ds in all_colors:
            colors[ds] = all_colors[ds]
        else:
            colors[ds] = '#999999'  # default gray for unknown

    return colors

def plot_inter_dataset_comparison(stats_df, figsize=(20, 12)):
    """Create inter-dataset comparison with split count panels"""

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])   # Malignancy % - full width
    ax2 = fig.add_subplot(gs[1, 0])   # HSC_MPP counts
    ax3 = fig.add_subplot(gs[1, 1:])  # LMPP + Early_GMP counts

    fig.suptitle('Malignancy across cell type', fontsize=14, fontweight='bold', y=0.93)

    dataset_stats = stats_df[stats_df['Dataset'] != 'Overall'].copy()

    if len(dataset_stats) > 0:
        # Get unique datasets and order them
        datasets = order_datasets(list(dataset_stats['Dataset'].unique()))
        n_datasets = len(datasets)

        # Create color map for datasets
        dataset_colors = get_dataset_colors(datasets)

        # Subplot 1: Malignancy Rate (%)
        group_width = n_datasets * 0.8
        group_spacing = 0.5

        for i, ct in enumerate(CELL_TYPES):
            group_start = i * (group_width + group_spacing)
            ct_data = dataset_stats[dataset_stats['CellType'] == ct]

            for j, ds in enumerate(datasets):
                x_pos = group_start + j * 0.8

                if ds in ct_data['Dataset'].values:
                    row = ct_data[ct_data['Dataset'] == ds].iloc[0]
                    pct = row['Pct_LSPC']

                    ax1.bar(x_pos, pct, width=0.7, alpha=0.85,
                           color=dataset_colors[ds],
                           label=ds if i == 0 else "")

                    # Add percentage label
                    ax1.text(x_pos, pct + 1, f'{pct:.1f}%',
                           ha='center', va='bottom', fontsize=7, fontweight='bold')

        # Add group labels
        group_centers = []
        for i, ct in enumerate(CELL_TYPES):
            group_start = i * (group_width + group_spacing)
            group_center = group_start + group_width / 2
            group_centers.append(group_center)

        ax1.set_ylabel('Malignant Proportion (%)', fontweight='bold', fontsize=12)
        ax1.set_title('A. Malignancy Rate Across Datasets by Cell Type',
                     fontweight='bold', fontsize=12, pad=10)
        ax1.set_xticks(group_centers)
        ax1.set_xticklabels(CELL_TYPES, fontsize=11, fontweight='bold')
        ax1.set_ylim(0, 110)
        ax1.grid(axis='y', alpha=0.3)

        # Add vertical separators
        for i in range(1, len(CELL_TYPES)):
            sep_x = i * (group_width + group_spacing) - group_spacing / 2
            ax1.axvline(x=sep_x, color='gray', linestyle='--', alpha=0.3, linewidth=1)

        # Legend for datasets
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels, loc='upper left',
                  ncol=2, fontsize=9, framealpha=0.5)

        # Helper function to plot counts and calculate sum
        def plot_counts_for_celltypes(ax, cell_types_to_plot, title):
            # Recalculate group positions for this subplot
            local_group_width = n_datasets * 0.8
            local_group_spacing = 0.5
            local_group_centers = []
            max_heights = []

            for i, ct in enumerate(cell_types_to_plot):
                group_start = i * (local_group_width + local_group_spacing)
                group_center = group_start + local_group_width / 2
                local_group_centers.append(group_center)

                ct_data = dataset_stats[dataset_stats['CellType'] == ct]
                max_height = 0

                # Calculate sum excluding Setty and Mende
                sum_lspc = 0
                for _, row in ct_data.iterrows():
                    ds = row['Dataset']
                    if not (ds.startswith('Setty') or ds.startswith('Mende')):
                        sum_lspc += row['LSPC']

                for j, ds in enumerate(datasets):
                    x_pos = group_start + j * 0.8

                    if ds in ct_data['Dataset'].values:
                        row = ct_data[ct_data['Dataset'] == ds].iloc[0]
                        count = row['LSPC']
                        max_height = max(max_height, count)

                        ax.bar(x_pos, count, width=0.7, alpha=0.85,
                              color=dataset_colors[ds])

                        # Add count label
                        if count > 0:
                            ax.text(x_pos, count + count*0.02, f'{int(count):,}',
                                   ha='center', va='bottom', fontsize=7, fontweight='bold')

                max_heights.append(max_height)
                if ct=='HSC MPP':
                    # Add sum annotation above group
                    ax.text(group_center, 900, f'Total (excl. HC): {int(sum_lspc):,}',
                           ha='center', va='bottom', fontsize=10, fontweight='bold',
                           color='darkred')
                if ct=='LMPP' or ct =='Early GMP':
                    # Add sum annotation above group
                    ax.text(group_center, 12000, f'Total (excl. HC): {int(sum_lspc):,}',
                           ha='center', va='bottom', fontsize=10, fontweight='bold',
                           color='darkred')

            ax.set_ylabel('Malignant Cell Count', fontweight='bold', fontsize=12)
            ax.set_title(title, fontweight='bold', fontsize=12, pad=10)
            ax.set_xticks(local_group_centers)
            ax.set_xticklabels(cell_types_to_plot, fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y):,}'))

            # Add vertical separators
            for i in range(1, len(cell_types_to_plot)):
                sep_x = i * (local_group_width + local_group_spacing) - local_group_spacing / 2
                ax.axvline(x=sep_x, color='gray', linestyle='--', alpha=0.3, linewidth=1)

        # Subplot 2: HSC_MPP counts (left)
        plot_counts_for_celltypes(ax2, ['HSC MPP'], 'B. HSC/MPP Malignant Cell Count')

        # Subplot 3: LMPP + Early_GMP counts (right, 2x space)
        plot_counts_for_celltypes(ax3, ['LMPP', 'Early GMP'], 'C. LMPP & Early GMP Malignant Cell Counts')

    return fig

def generate_summary_table(stats_df):
    """Generate formatted summary table"""
    display_df = stats_df.copy()
    display_df['Malignant (%)'] = display_df['Pct_LSPC'].round(1)
    display_df['Normal (%)'] = display_df['Pct_HSPC'].round(1)

    summary = display_df[['CellType', 'Dataset', 'Total', 'LSPC', 'HSPC',
                          'Malignant (%)', 'Normal (%)']].copy()
    summary.columns = ['Cell Type', 'Source Dataset', 'Total Cells',
                       'Malignant', 'Normal', 'Malignant (%)', 'Normal (%)']

    return summary

def print_summary_report(stats_df):
    """Print summary to console"""
    print("\n" + "="*80)
    print("POOLED LSC MALIGNANCY DISTRIBUTION SUMMARY")
    print("="*80 + "\n")

    overall = stats_df[stats_df['Dataset'] == 'Overall']
    for _, row in overall.iterrows():
        print(f"{row['CellType']}:")
        print(f"  Total: {int(row['Total'])}, Malignant: {int(row['LSPC'])} ({row['Pct_LSPC']:.1f}%), "
              f"Normal: {int(row['HSPC'])} ({row['Pct_HSPC']:.1f}%)")

def main():
    combined_adata = load_classified_data()

    print(f"Loaded {combined_adata.n_obs} cells")

    stats_df = calculate_malignancy_stats(combined_adata, label_key=label_key)
    print_summary_report(stats_df)

    # Save summary table
    summary = generate_summary_table(stats_df)
    summary_path = OUTPUT_DIR / 'malignancy_summary.csv'
    summary.to_csv(summary_path, index=False)

    # Generate all figures
    print("\nGenerating figures...")

    fig1 = plot_malignancy_by_dataset(stats_df)
    fig1.savefig(OUTPUT_DIR / 'malignancy_by_dataset.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)

    fig2 = plot_celltype_distribution(stats_df)
    fig2.savefig(OUTPUT_DIR / 'celltype_distribution_by_dataset.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)

    fig3 = plot_inter_dataset_comparison(stats_df)
    fig3.savefig(OUTPUT_DIR / 'malignancy_by_celltype.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)

if __name__ == '__main__':
    main()
