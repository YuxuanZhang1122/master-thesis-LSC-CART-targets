import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path("../dataset/DotPlot")
OUTPUT_DIR = Path("../outputs/DotPlot/two_panel/functional_panel")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

GENES_HSCMPP_Marker_Zeng = [
    'CRHBP', 'AVP', 'IFITM3', 'MLLT3', 'THY1', 'CD164', 'TCEAL2',
    'CLIC2A', 'BST2', 'HMCN1', 'PCDH9', 'MDK', 'BEX1', 'CTCFL', 'UMCH1',
    'SYNE1', 'RBPMS', 'HOPX', 'NPR3', 'CCDC42', 'MMRN1', 'MECOM',
    'HIST1H2BE', 'PRKG2', 'TSC22D1', 'CCND2', 'CYL1', 'PFAS1', 'HSD17B11',
    'IDS', 'FGFR3', 'ELMO1', 'FHL3', 'ANKRD28', 'PTRF', 'TREM1', 'KRT8',
    'NFIB', 'C6orf48', 'AADL', 'PRDX1', 'ARHGAP4', 'CPVL4', 'KRT18',
    'PREX1', 'SOCS2', 'CMPK1', 'SNHG12'
]
GENES_earlyGMP_Marker_Zeng = [
    'PRSS57', 'CLQTNF4', 'MPO', 'NPW', 'CD38', 'MGST1', 'SPINK2', 'CLDN10',
    'NAALAD1', 'CSF3R', 'SERPINB1', 'CD34', 'CLEC11A', 'KIT', 'NUCB2',
    'RAB32', 'CDK6', 'FAM216A', 'TRH', 'CRYGD', 'NPM3', 'ENO1', 'CDCA7',
    'SPARC', 'IGFL1', 'TNFSF13B', 'IGLL1', 'IMPDH2', 'APEX1', 'SDCBP',
    'NIT2', 'PRAM1', 'PROM1', 'C9orf43', 'COMT', 'PPA1', 'NME1', 'TNFSF13',
    'CDK4', 'CEBPA', 'ATP8B4', 'HSPB1', 'ARC', 'VAMP5', 'PROM2', 'FAH'
]
GENES_HSPC_GMP_Marker_vanGalen = [
    'MEIS1', 'EGR1', 'MSI2', 'CD38', 'CD34', 'PROM1', 'EGFL7',
    'MPO', 'ELANE', 'CTSG', 'AZU1', 'LYST', 'LYZ', 'CEBPD', 'MNDA'
]

GENES_GMP_Marker_vanGalen = [
    'MPO', 'ELANE', 'CTSG', 'AZU1', 'LYST', 'LYZ', 'CEBPD', 'MNDA'
]

GENES_LSPC_ZENG = [
    'BCL11A', 'XBP1', 'NFE2', 'MAX', 'GLI3',
    'GATA2', 'ELK4', 'REST', 'CREB1', 'MYCN',
    'E2F1', 'MYBL2', 'E2F7', 'E2F8', 'CTCF'
]

ZENG_Markers = {
    'Quiscent': ['BCL11A', 'XBP1', 'NFE2', 'MAX', 'GLI3'],
    'Primed': ['GATA2', 'ELK4', 'REST', 'CREB1', 'MYCN'],
    'Cycle': ['E2F1', 'MYBL2', 'E2F7', 'E2F8', 'CTCF']
}

# Core stemness markers (surface antigens + key TFs)
stemness_genes = [
    'CD34', 'CD38', 'THY1', 'PROM1', 'KIT',  # Surface
    'HOPX', 'MECOM', 'GATA2', 'RUNX1', 'TAL1', 'ERG', 'HLF',  # TFs
    'BMI1', 'SOX4', 'MEIS1',  # Self-renewal
    'LEF1', 'TCF7', 'HES1', 'NOTCH1', 'PTEN'  # Signaling
]

# Proliferation genes
proliferation_genes = [
    'MKI67', 'TOP2A', 'PCNA', 'CDK1', 'CDK2', 'CDK4', 'CDK6',
    'CCNA2', 'CCNB1', 'CCND1', 'CCNE1',  # Cyclins
    'MCM2', 'MCM3', 'MCM4', 'MCM5', 'MCM6', 'MCM7',  # Replication licensing
    'AURKA', 'AURKB',  # Aurora kinases
    'MYC', 'MYCN', 'E2F1', 'E2F2'  # Proliferation TFs
]

# Housekeeping genes (commonly used references)
housekeeping_genes = [
    'ACTB', 'GAPDH', 'B2M', 'PPIA', 'RPL13A', 'RPLP0',
    'TBP', 'HPRT1', 'PGK1', 'GUSB', 'TFRC', 'UBC',
    'YWHAZ', 'HSP90AB1', 'HMBS', 'SDHA'
]

metabolic_stress = [
    'LDHA', 'PKM', 'G6PD', 'HSPA1A', 'HSPA1B', 'FOS', 'JUN'
]

FUNCTIONAL_GENES = [
    # Stemness (6 genes)
    'CD34', 'PROM1', 'KIT', 'HOPX', 'MECOM', 'MEIS1',
    # Proliferation (6 genes)
    'MKI67', 'TOP2A', 'PCNA', 'CDK4', 'CDK6', 'MYC',
    # Housekeeping (4 genes)
    'ACTB', 'GAPDH', 'B2M', 'HPRT1',
    # Metabolic stress (4 genes)
    'LDHA', 'PKM', 'HSPA1A', 'FOS'
]

FUNCTIONAL_MARKERS = {
    'Stemness': ['CD34', 'PROM1', 'KIT', 'HOPX', 'MECOM', 'MEIS1'],
    'Proliferation': ['MKI67', 'TOP2A', 'PCNA', 'CDK4', 'CDK6', 'MYC'],
    'Housekeeping': ['ACTB', 'GAPDH', 'B2M', 'HPRT1'],
    'Metabolic_Stress': ['LDHA', 'PKM', 'HSPA1A', 'FOS']
}

HSC_LMPP_earlyGMP = [
    'CD34', 'AVP', 'PROM1', 'HOPX', 'SPINK2', 'MLLT3',  # Stemness/HSC markers
    'FLT3', 'SATB1', 'DNTT', 'IL7R', 'RAG1',  # lymphoid-primed
    'MPO', 'ELANE', 'AZU1', 'CTSG', 'PRTN3', 'CSF3R',  # meyloid commitment
]

CELL_STATE_MARKERS = {
    'HSC/MPP': ['CD34', 'AVP', 'PROM1', 'HOPX', 'SPINK2', 'MLLT3'],
    'LMPP': ['FLT3', 'SATB1', 'DNTT', 'IL7R', 'RAG1'],
    'earlyGMP': ['MPO', 'ELANE', 'AZU1', 'CTSG', 'PRTN3', 'CSF3R']
}

DATASETS = [
    'vG_HSPC_Zeng_HSCMPP.h5ad',
    'vG_HSPC_Zeng_LMPP.h5ad',
    'vG_HSPC_Zeng_earlyGMP.h5ad',
    'vG_GMP-like_Zeng_earlyGMP.h5ad',
]

def calc_gene_stats(adata, genes):
    """Calculate mean expression and expression percentage for genes"""
    stats = []

    for gene in genes:
        if gene not in adata.var_names:
            stats.append({'mean_expr': 0, 'pct_expr': 0})
            continue

        expr = adata[:, gene].X
        if hasattr(expr, 'toarray'):
            expr = expr.toarray().flatten()
        else:
            expr = np.array(expr).flatten()

        mean_expr = np.mean(expr)
        pct_expr = 100 * np.sum(expr > 0) / len(expr)
        stats.append({'mean_expr': mean_expr, 'pct_expr': pct_expr})

    return stats

def calc_cell_state_scores(markers=None):
    """Calculate cell state scores for all datasets using sc.tl.score_genes"""
    if markers is None:
        markers = CELL_STATE_MARKERS

    all_scores = []
    aggregated_scores = []

    for ds_name in DATASETS:
        logger.info(f"Calculating scores for {ds_name}")
        adata = sc.read_h5ad(DATA_DIR / ds_name)

        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        cell_scores = {'dataset': ds_name.replace('.h5ad', '')}
        agg_scores = {'dataset': ds_name.replace('.h5ad', '')}

        for state_name, marker_genes in markers.items():
            available_markers = [g for g in marker_genes if g in adata.var_names]

            if len(available_markers) == 0:
                logger.warning(f"No markers found for {state_name} in {ds_name}")
                cell_scores[state_name] = np.zeros(adata.n_obs)
                agg_scores[state_name] = 0
                continue

            marker_expr = adata[:, available_markers].X
            if hasattr(marker_expr, 'toarray'):
                marker_expr = marker_expr.toarray()
            scores = marker_expr.mean(axis=1).flatten()

            cell_scores[state_name] = scores
            agg_scores[state_name] = np.mean(scores)

        all_scores.append(cell_scores)
        aggregated_scores.append(agg_scores)

    return all_scores, aggregated_scores

def load_and_process_data(genes):
    """Load all datasets and calculate gene expression statistics"""
    results = []

    for ds_name in DATASETS:
        logger.info(f"Processing {ds_name}")
        adata = sc.read_h5ad(DATA_DIR / ds_name)
        stats = calc_gene_stats(adata, genes)

        for gene, stat in zip(genes, stats):
            results.append({
                'dataset': ds_name.replace('.h5ad', ''),
                'gene': gene,
                'mean_expr': stat['mean_expr'],
                'pct_expr': stat['pct_expr']
            })

    return pd.DataFrame(results)

def create_wide_dotplot(df, output_path, genes, title):
    """Create wide dot plot with 90-degree rotated gene names"""
    pivot_mean = df.pivot(index='dataset', columns='gene', values='mean_expr')
    pivot_pct = df.pivot(index='dataset', columns='gene', values='pct_expr')

    pivot_mean = pivot_mean[[g for g in genes if g in pivot_mean.columns]]
    pivot_pct = pivot_pct[[g for g in genes if g in pivot_pct.columns]]

    dataset_order = [ds.replace('.h5ad', '') for ds in DATASETS]
    available_datasets = [ds for ds in dataset_order if ds in pivot_mean.index]
    pivot_mean = pivot_mean.reindex(available_datasets)
    pivot_pct = pivot_pct.reindex(available_datasets)

    n_genes = len(pivot_mean.columns)
    fig_width = max(14, n_genes * 0.5 + 4)
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    norm_mean = plt.Normalize(
        vmin=pivot_mean.values.min(),
        vmax=pivot_mean.values.max()
    )
    cmap = plt.cm.Reds

    for i, dataset in enumerate(pivot_mean.index):
        for j, gene in enumerate(pivot_mean.columns):
            mean_val = pivot_mean.loc[dataset, gene]
            pct_val = pivot_pct.loc[dataset, gene]

            color = cmap(norm_mean(mean_val))
            size = (pct_val / 100) * 400

            ax.scatter(j, i, s=size, c=[color], edgecolors='black',
                      linewidths=0.5, alpha=0.9)

    ax.set_xticks(range(len(pivot_mean.columns)))
    ax.set_xticklabels(pivot_mean.columns, rotation=90, fontsize=9)
    ax.set_yticks(range(len(pivot_mean.index)))
    ax.set_yticklabels(pivot_mean.index, fontsize=10)

    ax.set_xlabel('Genes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Datasets', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, linestyle='--')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_mean)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Mean Expression', shrink=0.6)

    size_legend = [20, 50, 80]
    legend_elements = [
        plt.scatter([], [], s=(s/100)*400, c='gray', alpha=0.7,
                   edgecolors='black', linewidths=0.5)
        for s in size_legend
    ]
    legend_labels = [f'{s}%' for s in size_legend]

    legend = ax.legend(legend_elements, legend_labels,
                      title='% Expressing',
                      loc='upper left', bbox_to_anchor=(1.12, 1),
                      frameon=True, fontsize=9)
    legend.get_title().set_fontsize(10)
    legend.get_title().set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved figure to {output_path}")
    plt.close()

    return fig

def create_aggregated_scores_heatmap(aggregated_scores, ax=None, markers=None):
    """Create heatmap showing mean scores per dataset and cell state"""
    df = pd.DataFrame(aggregated_scores)
    df = df.set_index('dataset')

    if markers is not None:
        state_order = list(markers.keys())
    else:
        state_order = ['HSC/MPP', 'LMPP', 'earlyGMP']
    df = df[state_order]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    sns.heatmap(df, cmap='Greys', annot=True, fmt='.3f',
                linewidths=0.2, linecolor='black', ax=ax,
                cbar_kws={'label': 'Mean Score', 'shrink': 0.8})

    ax.invert_yaxis()

    ax.set_xlabel('Cell State', fontsize=11, fontweight='bold')
    ax.set_ylabel('Dataset', fontsize=11, fontweight='bold')
    ax.set_title('Aggregated Cell State Scores', fontsize=12, fontweight='bold', pad=10)

    return fig

def create_two_panel_figure(output_path, genes=None, markers=None):
    """Create two-panel figure with dotplot and aggregated scores"""
    from matplotlib.gridspec import GridSpec

    if genes is None:
        genes = HSC_LMPP_earlyGMP
    if markers is None:
        markers = CELL_STATE_MARKERS

    logger.info("Calculating gene expression statistics...")
    df = load_and_process_data(genes)

    logger.info("Calculating cell state scores...")
    all_scores, aggregated_scores = calc_cell_state_scores(markers)

    n_genes = len(genes)
    dotplot_width = max(14, n_genes * 0.5 + 4)

    fig = plt.figure(figsize=(dotplot_width, 9))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[5, 4],
                  hspace=0.4)

    ax_dot = fig.add_subplot(gs[0])
    ax_agg = fig.add_subplot(gs[1])

    logger.info("Creating dot plot...")
    pivot_mean = df.pivot(index='dataset', columns='gene', values='mean_expr')
    pivot_pct = df.pivot(index='dataset', columns='gene', values='pct_expr')

    pivot_mean = pivot_mean[[g for g in genes if g in pivot_mean.columns]]
    pivot_pct = pivot_pct[[g for g in genes if g in pivot_pct.columns]]

    dataset_order = [ds.replace('.h5ad', '') for ds in DATASETS]
    available_datasets = [ds for ds in dataset_order if ds in pivot_mean.index]
    pivot_mean = pivot_mean.reindex(available_datasets)
    pivot_pct = pivot_pct.reindex(available_datasets)

    norm_mean = plt.Normalize(vmin=pivot_mean.values.min(), vmax=pivot_mean.values.max())
    cmap = plt.cm.Reds

    for i, dataset in enumerate(pivot_mean.index):
        for j, gene in enumerate(pivot_mean.columns):
            mean_val = pivot_mean.loc[dataset, gene]
            pct_val = pivot_pct.loc[dataset, gene]
            color = cmap(norm_mean(mean_val))
            size = (pct_val / 100) * 400
            ax_dot.scatter(j, i, s=size, c=[color], edgecolors='black',
                          linewidths=0.5, alpha=0.9)

    cumulative_pos = 0
    for i, (state_name, marker_list) in enumerate(markers.items()):
        if i > 0:
            ax_dot.axvline(x=cumulative_pos - 0.5, color='black',
                          linestyle='--', linewidth=0.5, alpha=0.7)
        cumulative_pos += len(marker_list)

    ax_dot.set_xticks(range(len(pivot_mean.columns)))
    ax_dot.set_xticklabels(pivot_mean.columns, rotation=90, fontsize=9)
    ax_dot.set_yticks(range(len(pivot_mean.index)))
    ax_dot.set_yticklabels(pivot_mean.index, fontsize=10)
    ax_dot.set_xlabel('Genes', fontsize=12, fontweight='bold')
    ax_dot.set_ylabel('Datasets', fontsize=12, fontweight='bold')
    ax_dot.set_title('HSC/LMPP/earlyGMP Gene Expression', fontsize=14, fontweight='bold', pad=20)
    ax_dot.spines['top'].set_visible(False)
    ax_dot.spines['right'].set_visible(False)
    ax_dot.grid(True, alpha=0.2, linestyle='--')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_mean)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_dot, label='Mean Expression', shrink=0.6)

    size_legend = [20, 50, 80]
    legend_elements = [
        plt.scatter([], [], s=(s/100)*400, c='gray', alpha=0.7,
                   edgecolors='black', linewidths=0.5)
        for s in size_legend
    ]
    legend_labels = [f'{s}%' for s in size_legend]
    legend = ax_dot.legend(legend_elements, legend_labels,
                          title='% Expressing',
                          loc='upper left', bbox_to_anchor=(1.12, 1),
                          frameon=True, fontsize=9)
    legend.get_title().set_fontsize(10)
    legend.get_title().set_fontweight('bold')

    logger.info("Creating aggregated scores heatmap...")
    create_aggregated_scores_heatmap(aggregated_scores, ax=ax_agg, markers=markers)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved two-panel figure to {output_path}")
    plt.close()

def main():
    logger.info(f"\n{'='*60}")
    logger.info("Creating two-panel HSC/LMPP/earlyGMP figure...")
    logger.info(f"{'='*60}")

    two_panel_path = OUTPUT_DIR / "two_panel_HSC_LMPP_earlyGMP.png"
    create_two_panel_figure(two_panel_path, genes=FUNCTIONAL_GENES, markers=FUNCTIONAL_MARKERS)


if __name__ == "__main__":
    main()
