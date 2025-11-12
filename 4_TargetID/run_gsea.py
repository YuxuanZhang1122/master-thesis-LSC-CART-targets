"""
GSEA Analysis for Malignant LSPC vs Healthy HSPC

This script performs Gene Set Enrichment Analysis (GSEA) on differential expression results
to identify enriched biological pathways in malignant vs healthy hematopoietic stem cells.

WORKFLOW:
1. Loads DEG results from linear mixed model (LMM) analysis
2. Ranks genes by t-statistic (coefficient/stderr from LMM)
3. Runs preranked GSEA with multiple gene set databases:
   - MSigDB Hallmark 2020
   - GO Biological Process 2023
   - KEGG 2021 Human
4. Generates summary visualizations and pathway tables

OUTPUT FILES (saved to OUTPUT_DIR):
Core Results:
- ranked_genes.txt: Gene list ranked by t-statistic (for GSEA input)
- deg_with_ranks.csv: DEG results with added t-statistic column
- gsea_hallmark_results.csv: GSEA results for Hallmark gene sets
- gsea_gobp_results.csv: GSEA results for GO Biological Process
- gsea_kegg_results.csv: GSEA results for KEGG pathways

Visualizations:
- gsea_hallmark_summary.png: Bar plot showing top enriched Hallmark pathways
- gsea_gobp_summary.png: Bar plot showing top enriched GO BP pathways
- gsea_kegg_summary.png: Bar plot showing top enriched KEGG pathways
- lsc_themed_lollipop.png: Themed lollipop plot for LSC-specific pathways (if CREATE_LOLLIPOP_PLOT=True)

Tables (if EXPORT_GENE_TABLES=True):
- pathway_gene_lists.xlsx: Excel file with leading edge genes per pathway (multiple sheets)

The script will automatically:
1. Load your DEG results and rank genes by t-statistic
2. Run GSEA on three databases (Hallmark, GO BP, KEGG)
3. Create summary bar plots for each database
4. Create themed LSC-specific lollipop plot (if enabled)
5. Export pathway gene tables to Excel/CSV (if enabled)
6. Print summary: "GSEA analysis complete: X significant pathways (FDR<0.25)"
"""

import pandas as pd
import numpy as np
import gseapy as gp
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuration
INPUT_FILE = 'DEG_results_singlecell_LMM/singlecell_deg_all_results.csv'
OUTPUT_DIR = 'gsea_results'
GENE_COL = 'gene'
COEF_COL = 'coefficient'
STDERR_COL = 'stderr'
LOG2FC_COL = 'log2FoldChange'
EXPORT_GENE_TABLES = True  # Set to True to export pathway gene lists to Excel/CSV
CREATE_LOLLIPOP_PLOT = True  # Set to True to create lollipop plot with refined LSC-specific themes
LSC_PLOT_FILE = 'lsc_themed_lollipop.png'
LSC_FDR_THRESHOLD = 0.25
LSC_TOP_N_PER_THEME = 5


def load_and_rank_genes(input_file, gene_col='gene', coef_col='coefficient',
                         stderr_col='stderr', log2fc_col='log2FC'):
    """
    Load DEG results and create ranked gene list.

    Returns:
        pandas.Series: Ranked genes (index=gene, value=rank)
        pandas.DataFrame: Full DEG table with ranks
    """
    deg_df = pd.read_csv(input_file)

    # Check if coefficient column exists, otherwise use log2FC
    effect_col = coef_col if coef_col in deg_df.columns else log2fc_col

    # Calculate t-statistic: coefficient / stderr
    deg_df['t_statistic'] = deg_df[effect_col] / deg_df[stderr_col]

    # Remove NaN/Inf values
    deg_df = deg_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['t_statistic'])

    # Sort by t-statistic
    deg_df_sorted = deg_df.sort_values('t_statistic', ascending=False)

    # Create ranked series
    rnk = deg_df_sorted.set_index(gene_col)['t_statistic']

    return rnk, deg_df_sorted


def run_gsea_analysis(rnk, output_dir, gene_sets='auto'):
    """
    Run preranked GSEA analysis on multiple gene set databases.

    Performs GSEA using a preranked gene list (ranked by t-statistic from DEG analysis)
    against standard gene set databases. Results are saved as CSV files in the output directory.

    Parameters
    ----------
    rnk : pd.Series
        Ranked gene list (index=gene names, values=ranking metric/t-statistic)
    output_dir : str or Path
        Directory where GSEA results will be saved
    gene_sets : str or list, default='auto'
        Gene set databases to use. If 'auto', uses:
        - MSigDB Hallmark 2020
        - GO Biological Process 2023
        - KEGG 2021 Human

    Returns
    -------
    dict
        Dictionary mapping database short names to results DataFrames
        Keys: 'hallmark', 'gobp', 'kegg'
        Values: DataFrames with columns Term, NES, NOM p-val, FDR q-val, etc.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    if gene_sets == 'auto':
        # Default gene sets for AML/stem cell analysis
        gene_sets_list = [
            ('MSigDB_Hallmark_2020', 'hallmark'),
            ('GO_Biological_Process_2023', 'gobp'),
            ('KEGG_2021_Human', 'kegg'),
        ]
    else:
        gene_sets_list = [(gs, gs.lower().replace('_', '')) for gs in gene_sets]
    
    results_dict = {}

    for gene_set_name, short_name in gene_sets_list:

        gsea_result = gp.prerank(
            rnk=rnk,
            gene_sets=gene_set_name,
            processes=4,
            permutation_num=1000,
            outdir=str(output_path / short_name),
            seed=42,
            verbose=True,
            min_size=15,  # Minimum genes per pathway
            max_size=500  # Maximum genes per pathway
        )

        # Save results
        results_df = gsea_result.res2d
        results_df.to_csv(output_path / f'gsea_{short_name}_results.csv', index=False)
        results_dict[short_name] = results_df

        # Convert FDR q-val to numeric
        results_df['FDR q-val'] = pd.to_numeric(results_df['FDR q-val'], errors='coerce')
    
    return results_dict


def create_summary_plot(results_dict, output_dir):
    """Create summary visualization of top pathways."""
    output_path = Path(output_dir)

    for db_name, results_df in results_dict.items():
        # Convert FDR q-val to numeric
        results_df['FDR q-val'] = pd.to_numeric(results_df['FDR q-val'], errors='coerce')

        # Filter significant results
        sig_results = results_df[results_df['FDR q-val'] < 0.25].copy()

        if len(sig_results) == 0:
            continue
        
        # Get top 10 from each direction
        top_pos = sig_results[sig_results['NES'] > 0].nsmallest(10, 'FDR q-val')
        top_neg = sig_results[sig_results['NES'] < 0].nsmallest(10, 'FDR q-val')
        
        plot_df = pd.concat([top_pos, top_neg]).sort_values('NES')
        
        if len(plot_df) == 0:
            continue
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.3)))
        
        colors = ['#d62728' if x > 0 else '#1f77b4' for x in plot_df['NES']]
        y_pos = range(len(plot_df))
        
        ax.barh(y_pos, plot_df['NES'], color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([term[:50] for term in plot_df['Term']], fontsize=8)
        ax.set_xlabel('Normalized Enrichment Score (NES)', fontsize=10)
        ax.set_title(f'Top GSEA Pathways - {db_name.upper()}', fontsize=12, fontweight='bold')
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.grid(axis='x', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#d62728', alpha=0.7, label='Enriched in Malignant'),
            Patch(facecolor='#1f77b4', alpha=0.7, label='Enriched in Healthy')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(output_path / f'gsea_{db_name}_summary.png', dpi=300, bbox_inches='tight')
        plt.close()


def categorize_pathway(pathway_name):
    """Auto-categorize pathway based on keywords."""
    pathway_lower = pathway_name.lower()

    # Define category keywords
    categories = {
        'Immune': ['immune', 'inflammatory', 'interferon', 'cytokine', 'antigen',
                   't cell', 'b cell', 'leukocyte', 'lymphocyte', 'innate', 'adaptive',
                   'il1', 'il6', 'tnf', 'nfkb', 'complement', 'chemokine'],
        'Cell Cycle': ['g1', 'g2m', 'g2 m', 'mitotic', 'cell cycle', 'dna replication',
                       'checkpoint', 'spindle', 'centrosome', 'chromosome', 'mitosis',
                       'e2f', 'cyclin', 'cdk'],
        'Metabolism': ['metabolism', 'metabolic', 'glycolysis', 'glycolytic', 'oxidative',
                       'tca', 'fatty acid', 'respiration', 'atp', 'nadh', 'krebs',
                       'carbon', 'amino acid', 'lipid', 'glucose'],
        'DNA Damage': ['dna damage', 'dna repair', 'apoptosis', 'p53', 'atm', 'atr',
                       'telomere', 'reactive oxygen', 'ros', 'stress', 'unfolded protein',
                       'er stress', 'upr'],
        'Signaling': ['signaling', 'signal', 'mapk', 'pi3k', 'akt', 'mtor', 'jak', 'stat',
                      'tgf', 'wnt', 'notch', 'hedgehog', 'ras', 'erk', 'jnk'],
        'Development': ['differentiation', 'development', 'developmental', 'stem cell',
                        'hematopoietic', 'embryonic', 'morphogenesis', 'organogenesis',
                        'lineage', 'progenitor'],
    }

    for category, keywords in categories.items():
        if any(keyword in pathway_lower for keyword in keywords):
            return category

    return 'Other'


def create_pathway_gene_tables(results_dir, output_file='pathway_gene_lists'):
    """
    Extract gene lists for each enriched pathway from GSEA results.

    Creates comprehensive tables with pathway name, NES, FDR, p-value,
    number of genes, and leading edge genes.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing GSEA results
    output_file : str
        Base name for output files (without extension)
    """
    results_path = Path(results_dir)

    # Read all result files
    all_results = []
    for db_file, db_name in [('gsea_hallmark_results.csv', 'HALLMARK'),
                              ('gsea_gobp_results.csv', 'GO_BP'),
                              ('gsea_kegg_results.csv', 'KEGG')]:
        file_path = results_path / db_file
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['FDR q-val'] = pd.to_numeric(df['FDR q-val'], errors='coerce')
            df['Database'] = db_name
            all_results.append(df)

    if not all_results:
        return

    all_results = pd.concat(all_results, ignore_index=True)

    # Filter significant (FDR < 0.25)
    sig_results = all_results[all_results['FDR q-val'] < 0.25].copy()

    if len(sig_results) == 0:
        return

    # Sort by abs(NES) for importance
    sig_results['abs_NES'] = sig_results['NES'].abs()
    sig_results = sig_results.sort_values('abs_NES', ascending=False)

    # Create comprehensive table
    output_table = []

    for idx, row in sig_results.iterrows():
        pathway = row['Term']
        database = row['Database']
        nes = row['NES']
        fdr = row['FDR q-val']
        pval = row['NOM p-val']
        gene_percent = row['Tag %']
        genes_str = row['Lead_genes'] if 'Lead_genes' in row else ''

        # Parse gene list
        if pd.notna(genes_str) and genes_str:
            genes = genes_str.split(';')
            n_genes = len(genes)
            gene_list = ', '.join(genes)
        else:
            n_genes = 0
            gene_list = ''

        output_table.append({
            'Pathway': pathway,
            'Database': database,
            'NES': nes,
            'FDR_qval': fdr,
            'Nominal_pval': pval,
            'Direction': 'Activated' if nes > 0 else 'Inhibited',
            'N_Genes': n_genes,
            'Gene_Percent': gene_percent,
            'Leading_Edge_Genes': gene_list
        })

    output_df = pd.DataFrame(output_table)

    # Save to Excel with multiple sheets
    excel_path = results_path / f'{output_file}.xlsx'

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        output_df.to_excel(writer, sheet_name='All_Pathways', index=False)

        activated = output_df[output_df['Direction'] == 'Activated'].copy()
        activated.to_excel(writer, sheet_name='Activated', index=False)

        inhibited = output_df[output_df['Direction'] == 'Inhibited'].copy()
        inhibited.to_excel(writer, sheet_name='Inhibited', index=False)

        for db in ['HALLMARK', 'GO_BP', 'KEGG']:
            db_df = output_df[output_df['Database'] == db].copy()
            if len(db_df) > 0:
                db_df.to_excel(writer, sheet_name=db, index=False)

if __name__ == '__main__':
    # ========================================================================
    # STEP 1: Load DEG results and rank genes by t-statistic
    # ========================================================================
    rnk, deg_df = load_and_rank_genes(
        INPUT_FILE,
        gene_col=GENE_COL,
        coef_col=COEF_COL,
        stderr_col=STDERR_COL,
        log2fc_col=LOG2FC_COL
    )

    # ========================================================================
    # STEP 2: Save ranked gene list
    # ========================================================================
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True, parents=True)
    rnk.to_csv(output_path / 'ranked_genes.txt', sep='\t', header=False)
    deg_df.to_csv(output_path / 'deg_with_ranks.csv', index=False)

    # ========================================================================
    # STEP 3: Run GSEA on Hallmark, GO BP, and KEGG databases
    # ========================================================================
    results_dict = run_gsea_analysis(rnk, OUTPUT_DIR, gene_sets='auto')

    # ========================================================================
    # STEP 4: Create summary visualizations
    # ========================================================================
    # Bar plots showing top pathways per database
    create_summary_plot(results_dict, OUTPUT_DIR)

    # ========================================================================
    # STEP 5: Export pathway gene tables (optional)
    # ========================================================================
    if EXPORT_GENE_TABLES:
        create_pathway_gene_tables(OUTPUT_DIR)

    # ========================================================================
    # STEP 6: Create refined LSC-specific themed visualization (optional)
    # ========================================================================
    if CREATE_LOLLIPOP_PLOT:
        from plot_gsea import (
            load_gsea_results as load_gsea_for_lsc,
            map_pathways_to_themes,
            select_top_pathways_per_theme,
            create_lollipop_plot as create_lsc_lollipop,
            print_summary as print_lsc_summary
        )

        lsc_results = load_gsea_for_lsc(OUTPUT_DIR)
        lsc_themes = map_pathways_to_themes(lsc_results, LSC_FDR_THRESHOLD)
        lsc_selected = select_top_pathways_per_theme(lsc_themes, LSC_TOP_N_PER_THEME)
        create_lsc_lollipop(lsc_selected, str(output_path / LSC_PLOT_FILE))
        print_lsc_summary(lsc_selected)

    # ========================================================================
    # SUMMARY
    # ========================================================================
    n_sig = sum((df['FDR q-val'] < 0.25).sum() for df in results_dict.values())
    print(f'\nGSEA analysis complete: {n_sig} significant pathways (FDR<0.25)')
    print(f'Results saved to: {OUTPUT_DIR}')
