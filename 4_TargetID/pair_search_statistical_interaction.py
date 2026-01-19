#!/usr/bin/env python3
"""
CAR-T Gene Pair Analysis - Statistical Interaction Model
Identifies optimal dual surface antigen targets using binomial mixed-effects models.
Uses continuous log-normalized expression for all terms: main effects (gene1, gene2)
and interaction term (gene1 * gene2).
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
import multiprocessing as mp

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from tqdm import tqdm

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
BASE_DIR = Path(__file__).parent
ADATA_PATH = BASE_DIR / "HSC_MPP_full_surface_filtered.h5ad"
DESEQ_PATH = BASE_DIR / "Outputs" / "DEG_results_pseudobulk_DESeq2_surface" / "deseq2_results_druggable.csv"
OUTPUT_DIR = BASE_DIR / "Pair_search/statistical_interaction/results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR = OUTPUT_DIR.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_prepare_data(
    adata_path: str,
    label_col: str = 'consensus_label_6votes'
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Load h5ad file and prepare expression matrix with metadata.

    Parameters
    ----------
    adata_path : str
        Path to h5ad file
    label_col : str
        Column name for cell type labels (LSPC/HSPC)

    Returns
    -------
    expr_df : pd.DataFrame
        Log-normalized expression matrix (cells × genes)
    metadata : pd.DataFrame
        Cell metadata with Study, Donor, and status columns
    binary_expr : np.ndarray
        Binary expression matrix (cells × genes), 1 if raw count > 0
    """
    logger.info(f"Loading data from {adata_path}")
    adata = sc.read_h5ad(adata_path)

    # Filter to LSPC and HSPC only
    valid_cells = adata.obs[label_col].isin(['LSPC', 'HSPC'])
    adata = adata[valid_cells, :].copy()

    logger.info(f"Cells after filtering: {adata.n_obs}")
    logger.info(f"LSPC: {(adata.obs[label_col] == 'LSPC').sum()}")
    logger.info(f"HSPC: {(adata.obs[label_col] == 'HSPC').sum()}")

    # Get raw counts for binary expression
    logger.info("Creating binary expression matrix (raw count > 0)")
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    binary_expr = (X > 0).astype(int)

    # Normalize: CPM + log1p (for continuous model)
    logger.info("Normalizing expression data (CPM + log1p)")
    total_counts = X.sum(axis=1, keepdims=True)
    cpm = (X / total_counts) * 1e6
    log_cpm = np.log1p(cpm)

    # Create expression dataframe
    expr_df = pd.DataFrame(
        log_cpm,
        index=adata.obs_names,
        columns=adata.var_names
    )

    # Create metadata
    metadata = pd.DataFrame({
        'cell_id': adata.obs_names,
        'status': (adata.obs[label_col] == 'LSPC').astype(int),  # 1=LSPC, 0=HSPC
        'status_label': adata.obs[label_col],
        'Study': adata.obs['Study'],
        'Donor': adata.obs['Donor']
    })

    # Create Study-Donor combined variable for nested random effects
    metadata['Study_Donor'] = metadata['Study'].astype(str) + '_' + metadata['Donor'].astype(str)

    logger.info(f"Expression matrix shape: {expr_df.shape}")
    logger.info(f"Binary matrix shape: {binary_expr.shape}")

    return expr_df, metadata, binary_expr


def select_gene_pairs(
    deseq_results_path: str,
    expr_df: pd.DataFrame,
    metadata: pd.DataFrame,
    anchor_fc: float = 0.6,
    anchor_padj: float = 0.05,
    partner_fc: float = 0,
    min_coverage_pct: float = 30.0
) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """
    Select anchor and partner genes, generate all pairs.

    Parameters
    ----------
    deseq_results_path : str
        Path to DESeq2 results CSV
    expr_df : pd.DataFrame
        Expression matrix (log1p CPM)
    metadata : pd.DataFrame
        Cell metadata
    anchor_fc : float
        log2FC threshold for anchor genes (default: 0.5)
    anchor_padj : float
        Adjusted p-value threshold for anchor genes (default: 0.05)
    partner_fc : float
        log2FC threshold for partner genes (default: -0.5)
    min_coverage_pct : float
        Minimum % of LSPC cells expressing gene (default: 30%)

    Returns
    -------
    anchor_genes : list
        Anchor gene candidates (upregulated in LSPC)
    partner_genes : list
        Partner gene candidates
    gene_pairs : list of tuples
        All unique gene pairs to test
    """
    logger.info(f"Loading DESeq2 results from {deseq_results_path}")
    deseq_df = pd.read_csv(deseq_results_path)

    # Calculate % coverage using binary expression (>0 = detected)
    lspc_mask = metadata['status'] == 1
    n_lspc = lspc_mask.sum()

    logger.info(f"Calculating coverage for {len(expr_df.columns)} genes...")

    gene_coverage = {}
    for gene in expr_df.columns:
        # Binary detection: expressed if > 0 (same as raw count > 0)
        lspc_expr = expr_df.loc[lspc_mask, gene]
        pct_expressed = 100 * (lspc_expr > 0).sum() / n_lspc
        gene_coverage[gene] = pct_expressed

    # Select ANCHOR genes: log2FC > 0.5, padj < 0.05, coverage >= 30%
    anchor_candidates = deseq_df[
        (deseq_df['log2FoldChange'] > anchor_fc) &
        (deseq_df['padj'] < anchor_padj)
    ]['gene'].tolist()

    anchors = [
        g for g in anchor_candidates
        if g in expr_df.columns and gene_coverage[g] >= min_coverage_pct
    ]

    logger.info(f"Anchor genes (log2FC>{anchor_fc}, padj<{anchor_padj}, LSPC coverage>={min_coverage_pct}%): {len(anchors)}")

    # Select PARTNER genes: log2FC > -0.5, coverage >= 30%
    partner_candidates = deseq_df[
        deseq_df['log2FoldChange'] > partner_fc
    ]['gene'].tolist()

    partners = [
        g for g in partner_candidates
        if g in expr_df.columns and gene_coverage[g] >= min_coverage_pct
    ]

    logger.info(f"Partner genes (log2FC>{partner_fc}, LSPC coverage>={min_coverage_pct}%): {len(partners)}")

    # Generate all pairs: anchors × partners (excluding self-pairs)
    gene_pairs = [(g1, g2) for g1 in anchors for g2 in partners if g1 != g2]
    logger.info(f"Total gene pairs to test: {len(gene_pairs)}")

    return anchors, partners, gene_pairs


def filter_by_dual_coverage(
    gene_pairs: List[Tuple[str, str]],
    binary_expr: np.ndarray,
    expr_df: pd.DataFrame,
    metadata: pd.DataFrame,
    min_dual_lspc_pct: float = 30.0,
    max_dual_hspc_pct: float = 10.0,
    min_specificity_ratio: float = 3.0,
    min_efficacy_retention: float = 0.5
) -> List[Tuple[str, str]]:
    """
    Filter gene pairs by dual coverage in LSPC, HSPC, and specificity ratio.
    Dual coverage = % of cells where BOTH genes are detected (raw count > 0).

    Parameters
    ----------
    gene_pairs : list of tuples
        All candidate gene pairs
    binary_expr : np.ndarray
        Binary expression matrix (cells × genes), 1 if raw count > 0
    expr_df : pd.DataFrame
        Expression dataframe (for gene name mapping)
    metadata : pd.DataFrame
        Cell metadata
    min_dual_lspc_pct : float
        Minimum dual coverage % in LSPC cells (default: 30%)
    max_dual_hspc_pct : float
        Maximum dual coverage % in HSPC cells (default: 10%)
    min_specificity_ratio : float
        Minimum specificity ratio (LSPC% / HSPC%) (default: 3.0)
    min_efficacy_retention : float
        Minimum ratio of dual coverage to anchor-only coverage in LSPC (default: 0.5)
        Ensures that adding a partner doesn't reduce anchor efficacy by more than 50%

    Returns
    -------
    list of tuples
        Filtered gene pairs meeting dual coverage thresholds
    """
    logger.info(f"Filtering pairs by dual coverage >= {min_dual_lspc_pct}% in LSPC, <= {max_dual_hspc_pct}% in HSPC, "
                f"specificity ratio > {min_specificity_ratio}, and efficacy retention >= {min_efficacy_retention}...")

    lspc_mask = metadata['status'] == 1
    hspc_mask = metadata['status'] == 0
    n_lspc = lspc_mask.sum()
    n_hspc = hspc_mask.sum()

    filtered_pairs = []

    for g1, g2 in gene_pairs:
        g1_idx = expr_df.columns.get_loc(g1)
        g2_idx = expr_df.columns.get_loc(g2)

        # Binary: both detected (raw count > 0)
        g1_binary_lspc = binary_expr[lspc_mask, g1_idx]
        g2_binary_lspc = binary_expr[lspc_mask, g2_idx]
        g1_binary_hspc = binary_expr[hspc_mask, g1_idx]
        g2_binary_hspc = binary_expr[hspc_mask, g2_idx]

        # Calculate anchor-only coverage in LSPC
        anchor_lspc = g1_binary_lspc.sum()
        anchor_lspc_pct = 100 * anchor_lspc / n_lspc

        # Calculate dual coverage
        dual_lspc = (g1_binary_lspc * g2_binary_lspc).sum()
        dual_hspc = (g1_binary_hspc * g2_binary_hspc).sum()

        dual_lspc_pct = 100 * dual_lspc / n_lspc
        dual_hspc_pct = 100 * dual_hspc / n_hspc

        # Calculate specificity ratio (avoid division by zero)
        specificity_ratio = dual_lspc_pct / (dual_hspc_pct + 0.01)

        # Calculate efficacy retention (dual coverage relative to anchor-only coverage)
        efficacy_retention = dual_lspc_pct / (anchor_lspc_pct + 0.01)

        if (dual_lspc_pct >= min_dual_lspc_pct and
            dual_hspc_pct <= max_dual_hspc_pct and
            specificity_ratio > min_specificity_ratio and
            efficacy_retention >= min_efficacy_retention):
            filtered_pairs.append((g1, g2))

    logger.info(f"Pairs after dual coverage filter: {len(filtered_pairs)} / {len(gene_pairs)} "
                f"({100*len(filtered_pairs)/len(gene_pairs):.1f}% retained)")

    return filtered_pairs


# ============================================================================
# MODEL FITTING AND METRICS
# ============================================================================

def calculate_coverage(
    expr_df: pd.DataFrame,
    metadata: pd.DataFrame,
    gene1: str,
    gene2: str,
    threshold: float = 0
) -> Dict[str, float]:
    """
    Calculate coverage metrics for a gene pair.

    Parameters
    ----------
    expr_df : pd.DataFrame
        Expression matrix
    metadata : pd.DataFrame
        Cell metadata
    gene1, gene2 : str
        Gene names
    threshold : float
        Expression threshold (log1p CPM), default 0 (equivalent to raw count > 0)

    Returns
    -------
    dict with coverage metrics
    """
    lspc_cells = metadata[metadata['status'] == 1].index
    hspc_cells = metadata[metadata['status'] == 0].index

    # Both genes expressed
    lspc_both = (
        (expr_df.loc[lspc_cells, gene1] > threshold) &
        (expr_df.loc[lspc_cells, gene2] > threshold)
    ).sum()

    hspc_both = (
        (expr_df.loc[hspc_cells, gene1] > threshold) &
        (expr_df.loc[hspc_cells, gene2] > threshold)
    ).sum()

    lspc_coverage = lspc_both / len(lspc_cells)
    hspc_coverage = hspc_both / len(hspc_cells)
    specificity_ratio = lspc_coverage / (hspc_coverage + 0.0001)

    return {
        'lspc_coverage': lspc_coverage,
        'hspc_coverage': hspc_coverage,
        'specificity_ratio': specificity_ratio,
        'lspc_n_both': lspc_both,
        'hspc_n_both': hspc_both
    }


def fit_gene_pair_model_hybrid(
    gene1: str,
    gene2: str,
    expr_df: pd.DataFrame,
    binary_expr: np.ndarray,
    metadata: pd.DataFrame
) -> Dict:
    """
    Fit binomial mixed-effects model with continuous predictors:
    - Main effects (gene1, gene2): continuous log1p(CPM) expression
    - Interaction term: continuous (gene1 * gene2 expression levels)

    Model: status ~ gene1 + gene2 + gene1:gene2 + (1|Study_Donor)

    Parameters
    ----------
    gene1, gene2 : str
        Gene names
    expr_df : pd.DataFrame
        Continuous expression matrix (log1p CPM)
    binary_expr : np.ndarray
        Binary expression matrix (raw count > 0)
    metadata : pd.DataFrame
        Cell metadata (must have 'Study_Donor' column)

    Returns
    -------
    dict with model results and metrics
    """
    try:
        # Prepare data
        df = metadata.copy()

        # Continuous main effects
        df['gene1'] = expr_df[gene1].values
        df['gene2'] = expr_df[gene2].values

        # Continuous interaction term (expression level multiplied)
        df['gene1_x_gene2'] = df['gene1'] * df['gene2']

        # Use logistic regression with clustered standard errors
        formula = "status ~ gene1 + gene2 + gene1_x_gene2"
        model = smf.logit(formula, df)
        result = model.fit(
            cov_type='cluster',
            cov_kwds={'groups': df['Study_Donor']},
            disp=False,
            maxiter=100
        )
        model_type = 'logit_clustered'

        # Extract coefficients and p-values
        params = result.params
        pvalues = result.pvalues

        gene1_coef = params.get('gene1', np.nan)
        gene2_coef = params.get('gene2', np.nan)
        interaction_coef = params.get('gene1_x_gene2', np.nan)

        gene1_pval = pvalues.get('gene1', np.nan)
        gene2_pval = pvalues.get('gene2', np.nan)
        interaction_pval = pvalues.get('gene1_x_gene2', np.nan)

        # Calculate predicted probabilities for ROC-AUC
        y_true = df['status'].values
        y_pred = result.predict(df)

        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = np.nan

        # Calculate coverage metrics
        coverage = calculate_coverage(expr_df, metadata, gene1, gene2)

        # Calculate mean expression and correlation
        lspc_cells = metadata[metadata['status'] == 1].index
        hspc_cells = metadata[metadata['status'] == 0].index

        mean_gene1_lspc = expr_df.loc[lspc_cells, gene1].mean()
        mean_gene1_hspc = expr_df.loc[hspc_cells, gene1].mean()
        mean_gene2_lspc = expr_df.loc[lspc_cells, gene2].mean()
        mean_gene2_hspc = expr_df.loc[hspc_cells, gene2].mean()

        corr_lspc = expr_df.loc[lspc_cells, [gene1, gene2]].corr().iloc[0, 1]
        corr_hspc = expr_df.loc[hspc_cells, [gene1, gene2]].corr().iloc[0, 1]

        # Calculate anchor-specific metrics
        anchor_lspc = (expr_df.loc[lspc_cells, gene1] > 0).sum() / len(lspc_cells)
        anchor_hspc = (expr_df.loc[hspc_cells, gene1] > 0).sum() / len(hspc_cells)
        anchor_specificity = anchor_lspc / (anchor_hspc + 0.001)

        # Metrics for composite score
        efficacy_retention = coverage['lspc_coverage'] / (anchor_lspc + 0.001)
        specificity_gain = coverage['specificity_ratio'] - anchor_specificity

        return {
            'gene1': gene1,
            'gene2': gene2,
            'gene1_coef': gene1_coef,
            'gene2_coef': gene2_coef,
            'interaction_coef': interaction_coef,
            'gene1_pval': gene1_pval,
            'gene2_pval': gene2_pval,
            'interaction_pval': interaction_pval,
            'roc_auc': auc,
            'aic': result.aic,
            'bic': result.bic,
            'model_type': model_type,
            'converged': result.converged,
            **coverage,
            'mean_gene1_lspc': mean_gene1_lspc,
            'mean_gene1_hspc': mean_gene1_hspc,
            'mean_gene2_lspc': mean_gene2_lspc,
            'mean_gene2_hspc': mean_gene2_hspc,
            'corr_gene1_gene2_lspc': corr_lspc,
            'corr_gene1_gene2_hspc': corr_hspc,
            'anchor_lspc_coverage': anchor_lspc,
            'anchor_specificity': anchor_specificity,
            'efficacy_retention': efficacy_retention,
            'specificity_gain': specificity_gain
        }

    except Exception as e:
        logger.warning(f"Failed to fit model for {gene1} × {gene2}: {str(e)}")
        return {
            'gene1': gene1,
            'gene2': gene2,
            'gene1_coef': np.nan,
            'gene2_coef': np.nan,
            'interaction_coef': np.nan,
            'gene1_pval': np.nan,
            'gene2_pval': np.nan,
            'interaction_pval': np.nan,
            'roc_auc': np.nan,
            'aic': np.nan,
            'bic': np.nan,
            'model_type': 'failed',
            'converged': False,
            'lspc_coverage': np.nan,
            'hspc_coverage': np.nan,
            'specificity_ratio': np.nan,
            'lspc_n_both': np.nan,
            'hspc_n_both': np.nan,
            'mean_gene1_lspc': np.nan,
            'mean_gene1_hspc': np.nan,
            'mean_gene2_lspc': np.nan,
            'mean_gene2_hspc': np.nan,
            'corr_gene1_gene2_lspc': np.nan,
            'corr_gene1_gene2_hspc': np.nan,
            'anchor_lspc_coverage': np.nan,
            'anchor_specificity': np.nan,
            'efficacy_retention': np.nan,
            'specificity_gain': np.nan
        }


def fit_gene_pair_wrapper(args):
    """Wrapper for multiprocessing."""
    gene1, gene2, expr_df, binary_expr, metadata = args
    return fit_gene_pair_model_hybrid(gene1, gene2, expr_df, binary_expr, metadata)


# ============================================================================
# FDR CORRECTION
# ============================================================================


def apply_fdr_correction(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply FDR correction to interaction p-values.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results with p-values

    Returns
    -------
    results_df : pd.DataFrame
        Results with FDR q-values
    """
    from statsmodels.stats.multitest import multipletests

    # Only correct non-NaN p-values
    valid_mask = ~results_df['interaction_pval'].isna()
    pvals = results_df.loc[valid_mask, 'interaction_pval'].values

    _, qvals, _, _ = multipletests(pvals, method='fdr_bh')

    results_df['interaction_qval'] = np.nan
    results_df.loc[valid_mask, 'interaction_qval'] = qvals

    n_significant = (qvals < 0.05).sum()
    logger.info(f"Pairs with FDR < 0.05: {n_significant}")

    return results_df


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_top_pairs_heatmap(
    results_df: pd.DataFrame,
    expr_df: pd.DataFrame,
    metadata: pd.DataFrame,
    output_dir: Path,
    n_top: int = 20
):
    """Create heatmap showing expression of top gene pairs."""
    logger.info(f"Creating heatmap for top {n_top} pairs")

    top_pairs = results_df.head(n_top)
    all_genes = list(set(top_pairs['gene1'].tolist() + top_pairs['gene2'].tolist()))

    # Aggregate by cell type
    lspc_cells = metadata[metadata['status'] == 1].index
    hspc_cells = metadata[metadata['status'] == 0].index

    mean_expr = pd.DataFrame({
        'LSPC': expr_df.loc[lspc_cells, all_genes].mean(),
        'HSPC': expr_df.loc[hspc_cells, all_genes].mean()
    })

    fig, ax = plt.subplots(figsize=(6, max(8, len(all_genes) * 0.3)))
    sns.heatmap(
        mean_expr.T,
        cmap='RdYlBu_r',
        center=0,
        cbar_kws={'label': 'Mean log1p(CPM)'},
        ax=ax
    )
    ax.set_title(f'Top {n_top} Gene Pairs - Mean Expression')
    ax.set_xlabel('Gene')
    ax.set_ylabel('Cell Type')

    plt.tight_layout()
    plt.savefig(output_dir / 'top_pairs_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def plot_interaction_scatter(
    results_df: pd.DataFrame,
    expr_df: pd.DataFrame,
    metadata: pd.DataFrame,
    output_dir: Path,
    n_top: int = 10
):
    """Create 2D scatter plots for top gene pairs."""
    logger.info(f"Creating scatter plots for top {n_top} pairs")

    top_pairs = results_df.head(n_top)

    n_cols = 5
    n_rows = int(np.ceil(n_top / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten() if n_top > 1 else [axes]

    for idx, (_, row) in enumerate(top_pairs.iterrows()):
        if idx >= len(axes):
            break

        ax = axes[idx]
        gene1, gene2 = row['gene1'], row['gene2']

        # Get expression values
        x = expr_df[gene1].values
        y = expr_df[gene2].values
        colors = metadata['status'].values

        # Scatter plot
        ax.scatter(
            x, y,
            c=colors,
            cmap='RdBu_r',
            alpha=0.3,
            s=1,
            rasterized=True
        )

        ax.set_xlabel(f'{gene1} [log1p(CPM)]')
        ax.set_ylabel(f'{gene2} [log1p(CPM)]')
        ax.set_title(
            f'{gene1} × {gene2}\n'
            f'AUC={row["roc_auc"]:.3f}, '
            f'Int. coef={row["interaction_coef"]:.3f}',
            fontsize=9
        )

    # Remove empty subplots
    for idx in range(len(top_pairs), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(output_dir / 'top_pairs_scatter.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def plot_volcano(results_df: pd.DataFrame, output_dir: Path):
    """Create volcano plot of interaction coefficients vs p-values."""
    logger.info("Creating volcano plot")

    df = results_df.copy()
    df['-log10_pval'] = -np.log10(df['interaction_pval'] + 1e-300)

    fig, ax = plt.subplots(figsize=(10, 8))

    # All points
    ax.scatter(
        df['interaction_coef'],
        df['-log10_pval'],
        alpha=0.3,
        s=10,
        c='gray',
        label='All pairs'
    )

    # Significant points (FDR < 0.05)
    sig = df[df['interaction_qval'] < 0.05]
    if len(sig) > 0:
        ax.scatter(
            sig['interaction_coef'],
            sig['-log10_pval'],
            alpha=0.6,
            s=20,
            c='red',
            label=f'FDR < 0.05 (n={len(sig)})'
        )

    ax.axhline(-np.log10(0.05), color='blue', linestyle='--', alpha=0.5, label='p=0.05')
    ax.axvline(0, color='black', linestyle='-', alpha=0.3)

    ax.set_xlabel('Interaction Coefficient')
    ax.set_ylabel('-log10(p-value)')
    ax.set_title('Gene Pair Interaction Effects (Continuous Interaction Term)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'volcano_plot.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curves(
    results_df: pd.DataFrame,
    expr_df: pd.DataFrame,
    binary_expr: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: Path,
    n_top: int = 10
):
    """Plot ROC curves for top gene pairs."""
    logger.info(f"Creating ROC curves for top {n_top} pairs")

    top_pairs = results_df.head(n_top)

    fig, ax = plt.subplots(figsize=(8, 8))

    for idx, (_, row) in enumerate(top_pairs.iterrows()):
        gene1, gene2 = row['gene1'], row['gene2']

        # Refit model to get predictions
        df = metadata.copy()
        df['gene1'] = expr_df[gene1].values
        df['gene2'] = expr_df[gene2].values

        # Continuous interaction
        df['gene1_x_gene2'] = df['gene1'] * df['gene2']

        try:
            # Use same model specification as main analysis
            formula = "status ~ gene1 + gene2 + gene1_x_gene2"
            model = smf.logit(formula, df)
            result = model.fit(
                cov_type='cluster',
                cov_kwds={'groups': df['Study_Donor']},
                disp=False,
                maxiter=50
            )

            y_true = df['status'].values
            y_pred = result.predict(df)

            fpr, tpr, _ = roc_curve(y_true, y_pred)

            ax.plot(
                fpr, tpr,
                label=f'{gene1}×{gene2} (AUC={row["roc_auc"]:.3f})',
                alpha=0.7
            )
        except:
            continue

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves - Top {n_top} Gene Pairs (Continuous Interaction)')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves_top10.pdf', dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def main():
    """Main analysis pipeline with continuous interaction model."""

    N_CORES = mp.cpu_count() - 1

    logger.info("="*80)
    logger.info("STATISTICAL INTERACTION ANALYSIS - BINOMIAL MIXED MODEL")
    logger.info("Continuous main effects + Continuous interaction term")
    logger.info("="*80)

    # Step 1: Load and prepare data
    expr_df, metadata, binary_expr = load_and_prepare_data(ADATA_PATH)

    # Step 2: Select gene pairs
    anchors, partners, gene_pairs = select_gene_pairs(
        DESEQ_PATH, expr_df, metadata
    )

    logger.info(f"Anchor genes: {anchors}")

    # Step 3: PRE-FILTER by dual coverage (>= 30% LSPC, <= 10% HSPC, specificity ratio > 3)
    gene_pairs_filtered = filter_by_dual_coverage(
        gene_pairs, binary_expr, expr_df, metadata,
        min_dual_lspc_pct=30.0,
        max_dual_hspc_pct=10.0,
        min_specificity_ratio=3.0
    )

    # Step 4: Run binomial mixed models (parallel)
    logger.info(f"\n{'='*80}")
    logger.info(f"Running binomial mixed models on {len(gene_pairs_filtered)} pairs using {N_CORES} cores")
    logger.info(f"{'='*80}")

    pool_args = [(g1, g2, expr_df, binary_expr, metadata) for g1, g2 in gene_pairs_filtered]

    with mp.Pool(N_CORES) as pool:
        results = list(tqdm(
            pool.imap(fit_gene_pair_wrapper, pool_args),
            total=len(gene_pairs_filtered),
            desc="Fitting binomial models"
        ))

    results_df = pd.DataFrame(results)

    # Step 5: Apply FDR correction (for reference only)
    logger.info("\nApplying FDR correction (for reference)...")
    results_df = apply_fdr_correction(results_df)

    n_significant_fdr = (results_df['interaction_qval'] < 0.05).sum()
    logger.info(f"Pairs with FDR < 0.05: {n_significant_fdr} (not used for filtering)")

    # Step 6: Save output files
    logger.info("\nSaving results...")

    # All gene pairs tested
    results_df.to_csv(OUTPUT_DIR / 'all_gene_pairs.csv', index=False)
    logger.info(f"Saved: {OUTPUT_DIR / 'all_gene_pairs.csv'}")

    # Positive interaction coefficient only (ignoring p-value)
    significant_df = results_df[
        (results_df['interaction_coef'] > 0)
    ].copy()

    # Remove duplicate pairs: keep orientation with higher lspc_coverage
    significant_df['pair_key'] = significant_df.apply(
        lambda row: tuple(sorted([row['gene1'], row['gene2']])), axis=1
    )
    significant_df = significant_df.sort_values('lspc_coverage', ascending=False)
    significant_df = significant_df.drop_duplicates(subset='pair_key', keep='first')
    significant_df = significant_df.drop(columns='pair_key')
    significant_df.to_csv(OUTPUT_DIR / 'positive_interactions.csv', index=False)
    logger.info(f"Saved: {OUTPUT_DIR / 'positive_interactions.csv'} ({len(significant_df)} unique pairs)")

    # Step 7: Generate visualizations
    logger.info("\nGenerating visualizations...")

    if len(significant_df) > 0:
        # Already sorted by lspc_coverage
        plot_interaction_scatter(significant_df, expr_df, metadata, FIGURES_DIR, n_top=10)
        plot_roc_curves(significant_df, expr_df, binary_expr, metadata, FIGURES_DIR, n_top=10)

    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"\nResults saved to: {OUTPUT_DIR}")
    logger.info(f"Total pairs tested: {len(results_df)}")
    logger.info(f"Pairs with positive interaction coefficient (unique): {len(significant_df)}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
