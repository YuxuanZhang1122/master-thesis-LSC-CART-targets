"""
Step 1: Data Preparation for van Galen Random Forest Classifiers
================================================================

This script prepares the data for both classifiers:
- Filters to normal cells (15 cell types)
- Applies normalization (Cp10k, log-transform)
- Filters genes by mean expression > 0.01
- Saves processed data for downstream analysis
"""

import scanpy as sc
import pandas as pd
import numpy as np

def main():

    # Load raw data
    adata = sc.read('../vanGalen_raw.h5ad')

    # Filter to normal cells only
    normal_mask = ~adata.obs['GroundTruth'].isin(['mutant', 'unknown'])
    adata_normal = adata[normal_mask].copy()

    # Normalize to Cp10k and log transform
    sc.pp.normalize_total(adata_normal, target_sum=1e4)
    sc.pp.log1p(adata_normal)

    # Filter genes by mean expression > 0.01
    gene_means = np.array(adata_normal.X.mean(axis=0)).flatten()
    gene_mask = gene_means > 0.01
    adata_filtered = adata_normal[:, gene_mask].copy()
    adata_filtered.var['mean_expression'] = gene_means[gene_mask]

    print(f"Genes before filtering: {adata_normal.n_vars}")
    print(f"Genes after filtering: {adata_filtered.n_vars}")

    # Save processed data
    adata_filtered.write('results/data_processed_normal.h5ad')

if __name__ == "__main__":
    main()