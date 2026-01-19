#!/usr/bin/env python3
"""Filter low-gene cells and save filtered h5ad."""

import numpy as np
import scanpy as sc
from pathlib import Path

INPUT_PATH = "HSC_MPP_full.h5ad"
OUTPUT_PATH = "HSC_MPP_full_filtered.h5ad"
MIN_GENES = 100

print("="*70)
print("FILTERING AND SAVING H5AD")
print("="*70)

# Load data
print(f"\nLoading {INPUT_PATH}")
adata = sc.read_h5ad(INPUT_PATH)
print(f"Original cells: {adata.n_obs}")
print(f"Genes: {adata.n_vars}")

# Calculate genes detected per cell
X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
genes_detected = (X > 0).sum(axis=1)

print(f"\nGenes detected per cell:")
print(f"  Min: {genes_detected.min()}")
print(f"  Median: {np.median(genes_detected):.0f}")
print(f"  Mean: {genes_detected.mean():.1f}")
print(f"  Max: {genes_detected.max()}")

# Apply QC filter
print(f"\nApplying QC filter: min_genes >= {MIN_GENES}")
qc_pass = genes_detected >= MIN_GENES

n_filtered = (~qc_pass).sum()
pct_filtered = n_filtered / adata.n_obs * 100
print(f"Filtering {n_filtered:,} cells ({pct_filtered:.1f}%)")

# Per cell type breakdown
if 'consensus_label_5votes' in adata.obs.columns:
    print(f"\nFiltering breakdown by cell type:")
    for celltype in ['LSPC', 'HSPC']:
        ct_mask = adata.obs['consensus_label_5votes'] == celltype
        ct_filtered = ct_mask & (~qc_pass)
        n_ct_filtered = ct_filtered.sum()
        n_ct_total = ct_mask.sum()
        pct_ct = n_ct_filtered / n_ct_total * 100 if n_ct_total > 0 else 0
        print(f"  {celltype}: {n_ct_filtered:,}/{n_ct_total:,} filtered ({pct_ct:.1f}%)")

# Per study breakdown
if 'Study' in adata.obs.columns:
    print(f"\nFiltering breakdown by study:")
    studies = sorted(adata.obs['Study'].unique())
    for study in studies:
        study_mask = adata.obs['Study'] == study
        study_filtered = study_mask & (~qc_pass)
        n_study_filtered = study_filtered.sum()
        n_study_total = study_mask.sum()
        pct_study = n_study_filtered / n_study_total * 100
        print(f"  {study:<15s}: {n_study_filtered:5,}/{n_study_total:5,} filtered ({pct_study:5.1f}%)")

# Filter adata
adata_filtered = adata[qc_pass, :].copy()

print(f"\nFiltered cells: {adata_filtered.n_obs:,}")

# Post-filtering stats
if 'consensus_label_5votes' in adata_filtered.obs.columns:
    lspc_count = (adata_filtered.obs['consensus_label_5votes'] == 'LSPC').sum()
    hspc_count = (adata_filtered.obs['consensus_label_5votes'] == 'HSPC').sum()
    print(f"  LSPC: {lspc_count:,}")
    print(f"  HSPC: {hspc_count:,}")

# Save filtered data
print(f"\nSaving to {OUTPUT_PATH}")
adata_filtered.write_h5ad(OUTPUT_PATH)

print(f"\n✓ Successfully saved filtered h5ad")
print("="*70)
