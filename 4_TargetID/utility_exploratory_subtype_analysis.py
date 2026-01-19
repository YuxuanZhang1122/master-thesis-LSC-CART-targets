import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sc.set_figure_params(dpi=150, facecolor='white', figsize=(10, 8))
sns.set_style("whitegrid")

# Load data
adata = sc.read_h5ad('subtype_analysis.h5ad')

print("=" * 80)
print("EXPLORATORY ANALYSIS: NPM1 & ELN2017 SUBTYPES")
print("=" * 80)

# === DONOR LEVEL SUMMARY ===
print("\n1. DONOR-LEVEL SUMMARY")
print("-" * 80)
donor_meta = adata.obs.groupby('Donor').first()[['NPM1_Mutation', 'ELN2017', 'Study']]
print(f"Total donors: {len(donor_meta)}")
print(f"\nNPM1 status by donor:")
print(donor_meta['NPM1_Mutation'].value_counts())
print(f"\nELN2017 by donor:")
print(donor_meta['ELN2017'].value_counts())

# === CELL LEVEL SUMMARY ===
print("\n2. CELL-LEVEL SUMMARY")
print("-" * 80)
print(f"Total cells: {adata.n_obs}")
print(f"\nNPM1 status by cell:")
print(adata.obs['NPM1_Mutation'].value_counts())
print(f"\nELN2017 by cell:")
print(adata.obs['ELN2017'].value_counts())

# === CHECK GENE PRESENCE ===
print("\n3. GENE PRESENCE CHECK")
print("-" * 80)
genes_of_interest = ['CD9', 'EMB']
for gene in genes_of_interest:
    if gene in adata.var_names:
        print(f"✓ {gene} found in dataset")
    else:
        print(f"✗ {gene} NOT found in dataset")

available_genes = [g for g in genes_of_interest if g in adata.var_names]

if len(available_genes) == 0:
    print("\nNo genes found. Exiting.")
    exit()

# === EXPRESSION STATISTICS ===
print("\n4. EXPRESSION STATISTICS")
print("-" * 80)

for gene in available_genes:
    expr = adata[:, gene].X.toarray().flatten()
    print(f"\n{gene} expression:")
    print(f"  Mean: {expr.mean():.3f}")
    print(f"  Median: {np.median(expr):.3f}")
    print(f"  % expressing (>0): {(expr > 0).sum() / len(expr) * 100:.1f}%")

    # By NPM1 status
    print(f"\n  By NPM1 status:")
    for npm1 in ['Pos', 'Neg']:
        expr_npm1 = adata[adata.obs['NPM1_Mutation'] == npm1, gene].X.toarray().flatten()
        print(f"    {npm1}: mean={expr_npm1.mean():.3f}, median={np.median(expr_npm1):.3f}, %expr={(expr_npm1 > 0).sum() / len(expr_npm1) * 100:.1f}%")

    # Statistical test
    pos_expr = adata[adata.obs['NPM1_Mutation'] == 'Pos', gene].X.toarray().flatten()
    neg_expr = adata[adata.obs['NPM1_Mutation'] == 'Neg', gene].X.toarray().flatten()
    u_stat, p_val = stats.mannwhitneyu(pos_expr, neg_expr, alternative='two-sided')
    print(f"    Mann-Whitney U test p-value: {p_val:.2e}")

    # By ELN2017
    print(f"\n  By ELN2017 classification:")
    for eln in ['Favorable', 'Intermediate', 'Adverse']:
        expr_eln = adata[adata.obs['ELN2017'] == eln, gene].X.toarray().flatten()
        print(f"    {eln}: mean={expr_eln.mean():.3f}, median={np.median(expr_eln):.3f}, %expr={(expr_eln > 0).sum() / len(expr_eln) * 100:.1f}%")

    # Kruskal-Wallis test
    fav_expr = adata[adata.obs['ELN2017'] == 'Favorable', gene].X.toarray().flatten()
    int_expr = adata[adata.obs['ELN2017'] == 'Intermediate', gene].X.toarray().flatten()
    adv_expr = adata[adata.obs['ELN2017'] == 'Adverse', gene].X.toarray().flatten()
    h_stat, p_val = stats.kruskal(fav_expr, int_expr, adv_expr)
    print(f"    Kruskal-Wallis test p-value: {p_val:.2e}")

# === VISUALIZATIONS ===
print("\n5. GENERATING VISUALIZATIONS")
print("-" * 80)

# Create output directory
import os
os.makedirs('Outputs/exploratory_subtype', exist_ok=True)

# Figure 1: Summary bar plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

npm1_counts = adata.obs['NPM1_Mutation'].value_counts()
axes[0].bar(npm1_counts.index, npm1_counts.values, color=['#e74c3c', '#3498db'])
axes[0].set_ylabel('Number of cells')
axes[0].set_xlabel('NPM1 Mutation Status')
axes[0].set_title('Cell Distribution by NPM1 Status')
for i, v in enumerate(npm1_counts.values):
    axes[0].text(i, v - 200, str(v), ha='center', va='bottom')

eln_counts = adata.obs['ELN2017'].value_counts()
colors = {'Favorable': '#2ecc71', 'Intermediate': '#f39c12', 'Adverse': '#e74c3c'}
axes[1].bar(range(len(eln_counts)), eln_counts.values,
           color=[colors[x] for x in eln_counts.index])
axes[1].set_xticks(range(len(eln_counts)))
axes[1].set_xticklabels(eln_counts.index)#, rotation=45)
axes[1].set_ylabel('Number of cells')
axes[1].set_xlabel('ELN2017 Classification')
axes[1].set_title('Cell Distribution by ELN2017')
for i, v in enumerate(eln_counts.values):
    axes[1].text(i, v + 200, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('Outputs/exploratory_subtype/cell_distributions.pdf', bbox_inches='tight')
plt.savefig('Outputs/exploratory_subtype/cell_distributions.png', bbox_inches='tight', dpi=300)
plt.close()
print("  Saved: cell_distributions.pdf/.png")

# Figure 2: Violin plots by NPM1
if len(available_genes) > 0:
    fig, axes = plt.subplots(1, len(available_genes), figsize=(6*len(available_genes), 6))
    if len(available_genes) == 1:
        axes = [axes]

    for idx, gene in enumerate(available_genes):
        expr_data = []
        labels = []
        for npm1 in ['Pos', 'Neg']:
            expr = adata[adata.obs['NPM1_Mutation'] == npm1, gene].X.toarray().flatten()
            expr_data.append(expr)
            labels.append(npm1)

        parts = axes[idx].violinplot(expr_data, positions=[0, 1], showmeans=True, showmedians=True)
        axes[idx].set_xticks([0, 1])
        axes[idx].set_xticklabels(labels)
        axes[idx].set_ylabel('Expression level')
        axes[idx].set_xlabel('NPM1 Mutation Status')
        axes[idx].set_title(f'{gene} Expression by NPM1 Status')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Outputs/exploratory_subtype/expression_npm1_violin.pdf', bbox_inches='tight')
    plt.savefig('Outputs/exploratory_subtype/expression_npm1_violin.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("  Saved: expression_npm1_violin.pdf/.png")

# Figure 3: Violin plots by ELN2017
if len(available_genes) > 0:
    fig, axes = plt.subplots(1, len(available_genes), figsize=(6*len(available_genes), 6))
    if len(available_genes) == 1:
        axes = [axes]

    for idx, gene in enumerate(available_genes):
        expr_data = []
        labels = []
        for eln in ['Favorable', 'Intermediate', 'Adverse']:
            expr = adata[adata.obs['ELN2017'] == eln, gene].X.toarray().flatten()
            expr_data.append(expr)
            labels.append(eln)

        parts = axes[idx].violinplot(expr_data, positions=[0, 1, 2], showmeans=True, showmedians=True)
        axes[idx].set_xticks([0, 1, 2])
        axes[idx].set_xticklabels(labels)#, rotation=45)
        axes[idx].set_ylabel('Expression level')
        axes[idx].set_xlabel('ELN2017 Classification')
        axes[idx].set_title(f'{gene} Expression by ELN2017')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Outputs/exploratory_subtype/expression_eln2017_violin.pdf', bbox_inches='tight')
    plt.savefig('Outputs/exploratory_subtype/expression_eln2017_violin.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("  Saved: expression_eln2017_violin.pdf/.png")

# Figure 4: UMAP by metadata
if 'X_umap' in adata.obsm:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # NPM1 status
    for npm1, color in zip(['Pos', 'Neg'], ['#e74c3c', '#3498db']):
        mask = adata.obs['NPM1_Mutation'] == npm1
        axes[0].scatter(adata.obsm['X_umap'][mask, 0],
                       adata.obsm['X_umap'][mask, 1],
                       c=color, label=npm1, alpha=0.6, s=5)
    axes[0].set_xlabel('UMAP1')
    axes[0].set_ylabel('UMAP2')
    axes[0].set_title('UMAP colored by NPM1 Mutation Status')
    axes[0].legend()

    # ELN2017
    colors = {'Favorable': '#2ecc71', 'Intermediate': '#f39c12', 'Adverse': '#e74c3c'}
    for eln, color in colors.items():
        mask = adata.obs['ELN2017'] == eln
        axes[1].scatter(adata.obsm['X_umap'][mask, 0],
                       adata.obsm['X_umap'][mask, 1],
                       c=color, label=eln, alpha=0.6, s=5)
    axes[1].set_xlabel('UMAP1')
    axes[1].set_ylabel('UMAP2')
    axes[1].set_title('UMAP colored by ELN2017 Classification')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('Outputs/exploratory_subtype/umap_metadata.pdf', bbox_inches='tight')
    plt.savefig('Outputs/exploratory_subtype/umap_metadata.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("  Saved: umap_metadata.pdf/.png")
else:
    print("  Note: No UMAP coordinates found, skipping UMAP visualization")

# Figure 5: Gene expression UMAPs
if 'X_umap' in adata.obsm and len(available_genes) > 0:
    fig, axes = plt.subplots(1, len(available_genes), figsize=(8*len(available_genes), 6))
    if len(available_genes) == 1:
        axes = [axes]

    for idx, gene in enumerate(available_genes):
        expr = adata[:, gene].X.toarray().flatten()
        scatter = axes[idx].scatter(adata.obsm['X_umap'][:, 0],
                                    adata.obsm['X_umap'][:, 1],
                                    c=expr, cmap='viridis', s=5, alpha=0.7)
        axes[idx].set_xlabel('UMAP1')
        axes[idx].set_ylabel('UMAP2')
        axes[idx].set_title(f'{gene} Expression')
        plt.colorbar(scatter, ax=axes[idx], label='Expression')

    plt.tight_layout()
    plt.savefig('Outputs/exploratory_subtype/umap_gene_expression.pdf', bbox_inches='tight')
    plt.savefig('Outputs/exploratory_subtype/umap_gene_expression.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("  Saved: umap_gene_expression.pdf/.png")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
