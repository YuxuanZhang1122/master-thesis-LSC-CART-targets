import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from CNV_MapGenes import prepare_anndata_for_cnv, quick_gene_position_test

# Set up scanpy settings
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white')

# Load the AML328_purified dataset
print("Loading AML328_purified dataset...")
adata = sc.read_h5ad('/Users/axuan/Desktop/Thesis/AML328_purified.h5ad')

print(f"Dataset shape: {adata.shape}")
print(f"Number of cells: {adata.n_obs}")
print(f"Number of genes: {adata.n_vars}")
print("\nFirst few gene names:")
print(list(adata.var_names[:10]))

# Quick test to see gene mapping success rate
print("\n" + "="*50)
print("TESTING GENE MAPPING...")
print("="*50)
test_positions = quick_gene_position_test(adata, n_test_genes=20)

# Run the full gene mapping and preparation
print("\n" + "="*50) 
print("RUNNING FULL GENE MAPPING...")
print("="*50)
adata_filtered, gene_order = prepare_anndata_for_cnv(adata, output_dir='cnv_analysis')
print(f"\nFiltered dataset shape: {adata_filtered.shape}")
print(f"Genes with chromosomal positions: {len(gene_order)}")

# Create chromosome-wise visualization
print("\n" + "="*50)
print("CREATING CHROMOSOME VISUALIZATIONS...")
print("="*50)

def visualize_chromosome_counts(adata, gene_order=None, n_cells_sample=100):
    """
    Visualize counts across chromosomes for individual cells
    """
    
    if gene_order is not None:
        # Use mapped genes organized by chromosome
        chr_data = gene_order.copy()
        
        # Sample cells for visualization (too many cells makes plot unreadable)
        if adata.n_obs > n_cells_sample:
            cell_indices = np.random.choice(adata.n_obs, n_cells_sample, replace=False)
            adata_vis = adata[cell_indices, :].copy()
            print(f"Sampling {n_cells_sample} cells for visualization")
        else:
            adata_vis = adata.copy()
            print(f"Using all {adata.n_obs} cells for visualization")
        
        # Calculate mean expression per chromosome per cell
        chromosomes = chr_data['chromosome'].unique()
        chr_means = []
        
        for chr_name in sorted(chromosomes, key=lambda x: (x.isdigit() and int(x) or 99, x)):
            chr_genes = chr_data[chr_data['chromosome'] == chr_name]['gene_symbol'].tolist()
            # Find genes that exist in our data
            existing_genes = [g for g in chr_genes if g in adata_vis.var_names]
            
            if existing_genes:
                # Calculate mean expression for this chromosome
                chr_expr = adata_vis[:, existing_genes].X.mean(axis=1)
                if hasattr(chr_expr, 'A1'):  # Handle sparse matrices
                    chr_expr = chr_expr.A1
                chr_means.append({
                    'chromosome': chr_name,
                    'mean_expression': chr_expr,
                    'n_genes': len(existing_genes)
                })
                print(f"Chr {chr_name}: {len(existing_genes)} genes")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Chromosome Expression Analysis - AML328_purified\n({adata_vis.n_obs} cells, {len(gene_order)} genes with positions)', fontsize=14)
        
        # 1. Heatmap of mean expression per chromosome per cell
        chr_matrix = np.column_stack([data['mean_expression'] for data in chr_means])
        chr_names = [data['chromosome'] for data in chr_means]
        
        im = axes[0,0].imshow(chr_matrix.T, aspect='auto', cmap='viridis')
        axes[0,0].set_title('Mean Expression per Chromosome per Cell')
        axes[0,0].set_xlabel('Cells')
        axes[0,0].set_ylabel('Chromosomes')
        axes[0,0].set_yticks(range(len(chr_names)))
        axes[0,0].set_yticklabels(chr_names)
        plt.colorbar(im, ax=axes[0,0], label='Mean Expression')
        
        # 2. Distribution of chromosome expression across all cells
        chr_expr_df = pd.DataFrame({
            data['chromosome']: data['mean_expression'] 
            for data in chr_means
        })
        
        # Box plot
        chr_expr_df.boxplot(ax=axes[0,1])
        axes[0,1].set_title('Expression Distribution per Chromosome')
        axes[0,1].set_xlabel('Chromosome')
        axes[0,1].set_ylabel('Mean Expression')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Mean expression per chromosome (averaged across all cells)
        chr_overall_means = [data['mean_expression'].mean() for data in chr_means]
        bars = axes[1,0].bar(chr_names, chr_overall_means)
        axes[1,0].set_title('Overall Mean Expression per Chromosome')
        axes[1,0].set_xlabel('Chromosome')
        axes[1,0].set_ylabel('Mean Expression')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Color bars by value
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # 4. Number of genes per chromosome
        n_genes_per_chr = [data['n_genes'] for data in chr_means]
        axes[1,1].bar(chr_names, n_genes_per_chr, color='orange', alpha=0.7)
        axes[1,1].set_title('Number of Genes per Chromosome')
        axes[1,1].set_xlabel('Chromosome')
        axes[1,1].set_ylabel('Number of Genes')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('cnv_analysis/chromosome_expression_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Additional detailed cell-wise visualization
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        
        # Create a more detailed heatmap with better cell sampling
        if adata_vis.n_obs > 50:
            # Sample cells more strategically
            cell_sample = np.random.choice(adata_vis.n_obs, min(50, adata_vis.n_obs), replace=False)
            chr_matrix_detailed = chr_matrix[cell_sample, :]
        else:
            chr_matrix_detailed = chr_matrix
            
        sns.heatmap(chr_matrix_detailed.T, 
                   xticklabels=False,
                   yticklabels=chr_names,
                   cmap='RdBu_r', center=0,
                   ax=ax2)
        ax2.set_title(f'Cell-wise Chromosome Expression Heatmap\n({chr_matrix_detailed.shape[0]} cells)')
        ax2.set_xlabel('Individual Cells')
        ax2.set_ylabel('Chromosomes')
        
        plt.tight_layout()
        plt.savefig('cnv_analysis/cell_wise_chromosome_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return chr_means, chr_expr_df
        
    else:
        print("No gene order information available. Creating basic visualization...")
        # Fallback visualization without chromosome mapping
        plt.figure(figsize=(10, 6))
        plt.hist(np.array(adata.X.sum(axis=1)).flatten(), bins=50, alpha=0.7)
        plt.title('Total UMI counts per cell')
        plt.xlabel('Total UMI counts')
        plt.ylabel('Number of cells')
        plt.savefig('cnv_analysis/basic_umi_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

# Run the visualization
if gene_order is not None:
    chr_results, chr_df = visualize_chromosome_counts(adata_filtered, gene_order)
    
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    print(f"Successfully mapped {len(gene_order)} genes to chromosomes")
    print(f"Chromosomes analyzed: {len(chr_results)}")
    print("\nMean expression per chromosome:")
    for result in chr_results:
        mean_val = result['mean_expression'].mean()
        print(f"  Chr {result['chromosome']}: {mean_val:.3f} (n_genes={result['n_genes']})")
        
else:
    visualize_chromosome_counts(adata_filtered, gene_order)
    print("Gene mapping failed. Basic visualization completed.")

print("\n" + "="*50)
print("ANALYSIS COMPLETE!")
print("="*50)
print("Output files saved to 'cnv_analysis/' directory")
print("Visualizations saved as PNG files")