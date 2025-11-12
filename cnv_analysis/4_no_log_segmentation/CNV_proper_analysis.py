import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

output_dir = 'cnv_analysis/4_no_log_segmentation'

def create_reference_profile(adata, normal_indices, gene_positions, chromosomes, window_size=100):
    """
    Create smoothed reference expression profile from normal cells
    """
    print("Creating reference profile from normal cells...")
    
    reference_profiles = {}
    
    for chr_name in chromosomes:
        # Get genes for this chromosome
        chr_genes = gene_positions[gene_positions['chromosome'] == chr_name]['gene_symbol'].tolist()
        chr_genes_in_data = [g for g in chr_genes if g in adata.var_names]
        
        if len(chr_genes_in_data) < 10:
            continue
            
        # Get gene positions and sort
        chr_positions = gene_positions[gene_positions['gene_symbol'].isin(chr_genes_in_data)].copy()
        chr_positions = chr_positions.sort_values('start').reset_index(drop=True)
        
        # Get gene indices in genomic order
        gene_indices = [list(adata.var_names).index(gene) for gene in chr_positions['gene_symbol']]
        
        # Extract expression data for normal cells
        if hasattr(adata.X, 'toarray'):
            chr_expr_normal = adata.X[normal_indices, :][:, gene_indices].toarray()
        else:
            chr_expr_normal = adata.X[normal_indices, :][:, gene_indices]
        
        # Calculate mean expression across normal cells
        normal_mean = chr_expr_normal.mean(axis=0)
        
        # Apply smoothing to the reference
        sigma = min(window_size, len(normal_mean)) / 6  # 99.7% of weight within window_size
        normal_smoothed = gaussian_filter1d(normal_mean, sigma=sigma, mode='nearest')
        
        reference_profiles[chr_name] = {
            'genes': chr_positions['gene_symbol'].tolist(),
            'positions': chr_positions['start'].values,
            'gene_indices': gene_indices,
            'reference_profile': normal_smoothed,
            'raw_profile': normal_mean
        }
        
        print(f"  Chr{chr_name}: {len(chr_genes_in_data)} genes")
    
    return reference_profiles

def calculate_cnv_ratios_from_means(adata, reference_profiles, normal_indices, malignant_indices, window_size=100):
    """
    Calculate log2 ratios by first computing group means, then taking ratio
    """
    print("Calculating CNV ratios from group means...")
    
    cnv_data = {}
    
    for chr_name, ref_data in reference_profiles.items():
        print(f"  Processing Chr{chr_name}...")
        
        gene_indices = ref_data['gene_indices']
        reference = ref_data['reference_profile']  # Already smoothed normal mean
        
        # Extract expression for malignant cells only
        if hasattr(adata.X, 'toarray'):
            malignant_expr = adata.X[malignant_indices, :][:, gene_indices].toarray()
        else:
            malignant_expr = adata.X[malignant_indices, :][:, gene_indices]
        
        # Calculate mean expression for malignant cells
        malignant_mean = malignant_expr.mean(axis=0)
        
        # Apply smoothing to malignant mean
        sigma = min(window_size, len(malignant_mean)) / 6
        malignant_smoothed = gaussian_filter1d(malignant_mean, sigma=sigma, mode='nearest')
        
        # Calculate log2 ratio of the two smoothed means
        log2_ratio_mean = np.log2((malignant_smoothed + 1e-8) / (reference + 1e-8))
        
        cnv_data[chr_name] = {
            'genes': ref_data['genes'],
            'positions': ref_data['positions'],
            'log2_ratio_mean': log2_ratio_mean,  # Single profile per chromosome
            'reference': reference,
            'malignant_mean': malignant_smoothed
        }
    
    return cnv_data

def segment_cnv_simple(log2_ratios, min_segment_size=10, threshold=0.3):
    """
    Simple segmentation based on consecutive gains/losses
    """
    segments = []
    current_segment_start = 0
    current_state = 'neutral'
    
    for i in range(len(log2_ratios)):
        if log2_ratios[i] > threshold:
            new_state = 'gain'
        elif log2_ratios[i] < -threshold:
            new_state = 'loss'
        else:
            new_state = 'neutral'
        
        if new_state != current_state and i - current_segment_start >= min_segment_size:
            # End current segment
            segments.append({
                'start': current_segment_start,
                'end': i,
                'state': current_state,
                'mean_log2': np.mean(log2_ratios[current_segment_start:i])
            })
            current_segment_start = i
            current_state = new_state
    
    # Add final segment
    if len(log2_ratios) - current_segment_start >= min_segment_size:
        segments.append({
            'start': current_segment_start,
            'end': len(log2_ratios),
            'state': current_state,
            'mean_log2': np.mean(log2_ratios[current_segment_start:])
        })
    
    return segments

# Load data
print("Loading data for proper CNV analysis...")
adata_filtered = sc.read_h5ad('cnv_analysis/filtered_adata.h5ad')
adata_original = sc.read_h5ad('/Users/axuan/Desktop/Thesis/AML328_purified.h5ad')
gene_positions = pd.read_csv('cnv_analysis/gene_positions.csv')

print(f"Dataset: {adata_filtered.shape[0]} cells, {adata_filtered.shape[1]} genes")

# Apply normalization: normalize to 10,000 counts per cell and log-transform
adata_normalized = adata_filtered.copy()
sc.pp.normalize_total(adata_normalized, target_sum=1e4)
#sc.pp.log1p(adata_normalized)

cell_types = adata_original.obs['CellType']
print(f"Cell type distribution: {cell_types.value_counts().to_dict()}")

# Normal cells: HSC, Prog
normal_mask = cell_types.isin(['HSC', 'Prog'])
normal_indices = np.where(normal_mask)[0]

# Malignant cells: HSC-like, Prog-like  
malignant_mask = cell_types.isin(['HSC-like', 'Prog-like'])
malignant_indices = np.where(malignant_mask)[0]

print(f"Normal cells (HSC + Prog): {len(normal_indices)}")
print(f"Malignant cells (HSC-like + Prog-like): {len(malignant_indices)}")

chromosomes = ['1','2','3','4','5','6', '7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']

print("\n" + "="*60)
print("CREATING REFERENCE PROFILE FROM NORMAL CELLS")
print("="*60)

reference_profiles = create_reference_profile(adata_normalized, normal_indices, gene_positions, chromosomes)

print("\n" + "="*60)
print("CALCULATING CNV RATIOS")
print("="*60)

cnv_data = calculate_cnv_ratios_from_means(adata_normalized, reference_profiles, normal_indices, malignant_indices)

# Process each chromosome separately (no concatenation)
print("\n" + "="*60)
print("ANALYZING EACH CHROMOSOME SEPARATELY")
print("="*60)

# Create individual plots for each chromosome
for chr_name in chromosomes:
    if chr_name not in cnv_data:
        continue
        
    print(f"Processing Chr{chr_name}...")
    
    # Get data for this chromosome
    chr_data = cnv_data[chr_name]
    log2_ratio = chr_data['log2_ratio_mean']
    reference_profile = chr_data['reference']
    malignant_profile = chr_data['malignant_mean']
    positions = chr_data['positions']
    
    # Apply segmentation
    segments = segment_cnv_simple(log2_ratio, min_segment_size=10, threshold=0.3)
    print(f"  Found {len(segments)} segments in Chr{chr_name}")
    
    # Create plot for this chromosome
    fig, axes = plt.subplots(4, 1, figsize=(16, 14))
    fig.suptitle(f'CNV Analysis - Chromosome {chr_name} (normalized, no log)', fontsize=16, y= 0.99)
    
    x_positions = np.arange(len(log2_ratio))
    
    # 1. Normal reference profile
    axes[0].plot(x_positions, reference_profile, 'g-', alpha=0.8, linewidth=2, label='Normal Reference')
    axes[0].set_title(f'Chr{chr_name} - Normal Reference Expression')
    axes[0].set_ylabel('Normalized Expression')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Malignant expression profile
    axes[1].plot(x_positions, malignant_profile, 'r-', alpha=0.8, linewidth=2, label='Malignant Mean')
    axes[1].set_title(f'Chr{chr_name} - Malignant Mean Expression')
    axes[1].set_ylabel('Normalized Expression')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Direct comparison
    axes[2].plot(x_positions, reference_profile, 'g-', alpha=0.8, linewidth=2, label='Normal Reference')
    axes[2].plot(x_positions, malignant_profile, 'r-', alpha=0.8, linewidth=2, label='Malignant Mean')
    axes[2].set_title(f'Chr{chr_name} - Expression Comparison')
    axes[2].set_ylabel('Normalized Expression')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 4. Log2 ratio with segmentation
    axes[3].plot(x_positions, log2_ratio, 'purple', alpha=0.8, linewidth=2, label='Log2 Ratio')
    axes[3].axhline(y=0.3, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Gain threshold')
    axes[3].axhline(y=-0.3, color='blue', linestyle='--', alpha=0.7, linewidth=2, label='Loss threshold')
    axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add segment highlighting
    for seg in segments:
        start, end = seg['start'], seg['end']
        state = seg['state']
        if state == 'gain':
            color = 'red'
            alpha = 0.8
            linewidth = 4
        elif state == 'loss':
            color = 'blue'
            alpha = 0.8
            linewidth = 4
        else:
            continue  # Don't highlight neutral regions
        
        # Highlight the actual data line
        axes[3].plot(x_positions[start:end], log2_ratio[start:end], 
                    color=color, alpha=alpha, linewidth=linewidth, zorder=5)
        
        # Add segment boundaries
        axes[3].axvline(start, color='black', linestyle=':', alpha=0.7, linewidth=1)
    
    axes[3].set_title(f'Chr{chr_name} - CNV Segmentation ({len(segments)} segments)')
    axes[3].set_xlabel('Gene Position')
    axes[3].set_ylabel('Log2 Ratio (Malignant/Normal)')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/chr{chr_name}_individual_cnv_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()