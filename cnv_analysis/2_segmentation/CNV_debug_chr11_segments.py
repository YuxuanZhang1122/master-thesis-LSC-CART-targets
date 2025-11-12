import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

def create_reference_profile(adata, normal_indices, gene_positions, chromosomes, window_size=100):
    """
    Create smoothed reference expression profile from normal cells
    """
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
        sigma = min(window_size, len(normal_mean)) / 6
        normal_smoothed = gaussian_filter1d(normal_mean, sigma=sigma, mode='nearest')
        
        reference_profiles[chr_name] = {
            'genes': chr_positions['gene_symbol'].tolist(),
            'positions': chr_positions['start'].values,
            'gene_indices': gene_indices,
            'reference_profile': normal_smoothed,
            'raw_profile': normal_mean
        }
    
    return reference_profiles

def calculate_cnv_ratios_from_means(adata, reference_profiles, normal_indices, malignant_indices, window_size=100):
    """
    Calculate log2 ratios by first computing group means, then taking ratio
    """
    cnv_data = {}
    
    for chr_name, ref_data in reference_profiles.items():
        gene_indices = ref_data['gene_indices']
        reference = ref_data['reference_profile']
        
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
            'log2_ratio_mean': log2_ratio_mean,
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
print("Loading data for Chr11 segment debugging...")
adata_filtered = sc.read_h5ad('cnv_analysis/filtered_adata.h5ad')
adata_original = sc.read_h5ad('/Users/axuan/Desktop/Thesis/AML328_purified.h5ad')
gene_positions = pd.read_csv('cnv_analysis/gene_positions.csv')

# Apply normalization
adata_normalized = adata_filtered.copy()
sc.pp.normalize_total(adata_normalized, target_sum=1e4)
#sc.pp.log1p(adata_normalized)

# Define cell groups
cell_types = adata_original.obs['CellType']
normal_mask = cell_types.isin(['HSC', 'Prog'])
malignant_mask = cell_types.isin(['HSC-like', 'Prog-like'])
normal_indices = np.where(normal_mask)[0]
malignant_indices = np.where(malignant_mask)[0]

# Focus on Chr11
chromosomes = ['11']
reference_profiles = create_reference_profile(adata_normalized, normal_indices, gene_positions, chromosomes)
cnv_data = calculate_cnv_ratios_from_means(adata_normalized, reference_profiles, normal_indices, malignant_indices)

# Get Chr11 data
chr_data = cnv_data['11']
log2_ratio = chr_data['log2_ratio_mean']
segments = segment_cnv_simple(log2_ratio, min_segment_size=10, threshold=0.3)

print(f"\nChr11 Analysis:")
print(f"Total genes: {len(log2_ratio)}")
print(f"Found {len(segments)} segments")

# Print detailed segment information
print("\nDetailed Segment Information:")
print("="*80)
for i, seg in enumerate(segments):
    start, end = seg['start'], seg['end']
    state = seg['state']
    mean_log2 = seg['mean_log2']
    length = end - start
    
    print(f"Segment {i+1:2d}: Positions {start:3d}-{end:3d} ({length:2d} genes)")
    print(f"            State: {state:7s}, Mean Log2: {mean_log2:6.3f}")
    
    # Show some example values from this segment
    segment_values = log2_ratio[start:end]
    print(f"            Value range: [{np.min(segment_values):.3f}, {np.max(segment_values):.3f}]")
    print(f"            Sample values: {segment_values[:3]} ...")
    print("-" * 60)

# Extract genes for Segment 4 specifically
if len(segments) >= 4:
    segment_4 = segments[3]  # 0-indexed, so segment 4 is index 3
    start_idx = segment_4['start']
    end_idx = segment_4['end']
    
    print(f"\n🧬 SEGMENT 4 GENES ({segment_4['state'].upper()}):")
    print("="*80)
    print(f"Genomic positions: {start_idx}-{end_idx} ({end_idx - start_idx} genes)")
    print(f"Mean Log2 ratio: {segment_4['mean_log2']:.3f}")
    print(f"State: {segment_4['state']}")
    
    # Get the genes in this segment
    segment_4_genes = chr_data['genes'][start_idx:end_idx]
    segment_4_positions = chr_data['positions'][start_idx:end_idx]
    segment_4_log2_ratios = log2_ratio[start_idx:end_idx]
    
    print(f"\nGenes in Segment 4:")
    print("-" * 60)
    for i, (gene, pos, ratio) in enumerate(zip(segment_4_genes, segment_4_positions, segment_4_log2_ratios)):
        print(f"{i+1:3d}. {gene:15s} | Position: {pos:>10,} | Log2 ratio: {ratio:6.3f}")
    
    # Save segment 4 genes to a CSV file
    segment_4_df = pd.DataFrame({
        'gene_symbol': segment_4_genes,
        'genomic_position': segment_4_positions,
        'log2_ratio': segment_4_log2_ratios,
        'segment': 4,
        'segment_state': segment_4['state'],
        'segment_mean_log2': segment_4['mean_log2']
    })
    
    segment_4_df.to_csv('cnv_analysis/chr11_segment4_genes.csv', index=False)
    print(f"\n✅ Segment 4 genes saved to: cnv_analysis/chr11_segment4_genes.csv")
    
    # Print summary statistics
    print(f"\nSegment 4 Summary:")
    print(f"Number of genes: {len(segment_4_genes)}")
    print(f"Genomic span: {segment_4_positions[0]:,} - {segment_4_positions[-1]:,}")
    print(f"Log2 ratio range: [{np.min(segment_4_log2_ratios):.3f}, {np.max(segment_4_log2_ratios):.3f}]")
    
else:
    print(f"\n❌ Segment 4 not found! Only {len(segments)} segments detected.")

print(f"\nSummary:")
print(f"Total segments: {len(segments)}")
print(f"Gain segments: {sum(1 for s in segments if s['state'] == 'gain')}")
print(f"Loss segments: {sum(1 for s in segments if s['state'] == 'loss')}")
print(f"Neutral segments: {sum(1 for s in segments if s['state'] == 'neutral')}")