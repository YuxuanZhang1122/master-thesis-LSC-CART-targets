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
print("Loading data for Chr6 segment debugging...")
adata_filtered = sc.read_h5ad('cnv_analysis/filtered_adata.h5ad')
adata_original = sc.read_h5ad('/Users/axuan/Desktop/Thesis/AML328_purified.h5ad')
gene_positions = pd.read_csv('cnv_analysis/gene_positions.csv')

# Apply normalization
adata_normalized = adata_filtered.copy()
sc.pp.normalize_total(adata_normalized, target_sum=1e4)
sc.pp.log1p(adata_normalized)

# Define cell groups
cell_types = adata_original.obs['CellType']
normal_mask = cell_types.isin(['HSC', 'Prog'])
malignant_mask = cell_types.isin(['HSC-like', 'Prog-like'])
normal_indices = np.where(normal_mask)[0]
malignant_indices = np.where(malignant_mask)[0]

# Focus on Chr6
chromosomes = ['6']
reference_profiles = create_reference_profile(adata_normalized, normal_indices, gene_positions, chromosomes)
cnv_data = calculate_cnv_ratios_from_means(adata_normalized, reference_profiles, normal_indices, malignant_indices)

# Get Chr6 data
chr_data = cnv_data['6']
log2_ratio = chr_data['log2_ratio_mean']
segments = segment_cnv_simple(log2_ratio, min_segment_size=10, threshold=0.3)

print(f"\nChr6 Analysis:")
print(f"Total genes: {len(log2_ratio)}")
print(f"Found {len(segments)} segments")

# Create detailed visualization
fig, axes = plt.subplots(3, 1, figsize=(20, 12))
fig.suptitle('Chr6 Segment Analysis - Detailed View', fontsize=16)

x_positions = np.arange(len(log2_ratio))

# 1. Log2 ratio with all segment boundaries marked
axes[0].plot(x_positions, log2_ratio, 'purple', alpha=0.7, linewidth=2, label='Log2 Ratio')
axes[0].axhline(y=0.3, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Gain threshold')
axes[0].axhline(y=-0.3, color='blue', linestyle='--', alpha=0.7, linewidth=2, label='Loss threshold')
axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.5)

# Mark ALL segment boundaries
for i, seg in enumerate(segments):
    start, end = seg['start'], seg['end']
    state = seg['state']
    mean_log2 = seg['mean_log2']
    
    # Color code by segment type
    if state == 'gain':
        color = 'red'
        alpha = 0.3
    elif state == 'loss':
        color = 'blue'
        alpha = 0.3
    else:
        color = 'gray'
        alpha = 0.2
    
    # Fill the segment region
    axes[0].axvspan(start, end, alpha=alpha, color=color)
    
    # Add boundary lines
    axes[0].axvline(start, color='black', linestyle=':', alpha=0.8, linewidth=2)
    
    # Add segment number labels
    mid_point = (start + end) / 2
    axes[0].text(mid_point, max(log2_ratio) * 0.9, f'S{i+1}', 
                ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

axes[0].set_title('Chr6 Log2 Ratio with ALL Segment Boundaries')
axes[0].set_ylabel('Log2 Ratio')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Zoomed view of threshold crossings
axes[1].plot(x_positions, log2_ratio, 'purple', alpha=0.7, linewidth=2)
axes[1].axhline(y=0.3, color='red', linestyle='--', alpha=0.7, linewidth=2)
axes[1].axhline(y=-0.3, color='blue', linestyle='--', alpha=0.7, linewidth=2)
axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)

# Highlight only gain/loss segments
for i, seg in enumerate(segments):
    if seg['state'] in ['gain', 'loss']:
        start, end = seg['start'], seg['end']
        state = seg['state']
        color = 'red' if state == 'gain' else 'blue'
        axes[1].plot(x_positions[start:end], log2_ratio[start:end], 
                    color=color, alpha=0.8, linewidth=4, zorder=5)

axes[1].set_ylim(-0.8, 0.8)  # Focus on threshold region
axes[1].set_title('Chr6 - Zoomed View of Gain/Loss Regions')
axes[1].set_ylabel('Log2 Ratio')
axes[1].grid(True, alpha=0.3)

# 3. Segment summary table as plot
segment_info = []
for i, seg in enumerate(segments):
    segment_info.append([
        f"S{i+1}",
        f"{seg['start']}-{seg['end']}",
        seg['state'],
        f"{seg['mean_log2']:.3f}",
        f"{seg['end'] - seg['start']}"
    ])

# Create table
axes[2].axis('tight')
axes[2].axis('off')
table_data = [['Segment', 'Positions', 'State', 'Mean Log2', 'Length']] + segment_info
table = axes[2].table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# Color code table rows
for i in range(1, len(table_data)):
    state = table_data[i][2]
    if state == 'gain':
        color = 'lightcoral'
    elif state == 'loss':
        color = 'lightblue'
    else:
        color = 'lightgray'
    
    for j in range(len(table_data[i])):
        table[(i, j)].set_facecolor(color)

axes[2].set_title('Chr6 Segment Details')

plt.tight_layout()
plt.savefig('cnv_analysis/chr6_segment_debugging.png', dpi=300, bbox_inches='tight')
plt.show()

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

print(f"\nSummary:")
print(f"Total segments: {len(segments)}")
print(f"Gain segments: {sum(1 for s in segments if s['state'] == 'gain')}")
print(f"Loss segments: {sum(1 for s in segments if s['state'] == 'loss')}")
print(f"Neutral segments: {sum(1 for s in segments if s['state'] == 'neutral')}")

print(f"\n✅ Check the visualization to see why some segments might not be visually obvious!")
print(f"✅ Small neutral segments between gains might appear as single regions")