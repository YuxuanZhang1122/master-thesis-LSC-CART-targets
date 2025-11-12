import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import seaborn as sns
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
            'raw_profile': normal_mean,
            'chr_positions_df': chr_positions
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
            'malignant_mean': malignant_smoothed,
            'chr_positions_df': ref_data['chr_positions_df']
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
print("Loading data for Chr6 gain genes analysis...")
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
genes = chr_data['genes']
positions = chr_data['positions']
chr_positions_df = chr_data['chr_positions_df']
segments = segment_cnv_simple(log2_ratio, min_segment_size=10, threshold=0.3)

print(f"\nChr6 Analysis:")
print(f"Total genes: {len(log2_ratio)}")
print(f"Found {len(segments)} segments")

# Extract gain segments
gain_segments = [seg for seg in segments if seg['state'] == 'gain']
print(f"Gain segments: {len(gain_segments)}")

# Analyze each gain segment
gain_regions_data = []
for i, seg in enumerate(gain_segments):
    start, end = seg['start'], seg['end']
    mean_log2 = seg['mean_log2']
    
    # Get genes in this segment
    segment_genes = genes[start:end]
    segment_positions = positions[start:end]
    segment_log2_ratios = log2_ratio[start:end]
    
    # Get genomic coordinates
    segment_chr_data = chr_positions_df.iloc[start:end].copy()
    
    # Find genomic span
    genomic_start = segment_chr_data['start'].min()
    genomic_end = segment_chr_data['end'].max()
    genomic_span_mb = (genomic_end - genomic_start) / 1e6
    
    gain_info = {
        'segment_id': i + 1,
        'array_start': start,
        'array_end': end,
        'genomic_start': genomic_start,
        'genomic_end': genomic_end,
        'genomic_span_mb': genomic_span_mb,
        'n_genes': len(segment_genes),
        'mean_log2': mean_log2,
        'fold_change': 2**mean_log2,
        'genes': segment_genes,
        'log2_ratios': segment_log2_ratios
    }
    
    gain_regions_data.append(gain_info)
    
    print(f"\nGain Segment {i+1}:")
    print(f"  Array positions: {start}-{end} ({len(segment_genes)} genes)")
    print(f"  Genomic span: {genomic_start:,} - {genomic_end:,} ({genomic_span_mb:.1f} Mb)")
    print(f"  Mean log2 ratio: {mean_log2:.3f} ({2**mean_log2:.2f}× fold change)")
    print(f"  Top genes by log2 ratio:")
    
    # Get top 10 genes by individual log2 ratio
    gene_log2_pairs = list(zip(segment_genes, segment_log2_ratios))
    gene_log2_pairs.sort(key=lambda x: x[1], reverse=True)
    
    for j, (gene, ratio) in enumerate(gene_log2_pairs[:10]):
        print(f"    {j+1:2d}. {gene}: {ratio:.3f} ({2**ratio:.2f}×)")

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 2, height_ratios=[2, 1, 1, 1], width_ratios=[3, 1])

# Main plot - Chr6 with gain regions highlighted
ax_main = fig.add_subplot(gs[0, :])
x_positions = np.arange(len(log2_ratio))

ax_main.plot(x_positions, log2_ratio, 'purple', alpha=0.7, linewidth=2, label='Log2 Ratio')
ax_main.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Gain threshold')
ax_main.axhline(y=0, color='black', linestyle='-', alpha=0.5)

# Highlight gain regions
colors = ['red', 'orange', 'darkred', 'crimson', 'maroon']
for i, gain_info in enumerate(gain_regions_data):
    start = gain_info['array_start']
    end = gain_info['array_end']
    color = colors[i % len(colors)]
    
    ax_main.plot(x_positions[start:end], log2_ratio[start:end], 
                color=color, alpha=0.8, linewidth=4, zorder=5,
                label=f'Gain {i+1} ({gain_info["genomic_span_mb"]:.1f}Mb, {gain_info["n_genes"]} genes)')
    
    # Add segment labels
    mid_point = (start + end) / 2
    ax_main.text(mid_point, max(log2_ratio) * 0.9, f'G{i+1}', 
                ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))

ax_main.set_title('Chr6 Gain Regions - Complete Overview', fontsize=16)
ax_main.set_xlabel('Gene Position (Array Index)')
ax_main.set_ylabel('Log2 Ratio (Malignant/Normal)')
ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax_main.grid(True, alpha=0.3)

# Individual gain region plots
for i, gain_info in enumerate(gain_regions_data[:3]):  # Show first 3 gain regions
    ax = fig.add_subplot(gs[i+1, 0])
    
    start = gain_info['array_start']
    end = gain_info['array_end']
    segment_genes = gain_info['genes']
    segment_ratios = gain_info['log2_ratios']
    
    x_seg = np.arange(len(segment_ratios))
    bars = ax.bar(x_seg, segment_ratios, color=colors[i], alpha=0.7)
    
    ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    ax.set_title(f'Gain Region {i+1}: {gain_info["genomic_span_mb"]:.1f}Mb, {gain_info["n_genes"]} genes')
    ax.set_xlabel('Gene Index within Region')
    ax.set_ylabel('Log2 Ratio')
    ax.grid(True, alpha=0.3)
    
    # Highlight top genes
    gene_indices_sorted = np.argsort(segment_ratios)[::-1]
    for j in gene_indices_sorted[:3]:  # Top 3 genes
        bars[j].set_color('darkred')
        bars[j].set_alpha(0.9)

# Summary table
ax_table = fig.add_subplot(gs[1:, 1])
ax_table.axis('tight')
ax_table.axis('off')

table_data = [['Region', 'Span (Mb)', 'Genes', 'Mean Log2', 'Fold Change']]
for i, gain_info in enumerate(gain_regions_data):
    table_data.append([
        f"G{i+1}",
        f"{gain_info['genomic_span_mb']:.1f}",
        f"{gain_info['n_genes']}",
        f"{gain_info['mean_log2']:.3f}",
        f"{gain_info['fold_change']:.2f}×"
    ])

table = ax_table.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2)

# Color table rows
for i in range(1, len(table_data)):
    color = colors[(i-1) % len(colors)]
    for j in range(len(table_data[i])):
        table[(i, j)].set_facecolor(color)
        table[(i, j)].set_alpha(0.3)

ax_table.set_title('Gain Regions Summary', fontweight='bold')

plt.tight_layout()
plt.savefig('cnv_analysis/chr6_gain_genes_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Create gene lists for each gain region
print("\n" + "="*80)
print("DETAILED GENE LISTS FOR EACH GAIN REGION")
print("="*80)

all_gain_genes_data = []

for i, gain_info in enumerate(gain_regions_data):
    print(f"\nGAIN REGION {i+1}:")
    print(f"Genomic coordinates: Chr6:{gain_info['genomic_start']:,}-{gain_info['genomic_end']:,}")
    print(f"Size: {gain_info['genomic_span_mb']:.1f} Mb, {gain_info['n_genes']} genes")
    print(f"Mean fold change: {gain_info['fold_change']:.2f}×")
    
    # Create detailed gene table
    segment_genes = gain_info['genes']
    segment_ratios = gain_info['log2_ratios']
    segment_positions = chr_positions_df.iloc[gain_info['array_start']:gain_info['array_end']]
    
    gene_details = []
    for j, (gene, ratio) in enumerate(zip(segment_genes, segment_ratios)):
        gene_info = segment_positions.iloc[j]
        gene_details.append({
            'region': f"G{i+1}",
            'gene_symbol': gene,
            'genomic_start': gene_info['start'],
            'genomic_end': gene_info['end'],
            'log2_ratio': ratio,
            'fold_change': 2**ratio
        })
        all_gain_genes_data.extend([{
            'region': f"G{i+1}",
            'gene_symbol': gene,
            'genomic_start': gene_info['start'],
            'genomic_end': gene_info['end'],
            'log2_ratio': ratio,
            'fold_change': 2**ratio
        }])
    
    # Sort by log2 ratio and print top genes
    gene_details.sort(key=lambda x: x['log2_ratio'], reverse=True)
    print(f"\nTop 15 genes in Gain Region {i+1}:")
    print(f"{'Rank':<4} {'Gene':<15} {'Position':<20} {'Log2':<8} {'Fold':<6}")
    print("-" * 60)
    
    for j, gene_info in enumerate(gene_details[:15]):
        pos_str = f"{gene_info['genomic_start']:,}-{gene_info['genomic_end']:,}"
        print(f"{j+1:<4} {gene_info['gene_symbol']:<15} {pos_str:<20} {gene_info['log2_ratio']:<8.3f} {gene_info['fold_change']:<6.2f}×")

# Save complete gene list
all_gain_genes_df = pd.DataFrame(all_gain_genes_data)
all_gain_genes_df = all_gain_genes_df.sort_values(['region', 'log2_ratio'], ascending=[True, False])
all_gain_genes_df.to_csv('cnv_analysis/chr6_gain_genes_complete_list.csv', index=False)
