import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def create_reference_profile(adata, normal_indices, gene_positions, chromosome, window_size=100):
    """
    Create smoothed reference expression profile from normal cells for a specific chromosome
    """
    print(f"Creating reference profile for chromosome {chromosome}...")
    
    # Get genes for this chromosome
    chr_genes = gene_positions[gene_positions['chromosome'] == chromosome]['gene_symbol'].tolist()
    chr_genes_in_data = [g for g in chr_genes if g in adata.var_names]
    
    if len(chr_genes_in_data) < 10:
        return None
        
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
    
    reference_profile = {
        'genes': chr_positions['gene_symbol'].tolist(),
        'positions': chr_positions['start'].values,
        'gene_indices': gene_indices,
        'reference_profile': normal_smoothed,
        'raw_profile': normal_mean,
        'chr_positions': chr_positions
    }
    
    return reference_profile

def calculate_cnv_ratios_chr7(adata, reference_profile, normal_indices, malignant_indices, window_size=100):
    """
    Calculate CNV ratios for chromosome 7
    """
    print("Calculating CNV ratios for Chr7...")
    
    gene_indices = reference_profile['gene_indices']
    reference = reference_profile['reference_profile']
    
    # Extract expression for malignant cells
    if hasattr(adata.X, 'toarray'):
        malignant_expr = adata.X[malignant_indices, :][:, gene_indices].toarray()
    else:
        malignant_expr = adata.X[malignant_indices, :][:, gene_indices]
    
    # Calculate mean expression for malignant cells
    malignant_mean = malignant_expr.mean(axis=0)
    
    # Apply smoothing to malignant mean
    sigma = min(window_size, len(malignant_mean)) / 6
    malignant_smoothed = gaussian_filter1d(malignant_mean, sigma=sigma, mode='nearest')
    
    # Calculate log2 ratio
    log2_ratio_mean = np.log2((malignant_smoothed + 1e-8) / (reference + 1e-8))
    
    cnv_data = {
        'genes': reference_profile['genes'],
        'positions': reference_profile['positions'],
        'log2_ratio_mean': log2_ratio_mean,
        'reference': reference,
        'malignant_mean': malignant_smoothed,
        'chr_positions': reference_profile['chr_positions']
    }
    
    return cnv_data

def get_aml_hotspot_regions():
    """
    Define commonly deleted regions in AML chr7q
    Returns genomic coordinates (approximate) for 7q22.1, 7q34, and 7q35-36.1
    """
    # Approximate genomic coordinates for chr7q deleted regions in AML
    hotspots = {
        '7q22.1': {
            'start': 98400001,
            'end': 104200000,
            'name': '7q22.1',
            'color': 'red',
            'alpha': 0.3
        },
        '7q34': {
            'start': 138500001,
            'end': 143400000,
            'name': '7q34',
            'color': 'orange',
            'alpha': 0.3
        },
        '7q35-36.1': {
            'start': 143400001,
            'end': 152800000,
            'name': '7q35-36.1',
            'color': 'darkred',
            'alpha': 0.3
        }
    }
    return hotspots

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

def find_hotspot_overlaps(cnv_data, hotspots, segments):
    """
    Find overlaps between detected CNV segments and AML hotspot regions
    """
    positions = cnv_data['positions']
    genes = cnv_data['genes']
    chr_positions = cnv_data['chr_positions']
    
    overlaps = {}
    
    for hotspot_name, hotspot_info in hotspots.items():
        hotspot_start = hotspot_info['start']
        hotspot_end = hotspot_info['end']
        
        # Find genes in this hotspot region
        hotspot_genes_mask = (chr_positions['start'] >= hotspot_start) & (chr_positions['start'] <= hotspot_end)
        hotspot_genes = chr_positions[hotspot_genes_mask]['gene_symbol'].tolist()
        
        # Find gene indices in our data
        hotspot_gene_indices = []
        for i, gene in enumerate(genes):
            if gene in hotspot_genes:
                hotspot_gene_indices.append(i)
        
        if len(hotspot_gene_indices) == 0:
            continue
            
        hotspot_start_idx = min(hotspot_gene_indices)
        hotspot_end_idx = max(hotspot_gene_indices)
        
        # Check overlap with segments
        overlapping_segments = []
        for seg in segments:
            seg_start, seg_end = seg['start'], seg['end']
            
            # Check if segment overlaps with hotspot region
            if not (seg_end <= hotspot_start_idx or seg_start >= hotspot_end_idx):
                overlap_start = max(seg_start, hotspot_start_idx)
                overlap_end = min(seg_end, hotspot_end_idx)
                overlap_length = overlap_end - overlap_start
                overlap_pct = overlap_length / (hotspot_end_idx - hotspot_start_idx) * 100
                
                overlapping_segments.append({
                    'segment': seg,
                    'overlap_start': overlap_start,
                    'overlap_end': overlap_end,
                    'overlap_length': overlap_length,
                    'overlap_percentage': overlap_pct
                })
        
        overlaps[hotspot_name] = {
            'gene_indices': hotspot_gene_indices,
            'start_idx': hotspot_start_idx,
            'end_idx': hotspot_end_idx,
            'genes': hotspot_genes,
            'overlapping_segments': overlapping_segments
        }
    
    return overlaps

# Load data
print("Loading data for Chr7q AML hotspots analysis...")
adata_filtered = sc.read_h5ad('cnv_analysis/filtered_adata.h5ad')
adata_original = sc.read_h5ad('/Users/axuan/Desktop/Thesis/AML328_purified.h5ad')
gene_positions = pd.read_csv('cnv_analysis/gene_positions.csv')

print(f"Dataset: {adata_filtered.shape[0]} cells, {adata_filtered.shape[1]} genes")

# Apply normalization
adata_normalized = adata_filtered.copy()
sc.pp.normalize_total(adata_normalized, target_sum=1e4)

cell_types = adata_original.obs['CellType']
print(f"Cell type distribution: {cell_types.value_counts().to_dict()}")

# Define cell populations
normal_mask = cell_types.isin(['HSC', 'Prog'])
normal_indices = np.where(normal_mask)[0]
malignant_mask = cell_types.isin(['HSC-like', 'Prog-like'])
malignant_indices = np.where(malignant_mask)[0]

print(f"Normal cells (HSC + Prog): {len(normal_indices)}")
print(f"Malignant cells (HSC-like + Prog-like): {len(malignant_indices)}")

# Focus on chromosome 7
chromosome = '7'
reference_profile = create_reference_profile(adata_normalized, normal_indices, gene_positions, chromosome)

if reference_profile is None:
    raise ValueError("Could not create reference profile for chromosome 7")

print(f"Chr7: {len(reference_profile['genes'])} genes")

# Calculate CNV ratios
cnv_data = calculate_cnv_ratios_chr7(adata_normalized, reference_profile, normal_indices, malignant_indices)

# Get AML hotspot regions
hotspots = get_aml_hotspot_regions()

# Apply segmentation
log2_ratio = cnv_data['log2_ratio_mean']
segments = segment_cnv_simple(log2_ratio, min_segment_size=10, threshold=0.3)
print(f"Found {len(segments)} segments in Chr7")

# Find overlaps with hotspot regions
overlaps = find_hotspot_overlaps(cnv_data, hotspots, segments)

# Print overlap analysis
print("\n" + "="*80)
print("AML HOTSPOT OVERLAP ANALYSIS")
print("="*80)

for hotspot_name, overlap_info in overlaps.items():
    print(f"\n{hotspot_name}:")
    print(f"  Genes in region: {len(overlap_info['genes'])}")
    print(f"  Gene indices: {overlap_info['start_idx']}-{overlap_info['end_idx']}")
    
    if overlap_info['overlapping_segments']:
        print(f"  Overlapping segments: {len(overlap_info['overlapping_segments'])}")
        for i, seg_overlap in enumerate(overlap_info['overlapping_segments']):
            seg = seg_overlap['segment']
            print(f"    Segment {i+1}: {seg['state']} (mean log2: {seg['mean_log2']:.3f})")
            print(f"               Overlap: {seg_overlap['overlap_percentage']:.1f}% of hotspot region")
    else:
        print(f"  No overlapping segments detected")

# Create comprehensive visualization
print("\nCreating visualization...")
fig, axes = plt.subplots(5, 1, figsize=(20, 18))
fig.suptitle('Chr7q CNV Analysis with AML Deletion Hotspots', fontsize=18, fontweight='bold', y = 0.99)

x_positions = np.arange(len(log2_ratio))
positions = cnv_data['positions']

# 1. Chromosome ideogram with hotspots
axes[0].set_xlim(0, len(x_positions))
axes[0].set_ylim(-0.5, 1.5)

# Add hotspot regions as colored bands
for hotspot_name, overlap_info in overlaps.items():
    if len(overlap_info['gene_indices']) > 0:
        start_idx = overlap_info['start_idx']
        end_idx = overlap_info['end_idx']
        hotspot_info = hotspots[hotspot_name]
        
        axes[0].barh(0.5, end_idx - start_idx, left=start_idx, height=0.8,
                    color=hotspot_info['color'], alpha=hotspot_info['alpha'],
                    label=f"{hotspot_name}")

axes[0].set_title('Chr7q AML Deletion Hotspots', fontsize=14)
axes[0].set_ylabel('Hotspot Regions')
axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0].set_yticks([])

# 2. Normal reference profile
axes[1].plot(x_positions, cnv_data['reference'], 'g-', alpha=0.8, linewidth=2, label='Normal Reference')
axes[1].set_title('Chr7 - Normal Reference Expression', fontsize=14)
axes[1].set_ylabel('Normalized Expression')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. Malignant expression profile
axes[2].plot(x_positions, cnv_data['malignant_mean'], 'r-', alpha=0.8, linewidth=2, label='Malignant Mean')
axes[2].set_title('Chr7 - Malignant Mean Expression', fontsize=14)
axes[2].set_ylabel('Normalized Expression')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# 4. Direct comparison with hotspot highlighting
axes[3].plot(x_positions, cnv_data['reference'], 'g-', alpha=0.6, linewidth=2, label='Normal Reference')
axes[3].plot(x_positions, cnv_data['malignant_mean'], 'r-', alpha=0.8, linewidth=2, label='Malignant Mean')

# Add hotspot region highlighting
for hotspot_name, overlap_info in overlaps.items():
    if len(overlap_info['gene_indices']) > 0:
        start_idx = overlap_info['start_idx']
        end_idx = overlap_info['end_idx']
        hotspot_info = hotspots[hotspot_name]
        
        axes[3].axvspan(start_idx, end_idx, color=hotspot_info['color'], 
                       alpha=0.2, label=f"{hotspot_name} region")

axes[3].set_title('Chr7 - Expression Comparison with AML Hotspots', fontsize=14)
axes[3].set_ylabel('Normalized Expression')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

# 5. Log2 ratio with segmentation and hotspots
axes[4].plot(x_positions, log2_ratio, 'purple', alpha=0.7, linewidth=2, label='Log2 Ratio')
axes[4].axhline(y=0.3, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Gain threshold')
axes[4].axhline(y=-0.3, color='blue', linestyle='--', alpha=0.7, linewidth=2, label='Loss threshold')
axes[4].axhline(y=0, color='black', linestyle='-', alpha=0.5)

# Highlight CNV segments
for seg in segments:
    start, end = seg['start'], seg['end']
    state = seg['state']
    if state == 'gain':
        color = 'red'
        alpha = 0.9
        linewidth = 5
    elif state == 'loss':
        color = 'blue'
        alpha = 0.9
        linewidth = 5
    else:
        continue
    
    axes[4].plot(x_positions[start:end], log2_ratio[start:end], 
                color=color, alpha=alpha, linewidth=linewidth, zorder=5)

# Add hotspot region highlighting
for hotspot_name, overlap_info in overlaps.items():
    if len(overlap_info['gene_indices']) > 0:
        start_idx = overlap_info['start_idx']
        end_idx = overlap_info['end_idx']
        hotspot_info = hotspots[hotspot_name]
        
        axes[4].axvspan(start_idx, end_idx, color=hotspot_info['color'], 
                       alpha=0.15, label=f"{hotspot_name}")

# Add segment boundaries
for seg in segments:
    axes[4].axvline(seg['start'], color='black', linestyle=':', alpha=0.7, linewidth=1)

axes[4].set_title(f'Chr7 - CNV Segmentation with AML Hotspots ({len(segments)} segments)', fontsize=14)
axes[4].set_xlabel('Gene Position (5\' to 3\' along chromosome)', fontsize=12)
axes[4].set_ylabel('Log2 Ratio (Malignant/Normal)', fontsize=12)
axes[4].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[4].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cnv_analysis/chr7q_AML_hotspots_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Create summary report
print("\n" + "="*80)
print("SUMMARY REPORT: Chr7q Deletions in AML Sample")
print("="*80)

deletion_segments = [seg for seg in segments if seg['state'] == 'loss']
print(f"Total deletion segments detected: {len(deletion_segments)}")

if deletion_segments:
    print("\nDeletion segments:")
    for i, seg in enumerate(deletion_segments):
        start_pos = positions[seg['start']] if seg['start'] < len(positions) else 'N/A'
        end_pos = positions[seg['end']-1] if seg['end']-1 < len(positions) else 'N/A'
        print(f"  Segment {i+1}: Genes {seg['start']}-{seg['end']} (positions {start_pos}-{end_pos})")
        print(f"             Mean log2 ratio: {seg['mean_log2']:.3f}")

print(f"\nAML hotspot overlap summary:")
for hotspot_name, overlap_info in overlaps.items():
    overlapping_deletions = [seg_overlap for seg_overlap in overlap_info['overlapping_segments'] 
                           if seg_overlap['segment']['state'] == 'loss']
    if overlapping_deletions:
        print(f"  {hotspot_name}: DELETION DETECTED - {len(overlapping_deletions)} overlapping deletion segment(s)")
        for seg_overlap in overlapping_deletions:
            print(f"    - {seg_overlap['overlap_percentage']:.1f}% of hotspot region affected")
    else:
        print(f"  {hotspot_name}: No deletions detected in this region")

print("\nAnalysis complete!")