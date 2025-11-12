import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import warnings
import os
warnings.filterwarnings('ignore')

output_dir = 'cnv_analysis/5_raw_count'

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

def calculate_cnv_ratios_from_means(adata, reference_profiles, malignant_indices, window_size=100):
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

def visualize_chromosome(chr_name, chr_data, segments, output_dir):
    """
    Create visualization for a single chromosome
    """
    log2_ratio = chr_data['log2_ratio_mean']
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(20, 12))
    fig.suptitle(f'Chr{chr_name} CNV Analysis (raw count) - {len(segments)} segments', fontsize=16, fontweight='bold')
    
    x_positions = np.arange(len(log2_ratio))
    
    # 1. Log2 ratio with segment boundaries
    axes[0].plot(x_positions, log2_ratio, 'purple', alpha=0.7, linewidth=2, label='Log2 Ratio')
    axes[0].axhline(y=0.3, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Gain threshold')
    axes[0].axhline(y=-0.3, color='blue', linestyle='--', alpha=0.7, linewidth=2, label='Loss threshold')
    axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Mark segment boundaries and color regions
    for i, seg in enumerate(segments):
        start, end = seg['start'], seg['end']
        state = seg['state']
        
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
        axes[0].axvline(start, color='black', linestyle=':', alpha=0.8, linewidth=1)
        
        # Add segment number labels (only for non-neutral segments)
        if state != 'neutral':
            mid_point = (start + end) / 2
            axes[0].text(mid_point, max(log2_ratio) * 0.9, f'S{i+1}', 
                        ha='center', va='center', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    axes[0].set_title(f'Chr{chr_name} Log2 Ratio with Segment Boundaries ({len(log2_ratio)} genes)')
    axes[0].set_ylabel('Log2 Ratio')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Histogram of log2 ratios
    axes[1].hist(log2_ratio, bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1].axvline(np.mean(log2_ratio), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {np.mean(log2_ratio):.3f}')
    axes[1].axvline(0.3, color='red', linestyle=':', alpha=0.7, label='Gain threshold')
    axes[1].axvline(-0.3, color='blue', linestyle=':', alpha=0.7, label='Loss threshold')
    axes[1].set_title(f'Chr{chr_name} Log2 Ratio Distribution')
    axes[1].set_xlabel('Log2 Ratio')
    axes[1].set_ylabel('Number of Genes')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Segment summary
    gain_segments = [s for s in segments if s['state'] == 'gain']
    loss_segments = [s for s in segments if s['state'] == 'loss']
    
    segment_info = []
    for i, seg in enumerate(segments):
        if seg['state'] != 'neutral':  # Only show gain/loss segments
            segment_info.append([
                f"S{i+1}",
                f"{seg['start']}-{seg['end']}",
                seg['state'],
                f"{seg['mean_log2']:.3f}",
                f"{seg['end'] - seg['start']}"
            ])
    
    if segment_info:
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
        
        axes[2].set_title(f'Chr{chr_name} Significant CNV Segments (Gains: {len(gain_segments)}, Losses: {len(loss_segments)})')
    else:
        axes[2].text(0.5, 0.5, f'Chr{chr_name}: No significant CNV segments detected', 
                    ha='center', va='center', fontsize=14, transform=axes[2].transAxes)
        axes[2].set_title(f'Chr{chr_name} - No Significant CNV Segments')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/chr{chr_name}_cnv_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return gain_segments, loss_segments

def main():
    print("Loading data for all-chromosome CNV analysis...")
    
    # Load data
    adata_filtered = sc.read_h5ad('cnv_analysis/filtered_adata.h5ad')
    adata_original = sc.read_h5ad('/Users/axuan/Desktop/Thesis/AML328_purified.h5ad')
    gene_positions = pd.read_csv('cnv_analysis/gene_positions.csv')
    
    # Define cell groups
    cell_types = adata_original.obs['CellType']
    normal_mask = cell_types.isin(['HSC', 'Prog'])
    malignant_mask = cell_types.isin(['HSC-like', 'Prog-like'])
    normal_indices = np.where(normal_mask)[0]
    malignant_indices = np.where(malignant_mask)[0]
    
    # Define chromosomes (all except Y)
    chromosomes = [str(i) for i in range(1, 23)] + ['X']
    
    # Create output directory
    #output_dir = 'cnv_analysis/all_chromosomes'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each chromosome
    all_segments_summary = []
    all_gain_genes = []
    all_loss_genes = []
    
    print(f"\nProcessing {len(chromosomes)} chromosomes...")
    print("="*80)
    
    for chr_name in chromosomes:
        print(f"\nProcessing Chr{chr_name}...")
        
        # Create reference profiles for this chromosome
        reference_profiles = create_reference_profile(adata_filtered, normal_indices,
                                                    gene_positions, [chr_name])
        
        # Calculate CNV ratios
        cnv_data = calculate_cnv_ratios_from_means(adata_filtered, reference_profiles,
                                                 malignant_indices)
        
        # Get chromosome data
        chr_data = cnv_data[chr_name]
        log2_ratio = chr_data['log2_ratio_mean']
        segments = segment_cnv_simple(log2_ratio, min_segment_size=10, threshold=0.3)
        
        # Generate visualization
        gain_segments, loss_segments = visualize_chromosome(chr_name, chr_data, segments, output_dir)
        
        print(f"   Chr{chr_name}: {len(chr_data['genes'])} genes, {len(segments)} segments")
        print(f"   Gains: {len(gain_segments)}, Losses: {len(loss_segments)}")
        
        # Collect segment information
        for i, seg in enumerate(segments):
            if seg['state'] != 'neutral':
                start_idx, end_idx = seg['start'], seg['end']
                segment_genes = chr_data['genes'][start_idx:end_idx]
                segment_positions = chr_data['positions'][start_idx:end_idx]
                segment_log2_ratios = log2_ratio[start_idx:end_idx]
                
                # Add to summary
                all_segments_summary.append({
                    'chromosome': chr_name,
                    'segment_number': i + 1,
                    'start_gene_idx': start_idx,
                    'end_gene_idx': end_idx,
                    'state': seg['state'],
                    'mean_log2': seg['mean_log2'],
                    'num_genes': len(segment_genes),
                    'genomic_start': segment_positions[0] if len(segment_positions) > 0 else None,
                    'genomic_end': segment_positions[-1] if len(segment_positions) > 0 else None,
                    'genomic_span_mb': (segment_positions[-1] - segment_positions[0]) / 1000000 if len(segment_positions) > 0 else 0
                })
                
                # Collect genes for detailed analysis
                for j, (gene, pos, ratio) in enumerate(zip(segment_genes, segment_positions, segment_log2_ratios)):
                    gene_info = {
                        'chromosome': chr_name,
                        'segment_number': i + 1,
                        'gene_symbol': gene,
                        'genomic_position': pos,
                        'log2_ratio': ratio,
                        'segment_state': seg['state'],
                        'segment_mean_log2': seg['mean_log2']
                    }
                    
                    if seg['state'] == 'gain':
                        all_gain_genes.append(gene_info)
                    else:
                        all_loss_genes.append(gene_info)
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE!")
    print(f"{'='*80}")

    # Save comprehensive results
    segments_df = pd.DataFrame(all_segments_summary)
    gain_genes_df = pd.DataFrame(all_gain_genes)
    loss_genes_df = pd.DataFrame(all_loss_genes)
    
    if not segments_df.empty:
        segments_df.to_csv(f'{output_dir}/all_cnv_segments_summary.csv', index=False)
        print(f" Segment summary saved: {output_dir}/all_cnv_segments_summary.csv")

    if not gain_genes_df.empty:
        gain_genes_df.to_csv(f'{output_dir}/all_gain_genes.csv', index=False)
        print(f" All gain genes saved: {output_dir}/all_gain_genes.csv")

    if not loss_genes_df.empty:
        loss_genes_df.to_csv(f'{output_dir}/all_loss_genes.csv', index=False)
        print(f" All loss genes saved: {output_dir}/all_loss_genes.csv")


if __name__ == "__main__":
    main()