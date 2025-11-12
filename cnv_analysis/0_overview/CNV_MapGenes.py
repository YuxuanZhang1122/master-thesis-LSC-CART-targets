import pandas as pd
import scanpy as sc
import requests
import io


def get_gene_positions_mygene(gene_symbols, batch_size=1000):
    """
    Use mygene.info to get gene positions for HGNC symbols
    """
    all_results = []

    for i in range(0, len(gene_symbols), batch_size):
        batch = gene_symbols[i:i + batch_size]

        url = "http://mygene.info/v3/query"
        params = {
            'q': ','.join(batch),
            'scopes': 'symbol',
            'fields': 'symbol,ensembl.gene,genomic_pos',
            'species': 'human',
            'size': batch_size
        }

        response = requests.post(url, data=params)

        if response.ok:
            data = response.json()

            batch_results = []
            for item in data:
                if 'genomic_pos' in item and item.get('symbol'):
                    genomic_pos = item['genomic_pos']
                    if isinstance(genomic_pos, list):
                        genomic_pos = genomic_pos[0]  # Take first position

                    ensembl_info = item.get('ensembl', {})
                    if isinstance(ensembl_info, list):
                        ensembl_id = ensembl_info[0].get('gene', '') if ensembl_info else ''
                    else:
                        ensembl_id = ensembl_info.get('gene', '')
                        
                    batch_results.append({
                        'gene_symbol': item['symbol'],
                        'ensembl_id': ensembl_id,
                        'chromosome': str(genomic_pos.get('chr', '')),
                        'start': genomic_pos.get('start', 0),
                        'end': genomic_pos.get('end', 0),
                        'strand': genomic_pos.get('strand', 0)
                    })

            if batch_results:
                all_results.append(pd.DataFrame(batch_results))
                print(f"Processed batch {i // batch_size + 1}: {len(batch_results)} genes found")

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()


def prepare_anndata_for_cnv(adata, output_dir=None,
                                 use_raw=False, min_genes_per_chr=10):
    """
    Prepare AnnData object for CNV analysis

    Parameters:
    -----------
    adata : AnnData
        Filtered (only matching genes kept), reordered (sorted according to chromosome position) anndata
    output_dir : str
        Directory to save output files
    use_raw : bool
        Whether to use adata.raw.X instead of adata.X
    min_genes_per_chr : int
        Minimum genes required per chromosome to include it
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Get gene names from var_names (should be HGNC symbols)
    if use_raw and adata.raw is not None:
        gene_symbols = list(adata.raw.var_names)
        print(f"Using raw data with {len(gene_symbols)} genes")
    else:
        gene_symbols = list(adata.var_names)
        print(f"Using processed data with {len(gene_symbols)} genes")

    print(f"Example genes: {gene_symbols[:5]}")

    # Get chromosomal positions
    print("Fetching gene positions...")
    gene_positions = get_gene_positions_mygene(gene_symbols)

    if gene_positions.empty:
        raise ValueError("No gene positions found. Check gene symbol format.")

    # Filter for standard chromosomes
    standard_chr = [str(i) for i in range(1, 23)] + ['X', 'Y']
    gene_positions = gene_positions[gene_positions['chromosome'].isin(standard_chr)]

    # Check chromosome distribution
    chr_counts = gene_positions['chromosome'].value_counts()
    print("Genes per chromosome:")
    print(chr_counts.sort_index())

    # Filter chromosomes with too few genes
    valid_chrs = chr_counts[chr_counts >= min_genes_per_chr].index
    gene_positions = gene_positions[gene_positions['chromosome'].isin(valid_chrs)]

    # Find genes that match between AnnData and position data
    matched_genes = set(gene_symbols) & set(gene_positions['gene_symbol'])
    unmatched_genes = set(gene_symbols) - set(gene_positions['gene_symbol'])

    print(f"\nGene matching results:")
    print(f"  Total genes in data: {len(gene_symbols)}")
    print(f"  Genes with positions: {len(gene_positions)}")
    print(f"  Matched genes: {len(matched_genes)}")
    print(f"  Unmatched genes: {len(unmatched_genes)}")
    print(f"  Match rate: {len(matched_genes) / len(gene_symbols) * 100:.1f}%")

    # Create filtered AnnData
    matched_gene_list = list(matched_genes)
    if use_raw and adata.raw is not None:
        # Filter raw data
        gene_mask = adata.raw.var_names.isin(matched_gene_list)
        adata_filtered = adata[:, gene_mask].copy()
        # Copy raw data to main layers
        adata_filtered.X = adata_filtered.raw.X
        adata_filtered.var = adata_filtered.raw.var
    else:
        # Filter main data
        gene_mask = adata.var_names.isin(matched_gene_list)
        adata_filtered = adata[:, gene_mask].copy()

    # Create gene order dataframe
    filtered_positions = gene_positions[gene_positions['gene_symbol'].isin(matched_genes)]
    
    # Handle duplicate gene symbols by taking the first occurrence
    filtered_positions = filtered_positions.drop_duplicates(subset=['gene_symbol'], keep='first')
    
    filtered_positions['chr_num'] = pd.Categorical(
        filtered_positions['chromosome'],
        categories=standard_chr,
        ordered=True
    )
    gene_order = filtered_positions.sort_values(['chr_num', 'start']).reset_index(drop=True)

    # Add gene positions to AnnData.var
    gene_pos_dict = gene_order.set_index('gene_symbol').to_dict('index')

    # Reorder genes in AnnData to match chromosomal order
    ordered_genes = gene_order['gene_symbol'].tolist()
    adata_ordered = adata_filtered[:, ordered_genes].copy()

    # Add position info to var
    for gene in adata_ordered.var_names:
        if gene in gene_pos_dict:
            for key, value in gene_pos_dict[gene].items():
                adata_ordered.var.loc[gene, key] = value

    # Save filtered AnnData
    adata_file = os.path.join(output_dir, 'filtered_adata.h5ad')
    adata_ordered.write(adata_file)

    # Save gene positions for reference
    positions_file = os.path.join(output_dir, 'gene_positions.csv')
    gene_order.to_csv(positions_file, index=False)

    return adata_ordered, gene_order

# Usage
# adata = sc.read_h5ad('data.h5ad')
# adata_filtered, gene_order = prepare_anndata_for_cnv(adata)

def quick_gene_position_test(adata, n_test_genes=10):
    """
    Quick test to see how many genes will match
    """
    # Sample some genes
    test_genes = list(adata.var_names[:n_test_genes])
    print(f"Testing with genes: {test_genes}")

    # Get positions
    positions = get_gene_positions_mygene(test_genes)

    print(f"\nResults:")
    print(f"  Genes tested: {len(test_genes)}")
    print(f"  Positions found: {len(positions)}")
    print(f"  Success rate: {len(positions) / len(test_genes) * 100:.1f}%")

    if not positions.empty:
        print(f"\nExample results:")
        print(positions.head())

        # Check chromosome distribution
        print(f"\nChromosomes found: {sorted(positions['chromosome'].unique())}")

    return positions
