import anndata as ad
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATASETS = [
    'Henrik_DG',
    'Petti_DG',
    'Ennis_DG',
    'Ennis_MRD',
    'Ennis_REL',
    'Naldini_V03_DG',
    'Naldini_V03_MRD',
    'Naldini_V03_REL',
    'Naldini_V02_DG',
    'Naldini_V02_MRD',
    'vanGalen',
    'Guo',
    'Abbas'
]

TARGET_CELLTYPES = {
    'HSC_MPP': 'HSC MPP',
    'LMPP': 'LMPP',
    'Early_GMP': 'Early GMP'
}

UNCERTAINTY_THRESHOLD = 1
QUERY_DIR = Path('../dataset/Queries')
EMBED_DIR = Path('../outputs/embeddings')
OUTPUT_DIR = Path('../outputs/PooledLSC')

def harvest_cells(dataset_name, celltype_label, uncertainty_threshold=0.2):
    """
    Extract cells predicted as specific celltype with low uncertainty.

    Args:
        dataset_name: Name of dataset (e.g., 'Ennis_DG')
        celltype_label: Cell type label in predictions (e.g., 'HSC MPP')
        uncertainty_threshold: Maximum uncertainty to include

    Returns:
        AnnData object with filtered cells or None if no cells found
    """

    file_dataset_name = dataset_name

    query_path = QUERY_DIR / f'{file_dataset_name}.h5ad'
    embed_path = EMBED_DIR / f'{file_dataset_name}_embeddings.h5ad'

    if not query_path.exists() or not embed_path.exists():
        logger.warning(f"Skipping {dataset_name}: missing files")
        return None

    # Load embedding predictions
    embed = ad.read_h5ad(embed_path)

    # Filter by uncertainty and cell type
    mask = (
        (embed.obs['uncertainty_CellType_Broad'] < uncertainty_threshold) &
        (embed.obs['predicted_CellType_Broad'] == celltype_label)
    )

    n_cells = mask.sum()
    if n_cells == 0:
        logger.info(f"  {dataset_name}: 0 cells")
        return None

    # Get cell IDs that passed filter
    selected_cells = embed.obs_names[mask]

    # Load full query data and subset
    query = ad.read_h5ad(query_path)

    query_subset = query[selected_cells].copy()

    # Extract study name and timepoint from dataset name
    study_name = dataset_name.split('_')[0]

    # Add Study column
    query_subset.obs['Study'] = study_name

    # Add time_point column
    if dataset_name == 'vanGalen':
        # For vanGalen, use existing time_point column
        if 'time_point' in query.obs.columns:
            timepoint_vals = query_subset.obs['time_point'].copy()
            # Map D0 -> DG, exclude HC, everything else -> non-DG
            query_subset.obs['time_point'] = timepoint_vals.apply(
                lambda x: 'DG' if x == 'D0' else ('non-DG' if x != 'HC' else x)
            )
            # Filter out HC samples
            query_subset = query_subset[query_subset.obs['time_point'] != 'HC'].copy()
    else:
        # Extract timepoint from dataset suffix
        parts = dataset_name.split('_')
        timepoint = parts[-1] if len(parts) > 1 else 'DG'
        query_subset.obs['time_point'] = timepoint

    # Handle donor/patient ID - prefer Donor, fallback to patient_id
    donor_col = None
    if 'Donor' in query.obs.columns:
        donor_col = 'Donor'
    elif 'patient_id' in query.obs.columns:
        donor_col = 'patient_id'

    if donor_col:
        # Add study prefix and save as Donor
        query_subset.obs['Donor'] = study_name + '_' + query_subset.obs[donor_col].astype(str)

    # Add prediction metadata
    query_subset.obs['predicted_CellType_Broad'] = embed.obs.loc[selected_cells, 'predicted_CellType_Broad']
    query_subset.obs['uncertainty_CellType_Broad'] = embed.obs.loc[selected_cells, 'uncertainty_CellType_Broad']
    query_subset.obs['source_dataset'] = dataset_name

    return query_subset

def pool_celltypes(datasets, celltype_label, output_name):
    """
    Save cells of specific type from each dataset individually and as pooled dataset.

    Args:
        datasets: List of dataset names
        celltype_label: Cell type label in predictions
        output_name: Output subfolder name (cell type)

    Returns:
        Total number of cells across all datasets
    """
    logger.info(f"\nHarvesting {celltype_label} cells:")

    # Create subfolder for this cell type
    celltype_dir = OUTPUT_DIR / output_name
    celltype_dir.mkdir(parents=True, exist_ok=True)

    total_cells = 0
    saved_count = 0
    collected_datasets = []

    for ds in datasets:
        adata = harvest_cells(ds, celltype_label, UNCERTAINTY_THRESHOLD)
        if adata is not None:
            # Save individual dataset
            output_path = celltype_dir / f'{ds}.h5ad'
            adata.write_h5ad(output_path, compression='gzip')

            total_cells += adata.shape[0]
            saved_count += 1
            collected_datasets.append(adata)
            logger.info(f"  Saved {ds}: {adata.shape[0]} cells × {adata.shape[1]} genes")

    if saved_count == 0:
        logger.warning(f"No cells found for {celltype_label}")
        return 0

    # Concatenate and save pooled dataset
    if collected_datasets:
        pooled = ad.concat(collected_datasets, join='inner', merge='same')

        # Filter donors with > 100 cells
        if 'Donor' in pooled.obs.columns:
            donor_counts = pooled.obs['Donor'].value_counts()
            valid_donors = donor_counts[donor_counts > 100].index
            n_before = pooled.shape[0]
            pooled = pooled[pooled.obs['Donor'].isin(valid_donors)].copy()
            n_after = pooled.shape[0]
            logger.info(f"  Filtered donors: {len(valid_donors)}/{len(donor_counts)} donors kept, {n_after}/{n_before} cells")

        pooled_path = celltype_dir / f'pooled_{output_name}.h5ad'
        pooled.write_h5ad(pooled_path, compression='gzip')
        logger.info(f"  Saved pooled dataset: {pooled.shape[0]} cells × {pooled.shape[1]} genes")

    logger.info(f"Total: {saved_count} datasets, {total_cells} cells in {celltype_dir}")

    return total_cells

def main():
    logger.info("="*60)
    logger.info("Starting cell type harvest")
    logger.info(f"Datasets: {len(DATASETS)}")
    logger.info(f"Cell types: {list(TARGET_CELLTYPES.keys())}")
    logger.info(f"Uncertainty threshold: < {UNCERTAINTY_THRESHOLD}")
    logger.info("="*60)

    total_cells = {}
    for output_name, celltype_label in TARGET_CELLTYPES.items():
        n_cells = pool_celltypes(DATASETS, celltype_label, output_name)
        total_cells[output_name] = n_cells

    logger.info("\n" + "="*60)
    logger.info("Summary:")
    for ct, n in total_cells.items():
        logger.info(f"  {ct}: {n} cells")
    logger.info("="*60)

if __name__ == '__main__':
    main()
