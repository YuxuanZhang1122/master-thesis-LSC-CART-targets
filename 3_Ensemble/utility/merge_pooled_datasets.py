import scanpy as sc
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

POOLED_DIR = Path('')
OUTPUT_DIR = Path('')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

CELL_TYPES = ['HSC_MPP', 'LMPP', 'Early_GMP']

DATASETS = {
    'Ennis': ['DG', 'MRD', 'REL'],
    'Naldini_V03': ['DG', 'MRD', 'REL'],
    'Naldini_V02': ['DG', 'MRD'],
    'Henrik': ['DG'],
    'Petti': ['DG'],
    'vanGalen': [None]  # Special case: no time points
}

def merge_dataset(dataset_name, time_points):
    """Merge dataset across cell types and time points"""
    logger.info(f"\n{'='*70}")
    logger.info(f"Merging {dataset_name}")
    logger.info(f"{'='*70}")

    adatas = []

    # Special handling for vanGalen (no time points)
    if time_points == [None]:
        for ct in CELL_TYPES:
            file_path = POOLED_DIR / ct / f"{dataset_name}.h5ad"
            if file_path.exists():
                adata = sc.read_h5ad(file_path)
                logger.info(f"Loaded {ct}/{dataset_name}.h5ad: {adata.n_obs} cells")
                adatas.append(adata)
            else:
                logger.warning(f"File not found: {file_path}")
    else:
        # Regular datasets with time points
        for ct in CELL_TYPES:
            for tp in time_points:
                file_path = POOLED_DIR / ct / f"{dataset_name}_{tp}.h5ad"
                if file_path.exists():
                    adata = sc.read_h5ad(file_path)
                    logger.info(f"Loaded {ct}/{dataset_name}_{tp}.h5ad: {adata.n_obs} cells")
                    adatas.append(adata)
                else:
                    logger.warning(f"File not found: {file_path}")

    if not adatas:
        logger.error(f"No data found for {dataset_name}")
        return None

    # Merge all
    merged = sc.concat(adatas, join='outer', merge='same')
    logger.info(f"Merged {dataset_name}: {merged.n_obs} total cells")

    # For vanGalen, add prediction columns with CellType_Merged values
    if dataset_name == 'vanGalen':
        if 'CellType_Merged' in merged.obs.columns:
            merged.obs['consensus_prediction'] = merged.obs['CellType_Merged']
            merged.obs['consensus_label_5votes'] = merged.obs['CellType_Merged']
            merged.obs['consensus_label_6votes'] = merged.obs['CellType_Merged']
            logger.info(f"Added prediction columns for vanGalen with CellType_Merged values")
        else:
            logger.warning("CellType_Merged column not found in vanGalen data")

    # Save
    output_path = OUTPUT_DIR / f"{dataset_name}_DG.h5ad"
    merged.write_h5ad(output_path)
    logger.info(f"Saved to {output_path}")

    # Summary
    if 'cell_type' in merged.obs.columns:
        logger.info(f"Cell types: {merged.obs['cell_type'].value_counts().to_dict()}")
    if 'time_point' in merged.obs.columns:
        logger.info(f"Time points: {merged.obs['time_point'].value_counts().to_dict()}")

    return merged

def main():
    logger.info("="*70)
    logger.info("MERGING POOLED LSC DATASETS")
    logger.info("="*70)
    logger.info(f"Input directory: {POOLED_DIR}/")
    logger.info(f"Output directory: {OUTPUT_DIR}/")

    results = {}
    for dataset_name, time_points in DATASETS.items():
        result = merge_dataset(dataset_name, time_points)
        if result is not None:
            results[dataset_name] = result

    logger.info("\n" + "="*70)
    logger.info("MERGING COMPLETE")
    logger.info("="*70)
    logger.info(f"Merged {len(results)} datasets:")
    for name, adata in results.items():
        logger.info(f"  {name}_DG.h5ad: {adata.n_obs} cells, {adata.n_vars} genes")

if __name__ == '__main__':
    main()
