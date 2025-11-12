import scanpy as sc
import matplotlib.pyplot as plt

adata = sc.read_h5ad('vanGalen_patient_ZengLabels.h5ad')

subset = adata[adata.obs.CellType.isin(['HSC-like','Prog-like','HSC','Prog'])].copy()

interest_sample = ['AML921A','AML707B','AML328','AML916']
samples = subset[subset.obs.patient_id.isin(interest_sample)].copy()
del subset,adata

# log normalization
sc.pp.normalize_total(samples, target_sum=1e4)
sc.pp.log1p(samples)

# Analyze all interest samples
for sample_id in interest_sample:
    print(f"Processing {sample_id}...")
    sample_data = samples[samples.obs.patient_id == sample_id]

    # run sam on a subset of cell types
    sam_LSPC = sc.external.tl.sam(sample_data[sample_data.obs.CellType.isin(['HSC-like', 'Prog-like'])], standardization="Normalizer", inplace=True)

    # run sam on all cell types
    sam_all = sc.external.tl.sam(sample_data, standardization="Normalizer", inplace=True)

    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Upper panel: Comparison of Zeng's label vs original labels for LSPC
    sc.pl.umap(sam_LSPC.adata, color='CellType', ax=axes[0, 0], show=False, title=f'{sample_id} - Original Labels (LSPC)')
    sc.pl.umap(sam_LSPC.adata, color='CellType_Zeng', ax=axes[0, 1], show=False, title=f'{sample_id} - Zeng Labels (LSPC)')

    # Lower panel: LSPC vs All cell types
    sc.pl.umap(sam_LSPC.adata, color='CellType', ax=axes[1, 0], show=False, title=f'{sample_id} - HSC-like & Prog-like')
    sc.pl.umap(sam_all.adata, color='CellType', ax=axes[1, 1], show=False, title=f'{sample_id} - All cell types')

    plt.tight_layout()
    plt.savefig(f'{sample_id}_sam.png')
    plt.show()
