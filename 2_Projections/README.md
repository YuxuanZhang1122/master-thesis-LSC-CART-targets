# 2_Projections

Train scANVI on reference atlas and project query samples to obtain embeddings, UMAP coordinates, and cell type labels via KNN.

## Workflow

```
Reference → scANVI Training → Query Projection → Cell Labeling → Downstream Analysis
```

## Input

| File | Location | Description |
|------|----------|-------------|
| `Reference_raw_hvg.h5ad` | `../Reference_raw_hvg.h5ad` | Reference atlas (HVG-filtered, raw counts) |
| Query files | `dataset/Queries/*.h5ad` | Query datasets to project |

## Output

| Directory | Contents |
|-----------|----------|
| `outputs/models/` | `scanvi_reference/` - Trained scANVI model<br>`umap_model.pkl` - Fitted UMAP model<br>`hvg_genes.txt` - HVG gene list |
| `outputs/embeddings/` | `reference_embeddings.h5ad` - Reference latent embeddings<br>`{query}_embeddings.h5ad` - Query latent embeddings + predictions |
| `outputs/figures/` | `Reference_umap/` - Reference visualizations<br>`{query}/` - Query UMAP projections + uncertainty plots |
| `outputs/PooledLSC/` | `HSC_MPP/`, `LMPP/`, `Early_GMP/` - Harvested cell populations |

## Scripts

### Core Pipeline

**`main.py`** - Train reference model
```bash
python main.py --train  # Force retrain
python main.py          # Use existing if available
```
- Trains scVI → scANVI on reference
- Fits UMAP model
- Generates reference visualizations

**`process_query.py`** - Project query sample
```python
# Edit query_path in script, then:
python process_query.py
```
- Loads query and reference model
- Performs scANVI surgery (query as new batch)
- Trains query-specific model
- KNN label transfer (fine + broad cell types)
- Projects query onto reference UMAP
- Saves embeddings, predictions, figures

### Downstream Analysis (`utility/`)

| Script | Purpose |
|--------|---------|
| `harvest_predictions.py` | Extract cells by predicted type + uncertainty threshold |
| `plot_HCbm_projections.py` | Multi-panel UMAP for bone marrow samples |
| `plot_patientwise_projections.py` | Patient-wise UMAP (HSPC/LSPC split) |
| `plot_KNNthreshold_umaps.py` | UMAP grid across uncertainty thresholds |
| `plot_KNNuncertainty_histograms.py` | Uncertainty distribution histograms |
| `create_dotplot.py` | Gene expression dot plots + cell state scores |

## Configuration

**Model Hyperparameters** (`main.py`):
- scVI: 30 latent dims, 2 layers, 128 hidden units
- scANVI: 30 epochs, batch size 4096
- UMAP: 15 neighbors, min_dist=0.2

**Query Processing** (`process_query.py`):
- KNN: 20 neighbors, distance-weighted
- Surgery: Freeze encoder, unfreeze decoder
- Confidence threshold: 0.2 (for visualization filtering)