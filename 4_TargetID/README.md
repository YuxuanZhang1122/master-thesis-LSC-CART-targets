# 4_TargetID

Identify therapeutic targets by comparing malignant (LSPC) vs healthy (HSPC) cells using differential expression and pathway enrichment.

## Workflow

```
Stratified Cells → DEG Analysis → GSEA Enrichment → Target Prioritization
                  (Pseudobulk + Single-cell LMM)
```

## Input

| File | Location | Description |
|------|----------|-------------|
| `HSC_MPP.h5ad` | Root | Stratified cells from Step 3 (consensus predictions) |
| `HSC_MPP_surface.h5ad` | Root | Surface protein subset for targeted analysis |

**Required metadata columns**:
- `consensus_label_6votes`: HSPC (healthy) or LSPC (malignant)
- `Donor`: Patient/donor identifier
- `Study`: Dataset source

## Output

| Directory | Contents |
|-----------|----------|
| `DEG_results_pseudobulk_DESeq2/` | Pseudo-bulk DESeq2 results<br>`deseq2_results.csv` - All DEGs<br>`pseudobulk_volcano.png` - Volcano plot<br>`pseudobulk_vs_lmm_correlation.png` - Method comparison |
| `DEG_results_singlecell_LMM/` | Single-cell LMM results<br>`singlecell_deg_all_results.csv` - All genes<br>`singlecell_deg_significant.csv` - Significant DEGs<br>`singlecell_volcano.png` - Volcano plot |
| `DEG_results_singlecell_LMM_surfaceproteinonly/` | LMM results for surface proteins only |
| `gsea_results/` | GSEA pathway enrichment<br>`gsea_hallmark_results.csv`<br>`gsea_gobp_results.csv`<br>`gsea_kegg_results.csv`<br>Summary plots and gene tables |
| `umap/` | Feature expression UMAPs |
| `surface_proteins/` | Surface protein target lists |

## Scripts

### Differential Expression

**`pseudobulk_deg_DESeq2.py`** - Paired pseudo-bulk analysis
```bash
python pseudobulk_deg_DESeq2.py
```
- Aggregates cells to donor-level pseudo-bulk samples
- Requires paired donors (both HSPC + LSPC ≥ 10 cells)
- DESeq2: `~ Donor + Status` (paired design)
- Compares with single-cell LMM results

**`singlecell_deg_LMM.py`** - Single-cell linear mixed model
```bash
python singlecell_deg_LMM.py
```
- Model: `expression ~ Status + Study + (1|Donor)`
- Donor as random effect (accounts for pseudoreplication)
- Filters: ribosomal, mitochondrial, cell cycle, housekeeping genes
- Gene inclusion: expressed in ≥5% malignant cells, mean CPM ≥0.1

### Pathway Enrichment

**`run_gsea.py`** - Gene Set Enrichment Analysis
```bash
python run_gsea.py
```
- Ranks genes by t-statistic (coefficient/stderr)
- Runs preranked GSEA on 3 databases:
  - MSigDB Hallmark 2020
  - GO Biological Process 2023
  - KEGG 2021 Human
- Creates summary plots and pathway gene tables

### Visualization

**`plot_volcano.py`** - Volcano plot generation
```python
from plot_volcano import create_volcano_plot
# Called automatically by DEG scripts
```

**`plot_gsea.py`** - GSEA-specific plots
```python
# Creates themed lollipop plots for LSC-specific pathways
```

**`umap_featureplot.py`** - Gene expression UMAPs
```bash
python umap_featureplot.py
```

## Configuration

### Pseudobulk DESeq2
- Min cells per donor: 50
- Min cells per status (paired): 10
- Gene filter: expressed in ≥1% cells
- Thresholds: FDR < 0.05, |log2FC| > 1

### Single-cell LMM
- Min cells per donor: 50
- Gene inclusion: ≥5% malignant cells, mean CPM ≥0.1
- LSPC expression > HSPC expression (for upregulated targets)
- Thresholds: FDR < 0.05, |log2FC| > 1

### GSEA
- Permutations: 1000
- Pathway size: 15-500 genes
- FDR threshold: 0.25
- Ranking metric: t-statistic

## Output Format

**DEG results** (CSV columns):
- `gene`: Gene symbol
- `log2FoldChange`: Effect size
- `padj`: FDR-adjusted p-value
- `coefficient`: LMM effect estimate (single-cell only)
- `stderr`: Standard error (single-cell only)
- `pct_malignant`, `pct_healthy`: Expression percentages
- `mean_malignant`, `mean_healthy`: Mean CPM values

**GSEA results** (CSV columns):
- `Term`: Pathway name
- `NES`: Normalized enrichment score
- `FDR q-val`: FDR-corrected q-value
- `NOM p-val`: Nominal p-value
- `Lead_genes`: Leading edge genes

## Typical Workflow

```bash
# 1. Run single-cell LMM (all genes)
python singlecell_deg_LMM.py

# 2. Run surface protein-focused analysis
# Edit INPUT_FILE to 'HSC_MPP_surface.h5ad', then:
python singlecell_deg_LMM.py

# 3. Run paired pseudo-bulk DESeq2
python pseudobulk_deg_DESeq2.py

# 4. Pathway enrichment
python run_gsea.py

# 5. Visualize candidates
python umap_featureplot.py
```