# Reference Mapping of AML Samples onto Normal Bone Marrow Atlas

Pipeline for projecting AML single-cell RNA-seq data onto a healthy bone marrow reference atlas to infer differentiation states of leukemic stem/progenitor cells (LSPCs).

## Overview

This module maps query AML samples onto the [Normal Bone Marrow Atlas](https://github.com/andygxzeng/BoneMarrowMap) (Zeng et al. 2025) using scANVI for reference mapping and distance-weighted KNN for label transfer.

---

## Reference Atlas and Validation

![Figure](Figures/reference_mapping.png)
*Figure. Reference Mapping Reveals LSPC Heterogeneity Across Differentiation States. (A) UMAP visualization of the Zeng et al. (2025) normal bone marrow reference atlas comprising 263,159 cells from 45 healthy donors, annotated with 26 broad cell types. (B-C) Validation of scANVI reference mapping using enriched healthy bone marrow samples. Cells from CD34+CD38-CD45RA- sorted samples (Kaufmann et al. 2021, B; van Galen et al. 2019, C) mapped predominantly to HSC/MPP regions as expected, confirming mapping fidelity. (D-E) Mapping of author-annotated "primitive" LSPC from published AML studies reveals unexpected differentiation heterogeneity. LSPC from van Galen et al. 2019 unsorted samples (D) and Petti et al. 2019 unsorted samples (E) distribute across HSC/MPP, LMPP, and Early GMP states rather than uniformly mapping to the most primitive HSC/MPP compartment. (F) Reference mapping of 32 total unsorted AML bone marrow samples from Liljebjorn et al. (2025) demonstrates robust cross-dataset integration, with clear delineation of multiple hematopoietic lineages. Density contours highlight LSPC enrichment across primitive to myeloid progenitor states.*

---

## Query Datasets

Seven public AML scRNA-seq datasets were curated for reference mapping:

| Dataset | van Galen 2019 | Petti 2019 | Abbas 2021 | Guo 2023 | Naldini 2023 | Ennis 2023 | Liljebjorn 2025 |
|---------|----------------|------------|------------|----------|--------------|------------|-----------------|
| **Mutation data** | Yes | Yes | No | No | Yes | No | No |
| **No. AML pt.** | 16 | 5 | 8 | 20 | 10 | 10 | 32 |
| **Timepoint** | DG + MRD | DG | REL | DG | DG + MRD + REL | DG + MRD + REL | DG |
| **Cell count** | 30,712 | 79,142 | 22,793 | 56,168 | 69,138 | 78,339 | 202,181 |
| **Chemistry** | seqwell | 10x v5 | 10x v5 | 10x v5 | 10x v2, v3 | 10x v3 | 10x v3 |
| **Enrichment** | No | Yes | No | No | Yes | No | No |

*DG: diagnosis; MRD: minimal residual disease; REL: relapse*

---

## Label Transfer with Distance-Weighted KNN

<p align="center">
  <img src="Figures/KNN.png" width="500">
</p>
<p align="center"><em>Distance-weighted KNN label transfer. Query cells are projected onto the reference latent space via scANVI, and cell state labels are transferred using distance-weighted voting from k nearest reference neighbors.</em></p>

---

## Uncertainty Threshold Selection

Setting higher confidence thresholds (lower uncertainty) focuses projections on cells with reliable mappings:

![Threshold UMAPs](Figures/KNN_uncertainty_threshold_umaps.png)
*Effect of uncertainty thresholds on projected cell populations. Higher thresholds retain only confidently mapped cells.*

![Uncertainty Distributions](Figures/KNN_uncertainty_distributions.png)
*Quantification of cell retention across uncertainty thresholds. At threshold=0.2, >60% of HSC/MPP cells are filtered out, indicating high mapping uncertainty for the most primitive compartment.*

---

## Files

| File | Description |
|------|-------------|
| `main.py` | Train scANVI reference model on bone marrow atlas |
| `process_query.py` | Project query datasets and perform label transfer |
| `utils.py` | Utilities for contour drawing, highlighting, and visualization |

---

## Usage

```bash
# 1. Train reference model
python main.py

# 2. Project query dataset onto atlas
python process_query.py
```