### General information

Authors: Henrik Lilljebjörn, Thoas Fioretos.
Contact e-mail: thoas.fioretos@med.lu.se; henrik.lilljebjorn@med.lu.se; 
DOI: 10.17044/scilifelab.23715648
License: CC BY-NC-ND 4.0
This readme file was last updated: 2025-07-09

Please cite as: Henrik Lilljebjörn, Pablo Peña-Martínez, Hanna Thorsson, Rasmus Henningsson, Marianne Rissler, Niklas Landberg, Noelia Puente-Moncada, Sofia von Palffy, Vendela Rissler, Petr Stanek, Jonathan Desponds, Xiangfu Zhong, Gunnar Juliusson, Vladimir Lazarevic, Sören Lehmann, Magnus Fontes, Helena Ågerstam, Carl Sandén, Christina Orsmark-Pietras, Thoas Fioretos. The cellular state space of AML unveils novel NPM1 subtypes with distinct clinical outcomes and immune evasion properties.

### Dataset description

This dataset contains 10X single cell 3' RNA sequencing gene expression data from from 38 AML-samples from the subtypes NPM1 (n=12), AML-MR (n=11), TP53 (n=7), CBFB::MYH11 (n=3), RUNX1::RUNX1T1 (n=3), AML without class defining mutations (n=1), and AML meeting the criteria for two subtypes (n=1). In addition, reference samples from normal bone marrow mononuclear cells (n=5) and CD34 sorted cells (n=3) are included. The single cell libraries were constructed from viably frozen cells from bone marrow (n=29+8) or peripheral blood (n=9) using the Chromium Single Cell 3' Library & Gel Bead Kit v3 (10X genomics) and sequenced on a Novaseq 6000 or NextSeq 500. Data is available in h5 format for each sample, with raw count output from Cellranger, or as a processed Seurat object with scaled expression data, dimension reductions, and metadata. Raw sequencing reads (fastq) are available at the European Genome-Phenome Archive (EGA) under accession ID EGAD50000001577: https://ega-archive.org/datasets/EGAD50000001577.

The files AMLX.h5 and NBMX-MNC/CD34.h5 each contain two matrices, the "Gene Expression" matrix contains read counts for 32738 genes for each cell from this sample and the "Antibody Capture" matrix contains read counts for 20 feature barcodes included in the experiment.

The file "Lilljebjorn_etal_scRNA-data_AML_NBM_Seurat4.rds" is an R data serialization (rds) file with a Seurat4 object containing gene expression and metadata for 245 073 cells and 23906 genes.

### Available variables

Assays:
RNA - original count data
SCT - scTransform scaled data
ADT - Feature barcode data

Meta.data:
The following cell level annotations (meta.data) are included:
"orig.ident" - sample identifier.
"nCount_RNA" - total number of molecules detected within this cell.
"nFeature_RNA" - total number of genes detected within this cell.
"nCount_ADT" - total number of ADT reads detected within this cell.
"nFeature_ADT" - total number of ADT features detected within this cell.
"percent.mt" - percentage of read counts originating from mitochondrial genes.
"nCount_SCT" - number of detected molecules after normalization.
"nFeature_SCT" - number of detected genes after normalization.
"AML.type" - genetic subtype of the sample.
"SCT_snn_res.1" - cluster identity as determined using Seurat FindClusters with resolution 1.
"seurat_clusters" - same as above.
"S.Score" - score from Seurats CellCycleScoring() function, S-phase.
"G2M.Score" - score from Seurats CellCycleScoring() function, G2M-phase.
"Phase" - cell cycle phase prediction by Seurats CellCycleScoring() function.
"predicted.celltype.l1" - predicted celltype, layer 1, 
"predicted.celltype.l1.score" - predicted celltype, layer 1 score, less detailed.
"predicted.celltype.l2" - predicted celltype, layer 2, more detailed.
"predicted.celltype.l2.score" - predicted celltype, layer 2 score
"celltype.l4" - merging of similar celltypes from predicted celltype layer2, less detailed than l2, more detailed than l1.
"celltype.aml" - same as celltype.l4, but clusters of cells containing immature AML cells have been assigned the celltype "AML Immature".
"AML.type.NPM1class" - genetic subtype of the sample, with NPM1 samples designated either NPM1 class I or NPM1 class II.