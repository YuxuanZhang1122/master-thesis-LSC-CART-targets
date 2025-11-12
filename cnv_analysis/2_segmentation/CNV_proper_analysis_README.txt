CNV_proper_analysis.py - README
=====================================

OVERVIEW
--------
This script performs Copy Number Variation (CNV) analysis on single-cell RNA-seq data by comparing expression profiles between normal and malignant cell populations. It identifies genomic regions with significant expression changes that may indicate CNVs.

KEY FEATURES
------------
1. Proper data normalization to remove sequencing depth bias
2. Mean-based log2 ratio calculation (more accurate than averaging individual ratios)
3. Individual chromosome processing to avoid boundary artifacts
4. Gaussian smoothing for noise reduction
5. Segmentation analysis to identify CNV regions
6. Comprehensive 4-panel visualization for each chromosome

METHODOLOGY
-----------
1. Load and normalize data (total count normalization + log1p transformation)
2. Create reference profile from normal cells (HSC + Prog)
3. Calculate malignant group mean expression
4. Compute log2 ratio: log2(malignant_mean / normal_mean)  
5. Apply Gaussian smoothing to reduce noise
6. Segment into gain/loss/neutral regions using threshold-based approach
7. Generate individual plots for each chromosome

INPUT DATA
----------
- adata_filtered.h5ad: Filtered single-cell expression data
- AML328_purified.h5ad: Original data with cell type annotations  
- gene_positions.csv: Gene genomic coordinates

CELL POPULATIONS
----------------
- Normal: HSC (Hematopoietic Stem Cells) + Prog (Progenitor cells)
- Malignant: HSC-like + Prog-like cells

PARAMETERS
----------
- Chromosomes analyzed: 6, 7 (configurable)
- Smoothing window: 100 genes (sigma = window_size/6)
- Segmentation threshold: 0.3 (log2 fold-change)
- Minimum segment size: 10 genes

OUTPUT FILES
------------
For each chromosome:
- chr{X}_individual_cnv_analysis.png: 4-panel visualization showing:
  1. Normal reference expression profile
  2. Malignant mean expression profile  
  3. Direct expression comparison (normal vs malignant)
  4. Log2 ratio with CNV segmentation highlighting

SEGMENTATION RESULTS
--------------------
The script identifies and reports:
- Gain segments: log2 ratio > 0.3 (1.23x fold-change)
- Loss segments: log2 ratio < -0.3 (0.77x fold-change) 
- Neutral segments: |log2 ratio| ≤ 0.3

KEY IMPROVEMENTS
----------------
1. Fixed log2 ratio calculation artifacts from low expression values
2. Eliminated false "super large losses" from near-zero expressions
3. Removed chromosome boundary artifacts by processing individually
4. Applied proper normalization to account for sequencing depth differences
5. Corrected visualization to show actual data values instead of misleading fills

USAGE
-----
python CNV_proper_analysis.py

DEPENDENCIES  
------------
- pandas
- scanpy
- numpy
- matplotlib
- scipy.ndimage (gaussian_filter1d)
- warnings

NOTES
-----
- Normalization is critical: malignant cells have ~1.76x higher sequencing depth
- Mean-based ratio calculation is more robust than individual cell averaging
- Individual chromosome processing prevents artifacts at chromosome boundaries
- Segmentation threshold of 0.3 corresponds to ~23% expression change