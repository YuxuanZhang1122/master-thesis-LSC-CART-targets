Overview: mapping genes to their original physical location (CNV_MapGenes.py) then visualize the count/expression across chromosomes (CNV_overview_AML328.py); also check the expression profile across malignant/normal groups (CNV_sequencing_depth_analysis.py)

- id normal with variance: attempt to identify a normal reference based on their variance, std, variance and several metrics were calculated (across chromosomes) and consensus low variance cells were deemed as the normal reference (CNV_IdNormal_variance.py)

################################################################################################

chr6 - single cell comparison: take one single cell example out from each group (normal/malignant) to compare their expression profile (CNV_chr6_comparison.py); then different smoothing methods and window size were compared (CNV_chr6_smoothed_comparison.py)

################################################################################################

segmentation:
CNV_proper_analysis.py
1. Load and normalize data (total count normalization + log1p transformation)
2. Create reference profile from normal cells (HSC + Prog)
3. Calculate malignant group mean expression
4. Compute log2 ratio: log2(malignant_mean / normal_mean)  
5. Apply Gaussian smoothing to reduce noise (window size = 100)
6. Segment into gain/loss/neutral regions (10 consecutive genes) using threshold (0.3)
7. Generate individual plots for each chromosome

CNV_debug_chr6/11_segments.py: finer analysis

################################################################################################

hail_mary: since some alteration exists, pca+umap was used to identify distinct groups and failed (CNV_pca_umap.py)

################################################################################################

No log segmentation: same pipeline as CNV_proper_analysis.py but without log1p transform, and all chromosomes were processed with genes contributing to loss/gain identified (CNV_all_chromosomes_analysis.py) 
