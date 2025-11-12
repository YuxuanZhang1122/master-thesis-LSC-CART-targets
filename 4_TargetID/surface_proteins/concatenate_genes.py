import pandas as pd
from pathlib import Path

files = ["database/CSPA.xlsx", "database/CellphoneDB.xlsx", "database/ML_eth_surfaceome.xlsx"]

all_genes = set()

for file in files:
    df = pd.read_excel(file)

    if 'GENE_SYMBOL' in df.columns:
        genes = df['GENE_SYMBOL'].dropna().astype(str).unique()
        all_genes.update(genes)
        print(f"{file}: {len(genes)} unique genes")

all_genes = sorted(all_genes)
print(f"\nTotal unique genes across all databases: {len(all_genes)}")

output_df = pd.DataFrame({'GENE_SYMBOL': all_genes})
output_path = "MasterList_surface_protein_gene.xlsx"
output_df.to_excel(output_path, index=False)
print(f"\nSaved to: {output_path}")