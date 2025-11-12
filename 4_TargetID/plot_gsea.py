"""
LSC vs HSC - Refined GSEA Visualization

Publication-ready lollipop plot with:
- LSPC-specific pathways only
- X-axis: -log10(FDR) for significance
- Dot size: abs(NES) for effect size magnitude
- Clean pathway names (no GO IDs)
- Specific pathways excluded
- Expanded stemness, differentiation, and cell cycle themes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')

# Configuration
RESULTS_DIR = 'gsea_results'
OUTPUT_FILE = 'lsc_themed_lollipop.png'
FDR_THRESHOLD = 0.25
TOP_N_PER_THEME = 5


# Pathways to specifically EXCLUDE
EXCLUDE_PATHWAYS = [
    'UV Response Up',
    'IL-17 signaling pathway',
    'mTORC1 Signaling',
    'Fatty Acid Metabolism',
    'Xenobiotic Metabolism',
    'Response To Reactive Oxygen Species (GO:0000302)',
    'Regulation Of Signal Transduction By P53 Class',
    'Regulation Of B Cell Proliferation',
    'Mitochondrial ATP Synthesis Coupled Electron Transport',
    'Proton Motive Force-Driven Mitochondrial ATP Synthesis',
    'Oxidative Phosphorylation (GO:0006119)',  # GO version (keep KEGG/Hallmark)
    'Myeloid Leukocyte Differentiation',  # Keep only Regulation Of Myeloid Cell Differentiation,
    'Mitochondrial Electron Transport, NADH To Ubiquinone',
    'Regulation Of ERK1 And ERK2 Cascade',
    'Negative Regulation Of Cell Cycle'
]

# General exclusion keywords (non-HSPC biology)
EXCLUDE_KEYWORDS = [
    'th1', 'th2', 'th17', 'treg', 't helper',
    'osteoclast', 'osteoblast', 'bone',
    'neuron', 'synapse', 'axon', 'brain', 'substantia nigra',
    'cardiac', 'muscle contraction', 'cardiomyopathy',
    'parkinson', 'alzheimer', 'prion',
    'addiction', 'cocaine', 'amphetamine',
    'taste', 'olfactory', 'sensory',
    'circadian', 'sleep',
    'sperm', 'oocyte', 'fertilization',
    'photoreceptor', 'retina', 'vision',
    'spermatogenesis'
]

# Theme definitions with specific pathways
THEME_PATHWAYS = {
    'LSC Stemness & Self-Renewal': {
        'keywords': ['myc target', 'notch signaling', 'wnt', 'hedgehog',
                    'stem cell', 'pluripotency', 'self-renewal', 'hoxa', 'hox',
                    'myeloid differentiation'],
        'priority_pathways': [
            'Myc Targets V1',
            'Regulation Of Notch Signaling Pathway',
            'Regulation Of Stem Cell Differentiation',
            'Regulation Of Wnt Signaling Pathway',
            'Regulation Of Stem Cell Population Maintenance',
            'Positive Regulation Of Stem Cell Population Maintenance',
            'Regulation Of Myeloid Cell Differentiation'
        ]
    },
    'Cell Cycle & Proliferation': {
        'keywords': ['g2m', 'g2 m', 'e2f', 'cell cycle', 'mitotic', 'checkpoint',
                    'spindle', 'chromosome segregation', 'dna replication',
                    'cyclin', 'cdk', 'proliferation', 'quiescence'],
        'priority_pathways': [
            'E2F Targets',
            'G2-M Checkpoint',
            'Cell cycle',
            'Negative Regulation Of Cell Cycle',
            'PD-L1 expression and PD-1 checkpoint pathway in cancer',
            'Positive Regulation Of Cell Population Proliferation'
        ]
    },
    'Apoptosis & Cell Death Evasion': {
        'keywords': ['apoptosis', 'p53', 'cell death', 'bcl2', 'caspase',
                    'programmed cell death', 'necroptosis', 'pyroptosis'],
        'priority_pathways': [
            'Apoptosis',
            'p53 Pathway',
            'Necroptosis'
        ]
    },
    'Inflammatory & Stress Response': {
        'keywords': ['inflammatory response', 'il-6', 'il-1', 'tnf', 'nfkb',
                     'interferon gamma', 'interferon alpha', 'innate immune',
                     'stress response', 'unfolded protein', 'hypoxia',
                     'dna damage'],
        'priority_pathways': [
            'TNF signaling pathway',
            'Integrated Stress Response Signaling',
            'Hypoxia'
        ]
    },
    'Metabolic Reprogramming': {
        'keywords': ['metabolism', 'glycolysis', 'oxidative phosphorylation',
                    'oxphos', 'tca', 'respiration','mitochondrial',
                    'reactive oxygen'],
        'priority_pathways': [
            'Oxidative Phosphorylation',
            'Glycolysis',
            'Reactive Oxygen Species Pathway'
            'Cellular Respiration'
        ]
    },
    'HSPC Differentiation Block': {
        'keywords': ['hematopoietic', 'erythrocyte',
                    'megakaryocyte', 'granulocyte', 'monocyte', 'mpp',
                    'hematopoiesis', 'blood cell', 'lymphoid', 'leukocyte differentiation'],
        'priority_pathways': []
    },
    'Oncogenic Signaling Pathways': {
        'keywords': ['pi3k', 'akt', 'mtor', 'ras signaling', 'mapk', 'jak', 'stat',
                    'kras', 'erk', 'proteasome'],
        'priority_pathways': [
            'Proteasome',
            'PI3K/AKT/mTOR Signaling',
            'Ras signaling pathway',
            'Negative Regulation Of ERK1 And ERK2 Cascade'
        ]
    }
}


def clean_pathway_name(pathway):
    """Remove GO IDs and shorten pathway names."""
    # Remove GO IDs like (GO:0012345)
    pathway = re.sub(r'\s*\(GO:\d+\)', '', pathway)
    pathway = pathway.strip()

    # Mapping dictionary for shortening names
    name_map = {
        'Myc Targets V1': 'Myc Stemness',
        'Signaling pathways regulating pluripotency of stem cells': 'Pluripotency Signaling',
        'Cell cycle': 'Cell Cycle',
        'Positive Regulation Of Cell Population Proliferation': 'Cell Proliferation',
        'PD-L1 expression and PD-1 checkpoint pathway in cancer': 'PD-1/PD-L1 Checkpoint',
        'Negative Regulation Of Cell Cycle': 'Cell Cycle Inhibition',
        'Necroptosis': 'Necroptosis',
        'p53 Pathway': 'p53 Pathway',
        'Apoptosis': 'Apoptosis',
        'Oxidative phosphorylation': 'OXPHOS',
        #'Oxidative Phosphorylation': 'OXPHOS',
        'Cellular Respiration': 'Cellular Respiration',
        'Reactive Oxygen Species Pathway': 'ROS Pathway',
        'Glycolysis': 'Glycolysis',
        'Cellular Response To Reactive Oxygen Species': 'ROS Response',
        'Regulation Of Myeloid Cell Differentiation': 'Myeloid Differentiation',
        'Myeloid Leukocyte Differentiation': 'Myeloid Differentiation',
        'Hypoxia': 'Hypoxia',
        'TNF signaling pathway': 'TNF Signaling',
        'Integrated Stress Response Signaling': 'Stress Response',
        'Proteasome': 'Proteasome',
        'Negative Regulation Of ERK1 And ERK2 Cascade': 'ERK Inhibition',
        'Regulation Of ERK1 And ERK2 Cascade': 'ERK Regulation',
        'Regulation Of Stress-Activated MAPK Cascade': 'MAPK Regulation',
        'Ras signaling pathway': 'RAS Signaling',
        'PI3K/AKT/mTOR  Signaling': 'PI3K/AKT/mTOR',
        'PI3K/AKT/mTOR Signaling': 'PI3K/AKT/mTOR',
        'mTOR signaling pathway': 'mTOR Signaling'
    }

    # Check if we have a shortened version
    if pathway in name_map:
        return name_map[pathway]

    return pathway


def load_gsea_results(results_dir):
    """Load GSEA results from all databases."""
    results_path = Path(results_dir)
    all_results = []

    for db_file in ['gsea_hallmark_results.csv', 'gsea_gobp_results.csv',
                     'gsea_kegg_results.csv']:
        file_path = results_path / db_file
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['FDR q-val'] = pd.to_numeric(df['FDR q-val'], errors='coerce')
            df['Database'] = db_file.replace('gsea_', '').replace('_results.csv', '').upper()
            all_results.append(df)

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()


def map_pathways_to_themes(results_df, fdr_threshold=0.25):
    """Map pathways to themes with exclusions and prioritization."""
    theme_data = {theme: [] for theme in THEME_PATHWAYS.keys()}

    sig_results = results_df[results_df['FDR q-val'] < fdr_threshold].copy()

    for _, row in sig_results.iterrows():
        pathway = row['Term']
        pathway_clean = clean_pathway_name(pathway)
        pathway_lower = pathway_clean.lower()
        nes = row['NES']
        fdr = row['FDR q-val']
        db = row['Database']

        # EXCLUDE specific pathways
        if any(excl in pathway for excl in EXCLUDE_PATHWAYS):
            continue

        # EXCLUDE irrelevant keywords
        if any(exclude_kw in pathway_lower for exclude_kw in EXCLUDE_KEYWORDS):
            continue

        # Prioritize Hallmark and priority pathways
        is_hallmark = (db == 'HALLMARK')
        is_priority = False

        for theme, info in THEME_PATHWAYS.items():
            if pathway in info.get('priority_pathways', []) or \
               pathway_clean in info.get('priority_pathways', []):
                is_priority = True
                break

        priority_score = abs(nes) * (3.0 if is_priority else (2.0 if is_hallmark else 1.0))

        # Match to theme
        matched = False
        for theme, info in THEME_PATHWAYS.items():
            # Check if it's a priority pathway for this theme
            if pathway in info.get('priority_pathways', []) or \
               pathway_clean in info.get('priority_pathways', []):
                theme_data[theme].append({
                    'Pathway': pathway_clean,
                    'NES': nes,
                    'FDR': fdr,
                    'Database': db,
                    'abs_NES': abs(nes),
                    'priority_score': priority_score
                })
                matched = True
                break

            # Otherwise check keywords
            if not matched and any(keyword in pathway_lower for keyword in info['keywords']):
                theme_data[theme].append({
                    'Pathway': pathway_clean,
                    'NES': nes,
                    'FDR': fdr,
                    'Database': db,
                    'abs_NES': abs(nes),
                    'priority_score': priority_score
                })
                matched = True
                break

    return theme_data


def select_top_pathways_per_theme(theme_data, top_n=6):
    """Select top pathways per theme prioritizing important ones."""
    selected = {}

    for theme, pathways in theme_data.items():
        if len(pathways) == 0:
            selected[theme] = []
            continue

        df = pd.DataFrame(pathways)
        df = df.sort_values('priority_score', ascending=False)

        # Get top N from each direction
        pos = df[df['NES'] > 0].head(top_n)
        neg = df[df['NES'] < 0].head(top_n)

        combined = pd.concat([pos, neg])

        selected[theme] = combined.to_dict('records')

    return selected


def create_lollipop_plot(theme_data, output_file):
    """Create publication-ready lollipop plot with legends on right."""

    theme_colors = {
        'LSC Stemness & Self-Renewal': '#E74C3C',
        'Cell Cycle & Proliferation': '#F39C12',
        'Apoptosis & Cell Death Evasion': '#8E44AD',
        'Metabolic Reprogramming': '#3498DB',
        'HSPC Differentiation Block': '#16A085',
        'Inflammatory & Stress Response': '#27AE60',
        'Oncogenic Signaling Pathways': '#E67E22'
    }

    # Prepare data
    plot_data = []
    theme_positions = {}
    current_y = 0

    for theme in THEME_PATHWAYS.keys():
        pathways = theme_data.get(theme, [])
        if len(pathways) == 0:
            continue

        pathways_sorted = sorted(pathways, key=lambda x: x['NES'])

        theme_start = current_y
        for pathway_info in pathways_sorted:
            # Calculate -log10(FDR) for significance
            fdr = pathway_info['FDR']
            neg_log_fdr = -np.log10(max(fdr, 1e-10))  # Avoid log(0)

            plot_data.append({
                'y': current_y,
                'NES': pathway_info['NES'],
                'neg_log_fdr': neg_log_fdr,
                'Pathway': pathway_info['Pathway'],
                'Theme': theme,
                'FDR': fdr,
                'Database': pathway_info['Database']
            })
            current_y += 1

        theme_positions[theme] = (theme_start, current_y - 1)
        current_y += 0.5

    if len(plot_data) == 0:
        print("No pathways to plot!")
        return

    # Create figure (narrower width)
    fig, ax = plt.subplots(figsize=(7, max(10, len(plot_data) * 0.4)))

    # Plot lollipops
    for item in plot_data:
        y = item['y']
        nes = item['NES']
        sig = item['neg_log_fdr']
        theme = item['Theme']
        color = theme_colors[theme]

        # Make x-position symmetric: positive NES to right, negative to left
        x_pos = sig if nes > 0 else -sig

        # Color line based on direction
        line_color = '#E74C3C' if nes > 0 else '#3498DB'
        ax.plot([0, x_pos], [y, y], color=line_color, linewidth=2.5, alpha=0.5)

        # Dot size based on abs(NES)
        size = abs(nes) * 250
        ax.scatter(x_pos, y, s=size, color=color, alpha=0.85,
                  edgecolors='white', linewidths=2, zorder=3)

        # Add NES value as text label
        offset = -0.45 if nes > 0 else 0.5
        ha = 'left' if nes > 0 else 'right'
        ax.text(x_pos + offset, y, f'{nes:.1f}',
               va='center', ha=ha, fontsize=8, fontweight='bold')

    # Set pathway labels (shortened names, no database tags)
    ax.set_yticks([item['y'] for item in plot_data])
    pathway_labels = [item['Pathway'] for item in plot_data]
    ax.set_yticklabels(pathway_labels, fontsize=10)

    # Vertical line at x=0 (center divider)
    ax.axvline(0, color='grey', linewidth=1.5, linestyle='--', alpha=0.6)

    # Styling
    ax.set_xlabel('-log10(FDR q-value)',
                 fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Add subtle separator lines between themes (no labels)
    for theme, (start, end) in theme_positions.items():
        if end < len(plot_data) - 1:
            ax.axhline(end + 0.5, color='gray', linewidth=0.5,
                      linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def print_summary(theme_data):
    """Print minimal summary of selected pathways."""
    n_pathways = sum(len(pathways) for pathways in theme_data.values())
    if n_pathways > 0:
        print(f'\nLSC-specific GSEA visualization created with {n_pathways} pathways')


if __name__ == '__main__':
    # Load results
    results_df = load_gsea_results(RESULTS_DIR)

    # Map to themes
    theme_data = map_pathways_to_themes(results_df, FDR_THRESHOLD)

    # Select top pathways
    selected = select_top_pathways_per_theme(theme_data, TOP_N_PER_THEME)

    # Create plot
    create_lollipop_plot(selected, OUTPUT_FILE)

    # Print summary
    print_summary(selected)
