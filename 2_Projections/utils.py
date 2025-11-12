import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import gaussian_kde
import logging
import umap.umap_ as umap

logger = logging.getLogger(__name__)


def load_data(file_path):
    """Load and validate h5ad file."""
    adata = ad.read_h5ad(file_path)
    return adata


def save_embeddings(adata, output_path):
    """Save embeddings with metadata."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)


def compute_knn_predictions(ref_embed, ref_labels, query_embed, n_neighbors=15, weights='distance', metric='euclidean'):
    """
    Transfer labels using KNN classifier.

    Returns:
        predictions: Predicted labels
        probabilities: Class probabilities
        uncertainty: 1 - max(probability)
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
    knn.fit(ref_embed, ref_labels)

    predictions = knn.predict(query_embed)
    probabilities = knn.predict_proba(query_embed)

    max_prob = probabilities.max(axis=1)
    uncertainty = 1 - max_prob

    return predictions, probabilities, uncertainty


def fit_umap_model(ref_embed, n_neighbors=15, min_dist=0.2, random_state=42):
    """
    Fit UMAP model on reference embeddings.

    Args:
        ref_embed: Reference embeddings
        n_neighbors: Number of neighbors
        min_dist: Minimum distance
        random_state: Random state

    Returns:
        Fitted UMAP model and transformed coordinates
    """
    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric='euclidean',
        random_state=random_state
    )

    ref_umap = umap_model.fit_transform(ref_embed)

    return umap_model, ref_umap


def compute_combined_umap(ref_embed, query_embed, umap_model=None, n_neighbors=15, min_dist=0.2, random_state=42, preserve_reference=True):
    """
    Project query embeddings onto reference UMAP space.

    Args:
        ref_embed: Reference embeddings (latent space)
        query_embed: Query embeddings (latent space)
        umap_model: Pre-fitted UMAP model (if None, will fit on reference)
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance for UMAP
        random_state: Random state for reproducibility
        preserve_reference: If True, project query onto reference UMAP

    Returns:
        Query UMAP coordinates
    """
    if preserve_reference and umap_model is not None:
        query_umap = umap_model.transform(query_embed)
    else:
        combined = np.vstack([ref_embed, query_embed])

        umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            metric='euclidean',
            random_state=random_state
        )

        umap_coords = umap_model.fit_transform(combined)
        n_ref = len(ref_embed)
        query_umap = umap_coords[n_ref:]

    return query_umap


def plot_umap(
    coords,
    labels,
    title,
    output_path,
    figsize=(10, 8),
    point_size=0.8,
    alpha=0.6,
    dpi=300,
    palette=None,
    show_legend=True
):
    """Create UMAP visualization with configurable legend placement."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)

    if palette is None:
        if n_labels <= 10:
            palette = sns.color_palette("tab10", n_labels)
        elif n_labels <= 20:
            palette = sns.color_palette("tab20", n_labels)
        else:
            palette = sns.color_palette("husl", n_labels)

    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            alpha=alpha,
            label=label,
            c=[palette[i]],
            rasterized=True
        )

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')

    if show_legend:
        if n_labels <= 15:
            ax.legend(loc='center right', fontsize=9, frameon=True, fancybox=True,
                     framealpha=0.8, edgecolor='gray')
        elif n_labels <= 30:
            ax.legend(bbox_to_anchor=(1.1, 0.95), loc='best', fontsize=12, markerscale=6,
                    frameon=False)
        elif n_labels <= 60:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='best', fontsize=11, markerscale=6, labelspacing = 0.6,
                     ncol=2, frameon=False, columnspacing=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_umap_with_labels(
    coords,
    labels,
    title,
    output_path,
    figsize=(10, 8),
    point_size=0.8,
    alpha=0.6,
    dpi=300,
    palette=None,
    show_legend=False,
    label_fontsize=8
):
    """Create UMAP visualization with text labels at cluster centroids."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)

    if palette is None:
        if n_labels <= 10:
            palette = sns.color_palette("tab10", n_labels)
        elif n_labels <= 20:
            palette = sns.color_palette("tab20", n_labels)
        else:
            palette = sns.color_palette("husl", n_labels)

    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            alpha=alpha,
            label=label,
            c=[palette[i]],
            rasterized=True
        )

    for i, label in enumerate(unique_labels):
        mask = labels == label
        coords_subset = coords[mask]
        if len(coords_subset) > 0:
            centroid_x = np.mean(coords_subset[:, 0])
            centroid_y = np.mean(coords_subset[:, 1])

            ax.text(
                centroid_x, centroid_y,
                label,
                fontsize=label_fontsize,
                fontweight='bold',
                ha='center',
                va='center',
                color='black',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.4, edgecolor='none')
            )

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')

    if show_legend:
        if n_labels <= 15:
            ax.legend(loc='center right', fontsize=9, frameon=True, fancybox=True,
                     framealpha=0.8, edgecolor='gray')
        elif n_labels <= 30:
            ax.legend(bbox_to_anchor=(1.1, 0.95), loc='best', fontsize=10, markerscale=6,
                    frameon=False)
        else:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='best', fontsize=9, markerscale=6,
                     ncol=2, frameon=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_umap_numbered(
    coords,
    labels,
    title,
    output_path,
    figsize=(16, 8),
    point_size=0.8,
    alpha=0.6,
    dpi=300,
    palette=None,
    number_fontsize=7
):
    """
    Create UMAP visualization with numbered labels and side table legend.
    Also saves the number-to-label mapping as CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)

    label_to_num = {label: i+1 for i, label in enumerate(unique_labels)}

    if palette is None:
        if n_labels <= 10:
            palette = sns.color_palette("tab10", n_labels)
        elif n_labels <= 20:
            palette = sns.color_palette("tab20", n_labels)
        else:
            palette = sns.color_palette("husl", n_labels)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1], wspace=0.3)
    ax_umap = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])

    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax_umap.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            alpha=alpha,
            c=[palette[i]],
            rasterized=True
        )

    for i, label in enumerate(unique_labels):
        mask = labels == label
        coords_subset = coords[mask]
        if len(coords_subset) > 0:
            centroid_x = np.mean(coords_subset[:, 0])
            centroid_y = np.mean(coords_subset[:, 1])
            number = label_to_num[label]

            ax_umap.text(
                centroid_x, centroid_y,
                str(number),
                fontsize=number_fontsize,
                fontweight='bold',
                ha='center',
                va='center',
                color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.5)
            )

    ax_umap.set_xlabel("UMAP 1", fontsize=12)
    ax_umap.set_ylabel("UMAP 2", fontsize=12)
    ax_umap.set_title(title, fontsize=12, fontweight='bold')
    ax_umap.axis('tight')

    ax_table.axis('off')

    n_cols = 2 if n_labels > 25 else 1
    items_per_col = (n_labels + n_cols - 1) // n_cols

    table_text = []
    for col in range(n_cols):
        col_items = []
        start_idx = col * items_per_col
        end_idx = min(start_idx + items_per_col, n_labels)

        for idx in range(start_idx, end_idx):
            label = unique_labels[idx]
            number = label_to_num[label]
            col_items.append(f"{number:2d}. {label}")

        table_text.append('\n'.join(col_items))

    if n_cols == 1:
        text_x = 0.05
        ax_table.text(text_x, 0.95, table_text[0],
                     fontsize=10, va='top', ha='left', family='monospace',
                     linespacing=1.6)
    else:
        text_x1 = -0.4
        text_x2 = 0.5
        ax_table.text(text_x1, 0.95, table_text[0],
                     fontsize=10, va='top', ha='left', family='monospace',
                     linespacing=1.6)
        if len(table_text) > 1:
            ax_table.text(text_x2, 0.95, table_text[1],
                         fontsize=10, va='top', ha='left', family='monospace',
                         linespacing=1.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    csv_path = output_path.with_suffix('.csv')
    mapping_df = pd.DataFrame([
        {'Number': label_to_num[label], 'CellType': label}
        for label in unique_labels
    ])
    mapping_df.to_csv(csv_path, index=False)


def plot_umap_highlight(
    coords,
    labels,
    highlight_labels,
    title,
    output_path,
    figsize=(10, 8),
    point_size=0.8,
    alpha=0.2,
    dpi=300
):
    """
    Plot UMAP with specific cell types highlighted in color, others in grey.

    Args:
        coords: UMAP coordinates
        labels: Cell type labels
        highlight_labels: List of cell types to highlight (e.g., ['HSC MPP', 'GMP'])
        title: Plot title
        output_path: Save path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    if isinstance(highlight_labels, str):
        highlight_labels = [highlight_labels]

    unique_labels = np.unique(labels)
    highlight_set = set(highlight_labels)

    highlighted_labels = [l for l in unique_labels if l in highlight_set]
    n_highlight = len(highlighted_labels)

    palette = sns.color_palette("tab10", n_highlight)

    mask_bg = ~np.isin(labels, highlight_labels)
    ax.scatter(
        coords[mask_bg, 0],
        coords[mask_bg, 1],
        s=point_size,
        alpha=0.5,
        c='lightgray',
        label='Other cell types',
        rasterized=True
    )

    for i, label in enumerate(highlighted_labels):
        mask = labels == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size * 2,
            alpha=alpha,
            label=label,
            c=[palette[i]],
            edgecolors='white',
            linewidths=0.3,
            rasterized=True
        )

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(loc='upper right', fontsize=12, frameon=True, markerscale=6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_combined_umap(
    ref_coords,
    query_coords,
    ref_labels,
    query_labels,
    title,
    output_path,
    figsize=(10, 8),
    point_size=5,
    alpha=0.5,
    dpi=300
):
    """Plot combined reference and query UMAP with enhanced dots."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        ref_coords[:, 0],
        ref_coords[:, 1],
        s=0.8,
        alpha=0.1,
        c='lightgray',
        label='Reference',
        rasterized=True
    )

    unique_labels = np.unique(query_labels)
    n_labels = len(unique_labels)

    if n_labels <= 10:
        palette = sns.color_palette("tab10", n_labels)
    elif n_labels <= 20:
        palette = sns.color_palette("tab20", n_labels)
    else:
        palette = sns.color_palette("husl", n_labels)

    for i, label in enumerate(unique_labels):
        mask = query_labels == label
        ax.scatter(
            query_coords[mask, 0],
            query_coords[mask, 1],
            s=point_size,
            alpha=alpha,
            label=f"{label}",
            c=[palette[i]],
            edgecolors='None',
            linewidths=0.3
        )

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.axis('tight')
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_combined_umap_contour(
    ref_coords,
    query_coords,
    ref_labels,
    query_labels,
    title,
    output_path,
    figsize=(10, 8),
    point_size=4,
    alpha=0.5,
    dpi=300
):
    """
    Plot density-focused UMAP: most frequent cell type with density gradient (red->purple),
    others as yellow dots, with direct labels for top 2 types.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        ref_coords[:, 0],
        ref_coords[:, 1],
        s=0.8,
        alpha=0.1,
        c='lightgray',
        rasterized=True
    )

    label_counts = pd.Series(query_labels).value_counts()
    total = len(query_labels)
    label_props = label_counts / total
    top_labels = label_props[label_props >= 0.1].index.tolist()
    most_frequent = label_counts.index[0]

    kde = gaussian_kde(query_coords.T)
    density = kde(query_coords.T)

    mask_top = query_labels == most_frequent
    coords_top = query_coords[mask_top]
    coords_others = query_coords[~mask_top]

    ax.scatter(
        coords_top[:, 0],
        coords_top[:, 1],
        c=density[mask_top],
        s=point_size,
        cmap='plasma',
        alpha=0.6,
        edgecolors='face',
        rasterized=True
    )
    ax.scatter(
        coords_others[:, 0],
        coords_others[:, 1],
        c=density[~mask_top],
        s=point_size*0.7,
        cmap='plasma',
        alpha=0.4,
        edgecolors='face',
        rasterized=True
    )

    if len(coords_top) > 10:
        x_min, x_max = coords_top[:, 0].min(), coords_top[:, 0].max()
        y_min, y_max = coords_top[:, 1].min(), coords_top[:, 1].max()

        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= x_range * 0.1
        x_max += x_range * 0.1
        y_min -= y_range * 0.1
        y_max += y_range * 0.1

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100),
            np.linspace(y_min, y_max, 100)
        )

        grid_coords = np.vstack([xx.ravel(), yy.ravel()])
        density_grid = kde(grid_coords).reshape(xx.shape)
        threshold = np.percentile(density_grid, 60)
        density_masked = np.ma.masked_where(density_grid < threshold, density_grid)

        ax.contour(
            xx, yy, density_masked,
            levels=4,
            colors='purple',
            alpha=0.75,
            linewidths=1.5
        )

    for label in top_labels:
        mask = query_labels == label
        coords_subset = query_coords[mask]
        if len(coords_subset) > 0:
            centroid_x = np.mean(coords_subset[:, 0])
            centroid_y = np.mean(coords_subset[:, 1])

            ax.text(
                centroid_x, centroid_y,
                label,
                fontsize=8,
                fontweight='bold',
                ha='center',
                va='center',
                color='black',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.4, edgecolor='none')
            )

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.axis('tight')
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_uncertainty(coords, uncertainty, title, output_path, ref_coords=None, figsize=(10, 8), point_size=5, dpi=300):
    """
    Plot UMAP colored by uncertainty scores with optional reference background.

    Args:
        coords: Query UMAP coordinates
        uncertainty: Uncertainty values for query cells
        title: Plot title
        output_path: Save path
        ref_coords: Optional reference UMAP coordinates (plotted as grey background)
        figsize: Figure size
        point_size: Point size for query cells
        dpi: DPI for saved figure
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    if ref_coords is not None:
        ax.scatter(
            ref_coords[:, 0],
            ref_coords[:, 1],
            s=0.8,
            alpha=0.1,
            c='lightgray',
            label='Reference',
            rasterized=True
        )

    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=uncertainty,
        s=point_size,
        cmap='viridis',
        alpha=0.6
    )

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Uncertainty (1 - max prob)", fontsize=11)
    plt.axis('tight')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def save_hvg_list(gene_names, output_path):
    """Save HVG gene names to text file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(gene_names))


def create_output_dirs(dirs):
    """Create output directories if they don't exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
