"""This example script clusters the smallest MatBench datasets
(matbench_steels and matbench_jdft2d) using different embedding and projection methods.
Resulting plots are colored by target property of each dataset.
"""

from __future__ import annotations

import gzip
import json
import os
from typing import TYPE_CHECKING, Any

import pandas as pd
from matminer.datasets import load_dataset
from pymatgen.core import Composition

import pymatviz as pmv
from pymatviz.cluster.composition import (
    EmbeddingMethod,
    matminer_featurize,
    one_hot_encode,
)
from pymatviz.enums import Key


if TYPE_CHECKING:
    import plotly.graph_objects as go

    from pymatviz.cluster.composition import ProjectionMethod


pmv.set_plotly_template("pymatviz_white")
module_dir = os.path.dirname(__file__)
plot_dir = f"{module_dir}/tmp/figs/composition_clustering"
cache_dir = f"{module_dir}/tmp/embeddings"
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)


def format_composition(formula: str) -> str:
    """Format long steel compositions into 2-column layout, sorted by amount."""
    comp = Composition(formula)
    # Sort elements by amount in descending order
    element_pairs = []
    for idx, (elem, amt) in enumerate(
        sorted(comp.items(), key=lambda x: x[1], reverse=True)
    ):
        suffix = "<br>" if idx % 2 == 1 else ""
        element_pairs.append(f"{elem}: {amt:.4}{suffix}")
    return "\t\t".join(element_pairs).replace("<br>\t\t", "<br>")


def process_dataset(
    dataset_name: str,
    target_col: str,
    target_label: str,
    embed_method: EmbeddingMethod,
    projection_method: ProjectionMethod,
    n_components: int,
) -> go.Figure:
    """Process a single dataset and create clustering visualizations.

    Args:
        dataset_name (str): Name of the MatBench dataset to load
        target_col (str): Name of the target property column
        target_label (str): Display label for the property
        embed_method (EmbeddingMethod): Method to convert compositions to vectors
        projection_method (ProjectionMethod): Method to reduce dimensionality
        n_components (int): Number of dimensions for projection (2 or 3)

    Returns:
        fig: Plotly figure
    """
    # Load dataset
    df_data = load_dataset(dataset_name)

    # Extract compositions and target values
    if Key.composition in df_data:
        compositions = df_data[Key.composition].tolist()
    else:
        # Extract formula from structure
        compositions = [struct.formula for struct in df_data[Key.structure]]

    properties = df_data[target_col].tolist()

    # Create a DataFrame to align compositions and properties
    df_with_prop = pd.DataFrame(
        {"composition": compositions, "property": properties}
    ).dropna()
    compositions = df_with_prop["composition"].tolist()
    properties = df_with_prop["property"].tolist()

    # Try to load cached embeddings
    cache_file = f"{cache_dir}/{dataset_name}_{embed_method}.json.gz"
    embeddings_dict = None
    if os.path.isfile(cache_file):
        with gzip.open(cache_file, mode="rt") as file:
            embeddings_dict = json.load(file)

    if embeddings_dict is None:
        # Create embeddings
        if embed_method == "one-hot":
            embeddings = one_hot_encode(compositions)
        elif embed_method in ["magpie", "matscholar_el"]:
            embeddings = matminer_featurize(compositions, preset=embed_method)
        else:
            raise ValueError(f"Unknown {embed_method=}")

        # Convert to dictionary mapping compositions to their embeddings
        embeddings_dict = dict(zip(compositions, embeddings, strict=True))

        # Cache the embeddings
        with gzip.open(cache_file, mode="wt") as file:
            default_handler = lambda x: x.tolist() if hasattr(x, "tolist") else x
            json.dump(embeddings_dict, file, default=default_handler)

    # Create plot with pre-computed embeddings
    fig = pmv.cluster_compositions(
        compositions=embeddings_dict,
        properties=dict(zip(compositions, properties, strict=True)),
        prop_name=target_label,
        projection_method=projection_method,
        n_components=n_components,
        marker_size=8,
        opacity=0.8,
        width=1000,
        height=600,
        show_chem_sys="shape" if len(compositions) < 1000 else None,
    )

    # Update title and margins
    title = f"{dataset_name} - {embed_method} + {projection_method} ({n_components}D)"
    fig.layout.update(title=dict(text=title, x=0.5), margin_t=50)
    # format compositions and coordinates in hover tooltip
    custom_data = [
        [format_composition(comp) if dataset_name == "matbench_steels" else comp]
        for comp in compositions
    ]
    fig.update_traces(
        hovertemplate=(
            "%{customdata[0]}<br>"  # Formatted composition
            f"{projection_method} 1: %{{x:.2f}}<br>"  # First projection coordinate
            f"{projection_method} 2: %{{y:.2f}}<br>"  # Second projection coordinate
            + (f"{projection_method} 3: %{{z:.2f}}<br>" if n_components == 3 else "")
            + f"{target_label}: %{{marker.color:.2f}}"  # Property value
        ),
        customdata=custom_data,
    )

    return fig


mb_jdft2d = ("matbench_jdft2d", "exfoliation_en", "Exfoliation Energy (meV/atom)")
mb_steels = ("matbench_steels", "yield strength", "Yield Strength (MPa)")
mb_dielectric = ("matbench_dielectric", "n", "Refractive index")
mb_perovskites = ("matbench_perovskites", "e_form", "Formation energy (eV/atom)")
mb_phonons = ("matbench_phonons", "last phdos peak", "Max Phonon Peak (cm⁻¹)")
mb_bulk_modulus = ("matbench_log_kvrh", "log10(K_VRH)", "Bulk Modulus (GPa)")
plot_combinations: list[
    tuple[str, str, str, EmbeddingMethod, ProjectionMethod, int, dict[str, Any]]
] = [
    # 1. Steels with PCA (2D) - shows clear linear trends
    (*mb_steels, "magpie", "pca", 2, dict(x=0.01, xanchor="left")),
    # 2. Steels with t-SNE (2D) - shows non-linear clustering
    (*mb_steels, "magpie", "tsne", 2, dict(x=0.01, xanchor="left")),
    # 3. JDFT2D with UMAP (2D) - shows modern non-linear projection
    (*mb_jdft2d, "magpie", "umap", 2, dict(x=0.01, xanchor="left")),
    # 4. JDFT2D with one-hot encoding and PCA (3D) - shows raw element relationships
    (*mb_jdft2d, "one-hot", "pca", 3, dict()),
    # 5. Steels with Matscholar embedding and t-SNE (3D) - shows advanced embedding
    (*mb_steels, "matscholar_el", "tsne", 3, dict(x=0.5, y=0.8)),
    # 6. Dielectric with PCA (2D) - shows clear linear trends
    (*mb_dielectric, "magpie", "pca", 2, dict(x=0.01, xanchor="left")),
    # 7. Perovskites with PCA (2D) - shows clear linear trends
    (*mb_perovskites, "magpie", "pca", 2, dict(x=0.01, xanchor="left")),
    # 8. Phonons with PCA (2D) - shows clear linear trends
    (*mb_phonons, "magpie", "pca", 2, dict(x=0.01, xanchor="left")),
    # 9. Bulk Modulus with PCA (2D) - shows clear linear trends
    (*mb_bulk_modulus, "magpie", "pca", 2, dict(x=0.99, y=0.96, yanchor="top")),
]

for (
    data_name,
    target_col,
    target_label,
    embed_method,
    proj_method,
    n_components,
    cbar_args,
) in plot_combinations:
    fig = process_dataset(
        dataset_name=data_name,
        target_col=target_col,
        target_label=target_label,
        embed_method=embed_method,
        projection_method=proj_method,
        n_components=n_components,
    )
    fig.update_layout(coloraxis_colorbar=cbar_args)

    # Save as HTML and SVG
    output_name = f"{data_name}_{embed_method}_{proj_method}_{n_components}d"
    fig.write_html(f"{plot_dir}/{output_name}.html", include_plotlyjs="cdn")
    pmv.io.save_and_compress_svg(fig, output_name)

    fig.show()
