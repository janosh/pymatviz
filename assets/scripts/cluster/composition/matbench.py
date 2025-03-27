"""Example script demonstrating composition clustering visualization across MatBench
datasets.

This script loads several MatBench datasets and creates 2D projections of their
compositions using both one-hot and Magpie embeddings, with both PCA and t-SNE
dimensionality reduction. The resulting plots are colored by the target property of
each dataset.

For efficiency, embeddings are cached in ./tmp/embeddings and reused if available.
"""

from __future__ import annotations

import gzip
import json
import os
from typing import TYPE_CHECKING, get_args

import numpy as np
import pandas as pd
from matminer.datasets import load_dataset
from tqdm import tqdm

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
cache_dir = f"{module_dir}/tmp/embeddings"
plot_dir = f"{module_dir}/tmp/figs/composition_clustering"

# Create directories
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)


def process_dataset(
    dataset_name: str,
    target_col: str,
    title: str,
    embedding_method: EmbeddingMethod,
    projection_method: ProjectionMethod,
) -> go.Figure:
    """Process a single dataset and create clustering visualizations.

    Args:
        dataset_name (str): Name of the MatBench dataset to load
        target_col (str): Name of the target property column
        title (str): Display title for the property
        embedding_method (EmbeddingMethod): Method to convert compositions to vectors
        projection_method (ProjectionMethod): Method to reduce dimensionality

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
    cache_file = f"{cache_dir}/{dataset_name}_{embedding_method}.json.gz"
    embeddings_dict = None
    if os.path.isfile(cache_file):
        with gzip.open(cache_file, mode="rt") as file:
            cache = json.load(file)
            # Verify cache matches current compositions
            if set(cache["compositions"]) >= set(compositions):
                # Convert list of embeddings back to numpy arrays
                # Only keep compositions that are in our filtered dataset
                embeddings_dict = {
                    comp: np.array(emb)
                    for comp, emb in zip(
                        cache["compositions"], cache["embeddings"], strict=False
                    )
                    if comp in compositions
                }

    if embeddings_dict is None:
        # Create embeddings
        if embedding_method == "one-hot":
            embeddings = one_hot_encode(compositions)
        elif embedding_method in get_args(EmbeddingMethod):
            embeddings = matminer_featurize(compositions, preset=embedding_method)
        else:
            raise ValueError(f"Unknown {embedding_method=}")

        # Convert to dictionary mapping compositions to their embeddings
        embeddings_dict = dict(zip(compositions, embeddings, strict=False))

        # Cache the embeddings
        with gzip.open(cache_file, mode="wt") as file:
            data = {
                "compositions": compositions,
                "embeddings": [emb.tolist() for emb in embeddings_dict.values()],
            }
            json.dump(data, file)

    # Create plot with pre-computed embeddings
    fig = pmv.cluster_compositions(
        compositions=embeddings_dict,
        properties=dict(zip(compositions, properties, strict=False)),
        prop_name=title,
        projection_method=projection_method,
        n_components=2,
        point_size=8,
        opacity=0.8,
    )

    title = f"{dataset_name} - {embedding_method} + {projection_method}"
    fig.layout.update(title=dict(text=title, x=0.5), margin_t=50)
    img_path = f"{plot_dir}/{dataset_name}_{embedding_method}_{projection_method}.html"
    fig.write_html(img_path, include_plotlyjs="cdn")
    return fig


datasets = {
    "matbench_phonons": {
        "target": "last phdos peak",
        "title": "Max Phonon Frequency (cm⁻¹)",
    },
    "matbench_perovskites": {
        "target": "e_form",
        "title": "Formation Energy (eV)",
    },
    "matbench_log_kvrh": {
        "target": "log10(K_VRH)",
        "title": "Bulk Modulus (log₁₀ GPa)",
    },
}
embedding_methods = ["magpie", "matscholar_el", "one-hot"]
projection_methods = ["pca", "tsne", "umap", "isomap", "kernel_pca"]

combinations = [
    (dataset_name, info["target"], info["title"], embed_method, proj_method)
    for dataset_name, info in datasets.items()
    for embed_method in embedding_methods
    for proj_method in projection_methods
]

pbar = tqdm(combinations)
for dataset_name, target_col, title, embed_method, proj_method in pbar:
    pbar.set_description(f"{dataset_name} with {embed_method} + {proj_method}")
    process_dataset(
        dataset_name=dataset_name,
        target_col=target_col,
        title=title,
        embedding_method=embed_method,
        projection_method=proj_method,
    )
