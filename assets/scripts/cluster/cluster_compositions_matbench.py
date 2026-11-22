"""This example script clusters the smallest MatBench datasets
(matbench_steels and matbench_jdft2d) using different embedding and projection methods.
Resulting plots are colored by target property of each dataset.
"""

# /// script
# dependencies = [
#     "matminer>=0.10.1",
#     "pymatgen>=2026.9.24",
#     "umap-learn>=0.5",
# ]
# ///
# ruff: noqa: RUF001

from __future__ import annotations

import gzip
import json
import os
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from matminer.datasets import load_dataset

import pymatviz as pmv
import pymatviz.cluster as pcc
from pymatviz.cluster import EmbeddingMethod as Embed
from pymatviz.cluster import ProjectionMethod as Project


if TYPE_CHECKING:
    from pymatviz.cluster import ClusterScatterPlot, ClusterScatterPlot3D

    # (dataset_name, target_key, target_label, target_symbol, embed_method,
    #  proj_method, n_components)
    type PlotConfig = tuple[str, str, str, str, Embed, Project, int]


module_dir = os.path.dirname(__file__)
plot_dir = f"{module_dir}/tmp/figs/composition_clustering"
cache_dir = f"{module_dir}/tmp/embeddings"
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)


def process_dataset(
    dataset_name: str,
    target_key: str,
    target_label: str,
    target_symbol: str,
    embed_method: pcc.EmbeddingMethod | str,
    projection: pcc.ProjectionMethod,
    n_components: int,
    **kwargs: Any,
) -> ClusterScatterPlot | ClusterScatterPlot3D:
    """Process a single dataset and create clustering visualizations.

    Args:
        dataset_name (str): Name of the MatBench dataset to load
        target_key (str): Name of the target property column
        target_label (str): Display label for the property
        target_symbol (str): Symbol for the property
        embed_method (EmbeddingMethod): Method to convert compositions to vectors
        projection (ProjectionMethod): Method to reduce dimensionality
        n_components (int): Number of dimensions for projection (2 or 3)
        kwargs: Passed to cluster_compositions()

    Returns:
        ClusterScatterPlot | ClusterScatterPlot3D: 2D or 3D cluster widget
    """
    # Load dataset
    df_data = load_dataset(dataset_name)

    # Extract compositions and target values
    if "composition" in df_data:
        compositions = df_data["composition"].tolist()
    else:
        # Extract formula from structure
        compositions = [struct.formula for struct in df_data["structure"]]

    properties = df_data[target_key].tolist()

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

    # recompute when the cache predates the dataset (e.g. new or reformatted entries)
    if embeddings_dict is None or not set(compositions) <= embeddings_dict.keys():
        # Create embeddings
        if embed_method == "one-hot":
            embeddings = pcc.one_hot_encode(compositions)
        elif embed_method == Embed.magpie:
            embeddings = pcc.magpie_featurize(compositions)
        elif embed_method == "matscholar_el":  # matminer preset, featurized here
            from matminer.featurizers.composition import ElementProperty
            from pymatgen.core import Composition

            featurizer = ElementProperty.from_preset("matscholar_el")
            comps = [Composition(comp) for comp in compositions]
            embeddings = np.nan_to_num(featurizer.featurize_many(comps))
        else:
            raise ValueError(f"Unknown {embed_method=}")

        # Convert to dictionary mapping compositions to their embeddings
        embeddings_dict = dict(zip(compositions, embeddings, strict=True))

        # Cache the embeddings
        with gzip.open(cache_file, mode="wt") as file:
            default_handler = lambda x: x.tolist() if hasattr(x, "tolist") else x
            json.dump(embeddings_dict, file, default=default_handler)

    df_plot = pd.DataFrame({"composition": compositions})
    df_plot[target_label] = properties

    # Prepare data for annotation: Find top 5 points by property value
    top_indices = df_plot.nlargest(5, target_label).index

    def annotate_top_points(row: pd.Series) -> dict[str, Any] | None:
        """Label the top 5 points concisely with composition and property value."""
        # row.name is the original DataFrame index
        if row.name not in top_indices:
            return None
        comp_str = row["composition"]
        if len(comp_str) > 20:  # if comp too long, use row ID/ material ID
            comp_str = f"ID: {row.name}"
        text = f"{comp_str.replace(' ', '')} {target_symbol}={row[target_label]:.2f}"
        return dict(text=text, font_size="11px", auto_placement=True)

    embed_col = "embeddings"
    if embed_col not in df_plot:
        df_plot[embed_col] = [embeddings_dict[comp] for comp in compositions]

    return pmv.cluster_compositions(
        df=df_plot,
        composition_col="composition",
        prop_name=target_label,
        embedding_method=embed_col,
        projection=projection,
        n_components=n_components,
        show_chem_sys="shape"
        if len(compositions) < 1000 and n_components == 2
        else None,
        annotate_points=annotate_top_points if n_components == 2 else None,
        # thousands of markers leave no free spot: keep labels apart, not off markers
        **(
            dict(label_placement_config={"weights": {"marker": 1, "overlap": 100}})
            if n_components == 2
            else {}
        ),
        # point labels sit on dense markers: a light halo keeps them readable (the halo
        # variable needs a MatterViz build newer than 0.10.0)
        style="height: 600px; --scatter-point-label-fill: #333; "
        "--scatter-point-label-halo: rgba(255, 255, 255, 0.85)",
        **kwargs,
    )


mb_jdft2d: tuple[str, str, str, str] = (
    "matbench_jdft2d",
    "exfoliation_en",
    "Exfoliation Energy (meV/atom)",
    "E_ex",
)
mb_steels: tuple[str, str, str, str] = (
    "matbench_steels",
    "yield strength",
    "Yield Strength (MPa)",
    "σ",
)
mb_dielectric: tuple[str, str, str, str] = (
    "matbench_dielectric",
    "n",
    "Refractive index",
    "n",
)
mb_perovskites: tuple[str, str, str, str] = (
    "matbench_perovskites",
    "e_form",
    "Formation energy (eV/atom)",
    "E_f",
)
mb_phonons: tuple[str, str, str, str] = (
    "matbench_phonons",
    "last phdos peak",
    "Max Phonon Peak (cm⁻¹)",
    "ν_max",
)
mb_bulk_modulus: tuple[str, str, str, str] = (
    "matbench_log_kvrh",
    "log10(K_VRH)",
    "Bulk Modulus (GPa)",
    "K_VRH",
)
plot_combinations: list[PlotConfig] = [
    # 1. Steels with PCA (2D) - shows clear linear trends
    (*mb_steels, Embed.magpie, Project.pca, 2),
    # 2. Steels with t-SNE (2D) - shows non-linear clustering
    (*mb_steels, Embed.magpie, Project.tsne, 2),
    # TODO umap-learn seemingly not installed by uv run in CI, fix later
    # 3. JDFT2D with UMAP (2D) - shows modern non-linear projection
    # (*mb_jdft2d, Embed.magpie, Project.umap, 2),
    # 4. JDFT2D with one-hot encoding and PCA (3D) - shows raw element relationships
    (*mb_jdft2d, Embed.one_hot, Project.pca, 3),
    # 5. Steels with Matscholar embedding and t-SNE (3D) - shows advanced embedding
    (*mb_steels, "matscholar_el", Project.tsne, 3),
    # 6. Dielectric with PCA (2D) - shows clear linear trends
    (*mb_dielectric, Embed.magpie, Project.pca, 2),
    # 7. Perovskites with PCA (2D) - shows clear linear trends
    (*mb_perovskites, Embed.magpie, Project.pca, 2),
    # 8. Phonons with PCA (2D) - shows clear linear trends
    (*mb_phonons, Embed.magpie, Project.pca, 2),
    # 9. Bulk Modulus with PCA (2D) - shows clear linear trends
    (*mb_bulk_modulus, Embed.magpie, Project.pca, 2),
    # 10. Perovskites with t-SNE (3D) - shows raw element relationships
    (*mb_perovskites, Embed.magpie, Project.tsne, 3),
]

for (
    data_name,
    target_key,
    target_label,
    target_symbol,
    embed_method,
    proj_method,
    n_components,
) in plot_combinations:
    widget = process_dataset(
        dataset_name=data_name,
        target_key=target_key,
        target_label=target_label,
        target_symbol=target_symbol,
        embed_method=embed_method,
        projection=proj_method,
        n_components=n_components,
        log=data_name == "matbench_dielectric",
    )
    img_name = f"{data_name}-{embed_method}-{proj_method}-{n_components}d".replace(
        "_", "-"
    )
    widget.to_html(f"{plot_dir}/{img_name}.html")
    widget.show()
    if data_name == "matbench_perovskites":  # shown in the readme
        # 3D scatter draws to a canvas, which has no vector form
        ext = "png" if n_components == 3 else "svg"
        widget.to_img(
            f"{pmv.utils.ROOT}/assets/{ext}/{img_name}.{ext}",
            width=600,
            transparent=True,
        )
