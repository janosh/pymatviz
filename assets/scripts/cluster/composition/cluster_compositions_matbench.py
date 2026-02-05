"""This example script clusters the smallest MatBench datasets
(matbench_steels and matbench_jdft2d) using different embedding and projection methods.
Resulting plots are colored by target property of each dataset.
"""

# /// script
# dependencies = [
#     "matminer>=0.9.1",
#     "umap-learn>=0.5",
# ]
# ///
# ruff: noqa: RUF001

from __future__ import annotations

import gzip
import json
import os
from typing import TYPE_CHECKING, Any

import pandas as pd
from matminer.datasets import load_dataset
from pymatgen.core import Composition
from pymatgen.util.string import htmlify

import pymatviz as pmv
import pymatviz.cluster.composition as pcc
from pymatviz.cluster.composition import EmbeddingMethod as Embed
from pymatviz.cluster.composition import ProjectionMethod as Project
from pymatviz.enums import Key


if TYPE_CHECKING:
    from typing import TypeAlias

    import plotly.graph_objects as go

    # (dataset_name, target_key, target_label, target_symbol, embed_method,
    #  proj_method, n_components, colorbar_kwargs)
    PlotConfig: TypeAlias = tuple[
        str, str, str, str, Embed, Project, int, dict[str, Any]
    ]


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
    target_key: str,
    target_label: str,
    target_symbol: str,
    embed_method: pcc.EmbeddingMethod,
    projection: pcc.ProjectionMethod,
    n_components: int,
    **kwargs: Any,
) -> go.Figure:
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

    if embeddings_dict is None:
        # Create embeddings
        if embed_method == "one-hot":
            embeddings = pcc.one_hot_encode(compositions)
        elif embed_method in (Embed.magpie, Embed.matscholar_el):
            embeddings = pcc.matminer_featurize(compositions, preset=embed_method.value)  # type: ignore[arg-type]
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
        """Annotate top 5 points concisely with composition and property value."""
        # row.name is the original DataFrame index
        if row.name not in top_indices:
            return None
        # Use raw composition string for brevity
        comp_str = row["composition"]
        if len(comp_str) > 20:  # if comp too long, use row ID/ material ID
            comp_str = f"ID: {row.name}"
        else:
            comp_str = htmlify(comp_str).replace(" ", "")
        prop_val = f"{target_symbol}={row[target_label]:.2f}"
        text = f"{comp_str}<br>{prop_val}"
        return dict(text=text, font_size=11, bgcolor="rgba(240, 240, 240, 0.5)")

    embed_col = "embeddings"
    if embed_col not in df_plot:
        df_plot[embed_col] = [embeddings_dict.get(comp) for comp in compositions]

    fig = pmv.cluster_compositions(
        df_in=df_plot,
        composition_col="composition",
        prop_name=target_label,
        embedding_method=embed_col,
        projection=projection,
        n_components=n_components,
        marker_size=8,
        opacity=0.8,
        width=1000,
        height=600,
        show_chem_sys="shape" if len(compositions) < 1000 else None,
        annotate_points=annotate_top_points,
        **kwargs,
    )

    # Update title and margins
    title = f"{dataset_name} - {embed_method} + {projection} ({n_components}D)"
    fig.layout.update(title=dict(text=title, x=0.5), margin_t=50)
    # format compositions and coordinates in hover tooltip
    custom_data = [
        [format_composition(comp) if dataset_name == "matbench_steels" else comp]
        for comp in compositions
    ]
    fig.update_traces(
        hovertemplate=(
            "%{customdata[0]}<br>"  # Formatted composition
            f"{projection} 1: %{{x:.2f}}<br>"  # First projection coordinate
            f"{projection} 2: %{{y:.2f}}<br>"  # Second projection coordinate
            + (f"{projection} 3: %{{z:.2f}}<br>" if n_components == 3 else "")
            + f"{target_label}: %{{marker.color:.2f}}"  # Property value
        ),
        customdata=custom_data,
    )

    return fig


mb_jdft2d: tuple[str, str, str, str] = (
    "matbench_jdft2d",
    "exfoliation_en",
    "Exfoliation Energy (meV/atom)",
    "E<sub>ex</sub>",
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
    "E<sub>f</sub>",
)
mb_phonons: tuple[str, str, str, str] = (
    "matbench_phonons",
    "last phdos peak",
    "Max Phonon Peak (cm⁻¹)",
    "ν<sub>max</sub>",
)
mb_bulk_modulus: tuple[str, str, str, str] = (
    "matbench_log_kvrh",
    "log10(K_VRH)",
    "Bulk Modulus (GPa)",
    "K<sub>VRH</sub>",
)
plot_combinations: list[PlotConfig] = [  # ty: ignore[invalid-assignment]
    # 1. Steels with PCA (2D) - shows clear linear trends
    (*mb_steels, Embed.magpie, Project.pca, 2, {"x": 0.01, "xanchor": "left"}),
    # 2. Steels with t-SNE (2D) - shows non-linear clustering
    (*mb_steels, Embed.magpie, Project.tsne, 2, {"x": 0.01, "xanchor": "left"}),
    # TODO umap-learn seemingly not installed by uv run in CI, fix later
    # 3. JDFT2D with UMAP (2D) - shows modern non-linear projection
    # (*mb_jdft2d, Embed.magpie, Project.umap, 2, {"x": 0.01, "xanchor": "left"}),
    # 4. JDFT2D with one-hot encoding and PCA (3D) - shows raw element relationships
    (*mb_jdft2d, Embed.one_hot, Project.pca, 3, {}),
    # 5. Steels with Matscholar embedding and t-SNE (3D) - shows advanced embedding
    (*mb_steels, Embed.matscholar_el, Project.tsne, 3, {"x": 0.5, "y": 0.8}),
    # 6. Dielectric with PCA (2D) - shows clear linear trends
    (*mb_dielectric, Embed.magpie, Project.pca, 2, {"x": 0.01, "xanchor": "left"}),
    # 7. Perovskites with PCA (2D) - shows clear linear trends
    (*mb_perovskites, Embed.magpie, Project.pca, 2, {"x": 0.01, "xanchor": "left"}),
    # 8. Phonons with PCA (2D) - shows clear linear trends
    (*mb_phonons, Embed.magpie, Project.pca, 2, {"x": 0.01, "xanchor": "left"}),
    # 9. Bulk Modulus with PCA (2D) - shows clear linear trends
    (
        *mb_bulk_modulus,
        Embed.magpie,
        Project.pca,
        2,
        {"x": 0.99, "y": 0.96, "yanchor": "top"},
    ),
    # 10. Perovskites with t-SNE (3D) - shows raw element relationships
    (*mb_perovskites, Embed.magpie, Project.tsne, 3, {}),
]

for (
    data_name,
    target_key,
    target_label,
    target_symbol,
    embed_method,
    proj_method,
    n_components,
    cbar_args,
) in plot_combinations:
    fig = process_dataset(
        dataset_name=data_name,
        target_key=target_key,
        target_label=target_label,
        target_symbol=target_symbol,
        embed_method=embed_method,
        projection=proj_method,
        n_components=n_components,
        color_scale="log" if data_name == "matbench_dielectric" else "linear",
    )
    fig.layout.coloraxis.colorbar.update(**cbar_args)

    # Save as HTML and SVG
    img_name = f"{data_name}-{embed_method}-{proj_method}-{n_components}d".replace(
        "_", "-"
    )
    fig.write_html(f"{plot_dir}/{img_name}.html", include_plotlyjs="cdn")
    pmv.io.save_and_compress_svg(fig, img_name)

    fig.show()
