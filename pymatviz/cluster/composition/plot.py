"""Plotting functions for chemical composition clustering."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, get_args

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator
from plotly.validators.scatter3d.marker import SymbolValidator as Symbol3dValidator
from pymatgen.core import Composition

from pymatviz.cluster.composition.embed import matminer_featurize, one_hot_encode
from pymatviz.cluster.composition.project import project_vectors


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pymatviz.typing import ColorType

EmbeddingMethod = Literal[
    "one-hot", "magpie", "deml", "matminer", "matscholar_el", "megnet_el"
]
ProjectionMethod = Literal["pca", "tsne", "umap", "isomap", "kernel_pca"]
ShowChemSys = Literal["color", "shape", "color+shape"]


def cluster_compositions(
    compositions: Sequence[str]
    | Sequence[Composition]
    | pd.Series
    | dict[str, np.ndarray],
    *,
    properties: Sequence[float]
    | Sequence[int]
    | dict[str, float]
    | pd.Series
    | None = None,
    prop_name: str | None = None,
    embedding_method: EmbeddingMethod
    | Callable[[Sequence[str], Any], np.ndarray] = "magpie",
    projection_method: ProjectionMethod
    | Callable[[np.ndarray, int, Any], np.ndarray] = "pca",
    n_components: int = 2,
    hover_format: str = ".2f",
    heatmap_colorscale: str = "Viridis",
    marker_size: int = 8,
    opacity: float = 0.8,
    show_chem_sys: ShowChemSys | None = None,
    color_discrete_map: dict[str, ColorType] | None = None,
    embedding_kwargs: dict[str, Any] | None = None,
    projection_kwargs: dict[str, Any] | None = None,
    title: str | None = None,
    width: int | None = None,
    height: int | None = None,
    sort: bool | int | Callable[[np.ndarray], np.ndarray] = True,
    show_projection_stats: bool | dict[str, Any] = True,
) -> go.Figure:
    """Plot chemical composition clusters with optional property coloring.

    Gives a 2D or 3D scatter plot of chemical compositions, using various
    embedding and dimensionality reduction techniques to visualize the relationships
    between different materials.

    Args:
        compositions (list[str] | list[Composition] | pd.Series | dict[str, np.array]):
            Chemical formulas as strings or pymatgen Composition objects.
            Can also be a dictionary mapping composition strings to pre-computed
            embedding vectors.
        properties (list[float] | list[int] | dict[str, float] | pd.Series | None):
            Optional list, dict, or pandas Series of property values to use for
            coloring. If a dict or Series, keys/index should match the composition
            strings. (default: None)
        prop_name (str | None): Name of the property for axis labels and colorbar.
            Required if properties is provided. (default: None)
        embedding_method (EmbeddingMethod | Callable[[Sequence[str], Any], np.ndarray]):
            Method to convert compositions to vectors (default: "magpie")
            - "one-hot": One-hot encoding of element fractions
            - "magpie": Matminer's MagPie featurization (for backward compatibility)
            - "deml": Matminer's DEML featurization
            - "matminer": Matminer's ElementProperty featurization
            - "matscholar_el": Matminer's Matscholar Element featurization
            - "megnet_el": Matminer's MEGNet Element featurization
            - Callable: Custom embedding function that takes (N, ) array of compositions
              and returns (N, D) array of values where D is the number of dimensions in
              the embedding space.
            Ignored if compositions is a dictionary mapping to pre-computed embeddings.
        projection_method: Method to reduce embedding dimensionality (default: "pca")
            - "pca": Principal Component Analysis (linear)
            - "tsne": t-distributed Stochastic Neighbor Embedding (non-linear)
            - "umap": Uniform Manifold Approximation and Projection (non-linear)
            - "isomap": Isometric Feature Mapping (non-linear)
            - "kernel_pca": Kernel Principal Component Analysis (non-linear)
            - Callable: Custom projection function that takes (N, D) array of values and
              returns (N, n_components) projected array where n_components is 2 or 3 and
              D is the number of dimensions in the embedding space.
        n_components (int): Projection dimensions (2 or 3) (default: 2)
        hover_format (str): Format string for hover data (default: ".2f")
        heatmap_colorscale (str): Colorscale for continuous property values
            (default: "Viridis")
        marker_size (int): Size of the scatter plot points (default: 8)
        opacity (float): Opacity of the scatter plot points (default: 0.8)
        show_chem_sys (ShowChemSys | None): How to visualize chemical systems:
            - "color": Color points by chemical system (if no properties)
            - None: Don't use chemical system visualization
            - "shape": Use different marker shapes for different chemical systems
              (works best with ≤10 different chemical systems)
            - "color+shape": Use both colors and shapes to distinguish chemical systems
              (works best with ≤10 different chemical systems)
            (default: "color")
        color_discrete_map (dict[str, ColorType] | None): Optional mapping of chemical
            systems to colors (default: None)
        embedding_kwargs (dict[str, Any] | None): Additional keyword arguments for the
            embedding function (default: None)
        projection_kwargs (dict[str, Any] | None): Additional keyword arguments for the
            projection function (default: None)
        title (str | None): Optional plot title (default: None)
        width (int | None): Optional plot width in pixels (default: None)
        height (int | None): Optional plot height in pixels (default: None)
        sort (bool | int | Callable[[np.ndarray], np.ndarray]): Controls point sorting
            before plotting (default: True)
            - True or 1: Sort by prop values in ascending order (highest points
              plotted last)
            - False or 0: No sorting
            - -1: Sort by prop values in descending order (highest points plotted first)
            - Callable: Custom sorting function. Takes an array of values and returns
              sorted indices
        show_projection_stats (bool | dict[str, Any]): Whether to show statistics about
            the projection (default: True)
            - True: Show default stats
            - False: Don't show stats
            - dict: Customize stats appearance with plotly annotation parameters
            For PCA, shows variance explained by each component
            For t-SNE, shows perplexity and learning rate
            For UMAP, shows n_neighbors and min_dist
            For Isomap, shows n_neighbors and metric
            For Kernel PCA, shows kernel type and parameters

    Returns:
        go.Figure: Plotly figure object
    """
    if n_components not in (2, 3):  # Validate inputs
        raise ValueError(f"{n_components=} must be 2 or 3")

    if properties is not None and prop_name is None:
        raise ValueError(
            f"{prop_name=} must be provided when {properties=} is not None"
        )

    if not isinstance(show_projection_stats, (bool, dict)):
        raise TypeError(f"{show_projection_stats=} must be bool or dict")

    projection_kwargs = projection_kwargs or {}

    # Handle pre-computed embeddings
    if isinstance(compositions, dict):
        comp_strs = list(compositions)
        embeddings = np.stack(list(compositions.values()))
    else:
        # Convert compositions to strings for consistent handling
        comp_strs = []
        for comp in compositions:
            if isinstance(comp, str):
                comp_strs.append(comp)
            elif isinstance(comp, Composition):
                comp_strs.append(comp.formula)
            else:
                raise TypeError(f"Expected str or Composition, got {comp=}")

        # Create embeddings
        if callable(embedding_method):
            # Use custom embedding function
            embeddings = embedding_method(compositions, **(embedding_kwargs or {}))  # type: ignore[call-arg]
        # Use built-in embedding methods
        elif embedding_method == "one-hot":
            embeddings = one_hot_encode(compositions, **(embedding_kwargs or {}))
        elif embedding_method in get_args(EmbeddingMethod):
            embeddings = matminer_featurize(
                compositions, preset=embedding_method, **(embedding_kwargs or {})
            )
        else:
            raise ValueError(
                f"{embedding_method=} must be in {get_args(EmbeddingMethod)}"
            )

    # Project to lower dimensions
    if callable(projection_method):
        # Use custom projection function
        projected = projection_method(
            embeddings,
            n_components=n_components,
            **projection_kwargs,  # type: ignore[call-arg]
        )
        pca_obj = None
    elif projection_method in get_args(ProjectionMethod):
        # Use built-in projection methods
        projected, pca_obj = project_vectors(
            embeddings,
            method=projection_method,
            n_components=n_components,
            **projection_kwargs,
        )
    else:
        raise ValueError(
            f"{projection_method=} must be in {get_args(ProjectionMethod)}"
        )

    # Extract chemical systems if needed
    chem_systems = None
    uniq_chem_sys = set()
    valid_symbols = []

    if show_chem_sys in ("shape", "color", "color+shape"):
        chem_systems = []
        symbol_filter = (
            lambda x: isinstance(x, str)
            and not x.isdigit()
            and not x.endswith("-dot")
            and "-open" not in x
        )
        for comp_str in comp_strs:
            if isinstance(comp_str, str):  # Extract element symbols
                comp = Composition(comp_str)
                chem_systems.append("-".join(sorted(comp.chemical_system.split("-"))))
            else:
                raise TypeError(f"Expected str, got {type(comp_str)} for {comp_str}")

        uniq_chem_sys = set(chem_systems)

        # Get valid symbols for the current plot type
        if n_components == 3:
            # For 3D plots, we need to use a more limited set of markers
            all_symbols = Symbol3dValidator("symbol", "scatter3d.marker").values  # noqa: PD011
            valid_symbols = list(filter(symbol_filter, all_symbols))
        else:
            # For 2D plots, we can use the full set of markers
            all_symbols = SymbolValidator("symbol", "scatter.marker").values  # noqa: PD011
            valid_symbols = list(filter(symbol_filter, all_symbols))

        # Check if we have more unique systems than available symbols
        if "shape" in show_chem_sys and len(uniq_chem_sys) > len(valid_symbols):
            warnings.warn(
                f"Number of unique chemical systems ({len(uniq_chem_sys)}) exceeds "
                f"available marker symbols ({len(valid_symbols)}). Some systems will "
                "use duplicate symbols. Recommended to set "
                "show_chem_sys='color+shape' | False.",
                UserWarning,
                stacklevel=2,
            )

    # Create a DataFrame for plotting
    df_plot = pd.DataFrame()
    df_plot["composition"] = comp_strs

    # Add projections
    if n_components == 2:
        df_plot["PC1"] = projected[:, 0]
        df_plot["PC2"] = projected[:, 1]
    else:  # n_components == 3
        df_plot["PC1"] = projected[:, 0]
        df_plot["PC2"] = projected[:, 1]
        df_plot["PC3"] = projected[:, 2]

    # Add chemical systems if needed
    if chem_systems is not None:
        df_plot["chem_system"] = chem_systems

    # Add properties or configure chemical system coloring
    prop_values: list[float] | None = None
    if properties is not None:
        if isinstance(properties, dict):
            # Ensure all compositions are in the dict
            if not all(comp in properties for comp in comp_strs):
                missing = [comp for comp in comp_strs if comp not in properties]
                raise ValueError(
                    f"Missing property values for compositions: {missing[:5]}..."
                )

            prop_values = [properties[comp] for comp in comp_strs]

        elif isinstance(properties, pd.Series):
            # Try to match by index
            try:
                prop_values = [properties.get(comp, np.nan) for comp in comp_strs]
            except TypeError:
                # Try converting index to string if direct lookup fails
                properties.index = properties.index.astype(str)
                prop_values = [properties.get(comp, np.nan) for comp in comp_strs]

        else:  # Assume sequence
            if len(properties) != len(comp_strs):
                raise ValueError(f"{len(properties)=} must match {len(comp_strs)=}")
            prop_values = list(properties)

        df_plot[prop_name] = prop_values
        color_column = prop_name
    elif chem_systems is not None and show_chem_sys in ("color", "color+shape"):
        # Only color by chem_system for "color" or "color+shape" modes
        color_column = "chem_system"
    else:
        color_column = None

    # Apply sorting if requested
    if sort and prop_values is not None:
        sort_direction = 1  # Default to ascending order
        if isinstance(sort, bool):
            sort_direction = 1 if sort else 0
        elif isinstance(sort, int):
            sort_direction = sort
        elif callable(sort):
            # Custom sort function
            sort_indices = sort(prop_values)
            df_plot = df_plot.iloc[sort_indices]
            projected = projected[sort_indices]
            embeddings = embeddings[sort_indices]
        else:
            raise TypeError(f"Invalid sort parameter type: {type(sort).__name__}")

        if not callable(sort):
            # Sort by property values
            sort_indices = np.argsort(prop_values)
            if sort_direction < 0:
                sort_indices = sort_indices[::-1]  # Reverse for descending order
            df_plot = df_plot.iloc[sort_indices]
            projected = projected[sort_indices]
            embeddings = embeddings[sort_indices]

    # Create hover text template
    hover_template: list[str] = []
    for idx, comp in enumerate(comp_strs):
        hover_text = f"Composition: {comp}<br>"

        # Add projected coordinates
        coord_fmt = (
            f"{{:.{hover_format[1:]}}}"
            if hover_format.startswith(".")
            else hover_format
        )
        hover_text += f"PC1: {coord_fmt.format(projected[idx, 0])}<br>"
        hover_text += f"PC2: {coord_fmt.format(projected[idx, 1])}<br>"
        if n_components == 3:
            hover_text += f"PC3: {coord_fmt.format(projected[idx, 2])}<br>"

        # Add property or chemical system
        if prop_values is not None:
            prop_fmt = (
                f"{{:.{hover_format[1:]}}}"
                if hover_format.startswith(".")
                else hover_format
            )
            hover_text += f"{prop_name}: {prop_fmt.format(prop_values[idx])}"
        elif chem_systems is not None:
            hover_text += f"Chemical System: {chem_systems[idx]}"

        hover_template.append(hover_text)

    df_plot["hover_text"] = hover_template

    # Calculate projection statistics
    projection_stats: str | None = None
    if show_projection_stats:
        if projection_method == "pca" and pca_obj is not None:
            # Get explained variance ratios from PCA object
            var_explained_ratio = pca_obj.explained_variance_ratio_[:n_components]
            cum_var_explained = np.cumsum(var_explained_ratio)

            # Create variance stats text
            stats_text: list[str] = []
            for idx, (var, cum_var) in enumerate(
                zip(var_explained_ratio, cum_var_explained, strict=False)
            ):
                stats_text.append(f"PC{idx + 1}: {var:.1%} (cumulative: {cum_var:.1%})")
            projection_stats = "<br>".join(stats_text)
        elif projection_method == "tsne":
            # For t-SNE, show perplexity and learning rate
            perplexity = projection_kwargs.get("perplexity", 30)
            learning_rate = projection_kwargs.get("learning_rate", 200)
            stats_text = [
                f"Perplexity: {perplexity}",
                f"Learning rate: {learning_rate}",
            ]
            projection_stats = "<br>".join(stats_text)
        elif projection_method == "umap":
            # For UMAP, show n_neighbors and min_dist
            n_neighbors = projection_kwargs.get("n_neighbors", 15)
            min_dist = projection_kwargs.get("min_dist", 0.1)
            stats_text = [
                f"n_neighbors: {n_neighbors}",
                f"min_dist: {min_dist}",
            ]
            projection_stats = "<br>".join(stats_text)
        elif projection_method == "isomap":
            # For Isomap, show n_neighbors and metric
            n_neighbors = projection_kwargs.get("n_neighbors", 15)
            metric = projection_kwargs.get("metric", "euclidean")
            stats_text = [
                f"n_neighbors: {n_neighbors}",
                f"metric: {metric}",
            ]
            projection_stats = "<br>".join(stats_text)
        elif projection_method == "kernel_pca":
            # For Kernel PCA, show kernel type and parameters
            kernel = projection_kwargs.get("kernel", "linear")
            gamma = projection_kwargs.get("gamma", 1.0)
            stats_text = [
                f"kernel: {kernel}",
                f"gamma: {gamma}",
            ]
            projection_stats = "<br>".join(stats_text)

    # Create the plot
    plot_func = px.scatter if n_components == 2 else px.scatter_3d
    plot_kwargs: dict[str, Any] = {
        "x": "PC1",
        "y": "PC2",
        "custom_data": ["composition", "hover_text"],
        "title": title,
        "opacity": opacity,
        "width": width,
        "height": height,
    }

    if n_components == 3:
        plot_kwargs["z"] = "PC3"

    # Define helper functions to reduce redundancy
    def create_shape_dataframe() -> pd.DataFrame:
        """Create a dataframe for shape-based visualization."""
        df_shape = pd.DataFrame()
        df_shape["composition"] = df_plot["composition"]
        df_shape["PC1"] = df_plot["PC1"]
        df_shape["PC2"] = df_plot["PC2"]
        if n_components == 3:
            df_shape["PC3"] = df_plot["PC3"]
        df_shape["hover_text"] = df_plot["hover_text"]
        if chem_systems is not None:
            df_shape["chem_system"] = chem_systems
        # Add properties for coloring if available
        if prop_values is not None:
            df_shape[color_column] = prop_values
        return df_shape

    def get_symbol_mapping() -> dict[str, str]:
        """Create a mapping of chemical systems to symbols."""
        return {
            system: valid_symbols[idx % len(valid_symbols)]
            for idx, system in enumerate(sorted(uniq_chem_sys))
        }

    def apply_symbol_mapping(fig: go.Figure, symbol_map: dict[str, str]) -> None:
        """Apply symbol mapping to the figure based on the visualization mode."""
        if prop_values is not None or show_chem_sys == "shape":
            # For property-colored plots or shape mode, we have one trace with
            # different symbols
            symbols = [symbol_map[cs] for cs in chem_systems]
            fig.data[0].marker.symbol = symbols
        elif show_chem_sys == "color":  # For chemically colored plots without
            # properties, we have one trace per chemical system
            for trace in fig.data:
                if hasattr(trace, "name") and trace.name in symbol_map:
                    fig.update_traces(
                        selector=dict(name=trace.name),
                        marker=dict(symbol=symbol_map[trace.name]),
                    )

    def configure_colorbar(fig: go.Figure) -> None:
        """Configure the colorbar for property-colored plots."""
        if prop_values is not None:
            color_bar = dict(
                orientation="h",
                yanchor="bottom",
                y=0,
                xanchor="right",
                x=0.99,
                thickness=12,
                len=350,
                lenmode="pixels",
                title_side="top",
            )
            fig.update_layout(coloraxis_colorbar=color_bar)
            # Ensure colorbar has a title
            if fig.layout.coloraxis.colorbar is not None:
                fig.layout.coloraxis.colorbar.title = {"text": prop_name}

    # Set up color mapping
    def configure_color_options(kwargs: dict[str, Any]) -> None:
        """Configure color options based on properties and chemical systems."""
        if prop_values is not None:
            kwargs.update(color=color_column, color_continuous_scale=heatmap_colorscale)
        elif color_column is not None:
            kwargs.update(color=color_column, color_discrete_map=color_discrete_map)

    # Handle different visualization modes
    if show_chem_sys == "shape":
        # Create dataframe for shape-based visualization
        df_shape = create_shape_dataframe()

        # Set up plot kwargs
        shape_kwargs = plot_kwargs.copy()

        # Configure color options
        configure_color_options(shape_kwargs)

        # Create the figure with a single trace
        fig = plot_func(df_shape, **shape_kwargs)

        # Apply symbols
        symbol_map = get_symbol_mapping()
        apply_symbol_mapping(fig, symbol_map)

        # Configure colorbar
        configure_colorbar(fig)

    elif show_chem_sys == "color+shape":
        # For color+shape mode, we need to ensure a single trace with both color and
        # shape info
        df_shape = create_shape_dataframe()

        # Set up plot kwargs without using color for grouping
        shape_kwargs = plot_kwargs.copy()

        # If we have properties, use them for coloring
        if prop_values is not None:
            shape_kwargs["color"] = color_column
            shape_kwargs["color_continuous_scale"] = heatmap_colorscale

        # Create the figure with a single trace
        # Do not group by chem_system to avoid multiple traces
        fig = plot_func(df_shape, **shape_kwargs)

        # Apply the symbols
        symbol_map = get_symbol_mapping()
        symbols = [symbol_map[cs] for cs in chem_systems]
        fig.data[0].marker.symbol = symbols

        # If we don't have properties to color by, but need to color by chem system in
        # one trace. We need to manually set the colors
        if prop_values is None:
            # Create a color mapping
            color_map = color_discrete_map or {}
            chem_sys_colors = []
            for cs in chem_systems:
                if cs in color_map:
                    chem_sys_colors.append(color_map[cs])
                else:
                    # Use Plotly default colors if no custom map provided
                    idx = sorted(uniq_chem_sys).index(cs)
                    default_colors = [
                        "#636EFA",
                        "#EF553B",
                        "#00CC96",
                        "#AB63FA",
                        "#FFA15A",
                    ]
                    chem_sys_colors.append(default_colors[idx % len(default_colors)])

            # Set marker colors directly
            fig.data[0].marker.color = chem_sys_colors

        # Configure colorbar
        configure_colorbar(fig)

    elif show_chem_sys == "color":
        # For color mode, we create a plot with color by chemical system
        # Update plot kwargs
        plot_kwargs.update(hover_data={"chem_system": True})

        # Configure color options
        configure_color_options(plot_kwargs)

        # Create the figure
        fig = plot_func(df_plot, **plot_kwargs)

        # Configure colorbar for property coloring
        configure_colorbar(fig)

    else:
        # No chemical system visualization
        # Configure color options
        configure_color_options(plot_kwargs)

        # Create the figure
        fig = plot_func(df_plot, **plot_kwargs)

        # Configure colorbar for property coloring
        configure_colorbar(fig)

    fig.update_traces(marker_size=marker_size / 3 if n_components == 3 else marker_size)

    # Update hover template
    fig.update_traces(hovertemplate="%{customdata[1]}<extra></extra>")

    # Update axis labels
    method_labels: dict[ProjectionMethod, str] = {
        "pca": "Principal Component",
        "tsne": "t-SNE Component",
        "umap": "UMAP Component",
        "isomap": "Isomap Component",
        "kernel_pca": "Kernel PCA Component",
    }

    if isinstance(projection_method, str):
        method_label = method_labels.get(projection_method, "Component")
    else:
        method_label = "Component"

    if n_components == 2:
        fig.layout.xaxis.title = f"{method_label} 1"
        fig.layout.yaxis.title = f"{method_label} 2"
    else:
        fig.layout.scene.xaxis.title = f"{method_label} 1"
        fig.layout.scene.yaxis.title = f"{method_label} 2"
        fig.layout.scene.zaxis.title = f"{method_label} 3"

    # Add projection stats annotation if available
    if projection_stats is not None:
        stats_kwargs = (
            show_projection_stats if isinstance(show_projection_stats, dict) else {}
        )
        default_stats = {
            "text": projection_stats,
            "xref": "paper",
            "yref": "paper",
            "x": 0.02,
            "y": 0.98,
            "showarrow": False,
            "font": dict(size=12),
            "bgcolor": "rgba(255,255,255,0.8)",
            "bordercolor": "rgba(0,0,0,0.1)",
            "borderwidth": 1,
            "borderpad": 4,
        }
        default_stats.update(stats_kwargs)
        fig.add_annotation(**default_stats)

    return fig
