"""Plotting functions for chemical composition clustering."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Final, Literal, get_args

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
# Method labels for hover tooltips and axis labels
method_labels: Final[dict[ProjectionMethod, str]] = {
    "pca": "Principal Component",
    "tsne": "t-SNE Component",
    "umap": "UMAP Component",
    "isomap": "Isomap Component",
    "kernel_pca": "Kernel PCA Component",
}


def cluster_compositions(
    df: pd.DataFrame,
    composition_col: str = "composition",
    *,
    prop_name: str | None = None,
    embedding_method: EmbeddingMethod
    | Callable[[Sequence[str], Any], np.ndarray]
    | str = "magpie",
    projection: ProjectionMethod | Callable[[np.ndarray, int, Any], np.ndarray] | str,
    n_components: int = 2,
    hover_format: str = ".2f",
    heatmap_colorscale: str = "Viridis",
    marker_size: int = 8,
    show_chem_sys: ShowChemSys | None = None,
    color_discrete_map: dict[str, ColorType] | None = None,
    embedding_kwargs: dict[str, Any] | None = None,
    projection_kwargs: dict[str, Any] | None = None,
    sort: bool | int | Callable[[np.ndarray], np.ndarray] = True,
    show_projection_stats: bool | dict[str, Any] = True,
    **kwargs: Any,
) -> go.Figure:
    """Plot chemical composition clusters with optional property coloring.

    Gives a 2D or 3D scatter plot of chemical compositions, using various
    embedding and dimensionality reduction techniques to visualize the relationships
    between different materials.

    Args:
        df (pd.DataFrame): DataFrame containing composition data and optionally
            properties and/or pre-computed embeddings.
        composition_col (str): Name of the column containing one of:
            - Chemical formulas (as strings)
            - pymatgen Composition objects
            - Pre-computed embeddings (as numpy arrays or lists)
            Default is "composition".
        prop_name (str | None): Name of the column to use for coloring points.
            If provided, the values in this column will be used to color the points.
            (default: None)
        embedding_method (str | EmbedMethod | Callable[[list[str], Any], np.ndarray]):
            Method to convert compositions to vectors (default: "magpie"). Options:
            - "one-hot": One-hot encoding of element fractions
            - "magpie": Matminer's MagPie featurization (for backward compatibility)
            - "deml": Matminer's DEML featurization
            - "matminer": Matminer's ElementProperty featurization
            - "matscholar_el": Matminer's Matscholar Element featurization
            - "megnet_el": Matminer's MEGNet Element featurization
            - Callable: Custom embedding function that takes (N, ) array of compositions
              and returns (N, D) array of values where D is the number of dimensions in
              the embedding space.
            - Column name in df: Name of column containing pre-computed embeddings.
              If using this option, the column must contain numpy arrays or lists.
            Ignored if composition_col contains pre-computed embeddings.
        projection: Method to reduce embedding dimensionality or column name with
            pre-computed coordinates
            - "pca": Principal Component Analysis (linear)
            - "tsne": t-distributed Stochastic Neighbor Embedding (non-linear)
            - "umap": Uniform Manifold Approximation and Projection (non-linear)
            - "isomap": Isometric Feature Mapping (non-linear)
            - "kernel_pca": Kernel Principal Component Analysis (non-linear)
            - Callable: Custom projection function that takes (N, D) array of values and
              returns (N, n_components) projected array where n_components is 2 or 3 and
              D is the number of dimensions in the embedding space.
            - Column name in df: Name of column containing pre-computed coordinates.
              If using this option, the column must contain numpy arrays or lists of
              length equal to n_components.
        n_components (int): Projection dimensions (2 or 3) (default: 2)
        hover_format (str): Format string for hover data (default: ".2f")
        heatmap_colorscale (str): Colorscale for continuous property values
            (default: "Viridis")
        marker_size (int): Size of the scatter plot points (default: 8)
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
        **kwargs: Passed to px.scatter or px.scatter_3d (depending on n_components)

    Returns:
        go.Figure: Plotly figure object with embeddings and projection object in
            '_pymatviz_meta' attribute:
            - fig._pymatviz_meta["projector"]: Fitted projection object
              (PCA, TSNE, etc.) or None for custom projection functions
            - fig._pymatviz_meta["embeddings"]: Embeddings used for projection
    """
    if n_components not in (2, 3):  # Validate inputs
        raise ValueError(f"{n_components=} must be 2 or 3")

    if not isinstance(show_projection_stats, (bool, dict)):
        raise TypeError(f"{show_projection_stats=} must be bool or dict")

    if composition_col not in df.columns:
        raise ValueError(
            f"{composition_col=} not found in DataFrame columns: {df.columns.tolist()}"
        )

    if prop_name is not None and prop_name not in df.columns:
        raise ValueError(
            f"{prop_name=} not found in DataFrame columns: {df.columns.tolist()}"
        )

    if projection is None:
        raise ValueError(
            "projection must be specified. Choose from: "
            f"{get_args(ProjectionMethod)} or provide a custom function or column name."
        )

    # Check if projection is a column name in the DataFrame
    using_precomputed_coords = projection in df and projection not in get_args(
        ProjectionMethod
    )

    if using_precomputed_coords:
        # Validate the coordinates
        first_coords = df[projection].iloc[0]
        if not isinstance(first_coords, (list, np.ndarray)):
            raise ValueError(
                f"Column {projection} must contain arrays or lists of coordinates"
            )
        if len(first_coords) != n_components:
            raise ValueError(
                f"Coordinates in {projection} column must have length {n_components}, "
                f"got {len(first_coords)}"
            )

    projection_kwargs = projection_kwargs or {}

    # Get compositions from DataFrame
    compositions = df[composition_col]
    properties = df[prop_name] if prop_name is not None else None

    # Check if pre-computed coordinates are provided in a DataFrame column
    if using_precomputed_coords:
        # Use pre-computed coordinates from the specified column
        projected = np.array([np.array(coords) for coords in df[projection]])
        projector = None
        embeddings = None  # We don't need to calculate embeddings

        # For hover text, we still need composition strings
        comp_strs = []
        for comp in compositions:
            if isinstance(comp, str):
                comp_strs.append(comp)
            elif isinstance(comp, Composition):
                comp_strs.append(comp.formula)
            else:  # Use string representation for unknown types
                comp_strs.append(str(comp))
    else:  # No pre-computed coordinates, follow normal embedding and projection flow
        # Handle embeddings based on the type of data in the composition column
        first_val = compositions.iloc[0]

        if isinstance(first_val, (list, np.ndarray)):
            # Direct pre-computed embeddings in composition_col
            comp_strs = df.index.tolist()  # Use DataFrame index for compositions
            embeddings = np.array([np.array(val) for val in compositions])
        elif isinstance(embedding_method, str) and embedding_method in df.columns:
            # Using a specified column for embeddings
            comp_strs = []
            for comp in compositions:
                if isinstance(comp, str):
                    comp_strs.append(comp)
                elif isinstance(comp, Composition):
                    comp_strs.append(comp.formula)
                else:
                    raise TypeError(f"Expected str or Composition, got {comp=}")

            # Get embeddings from the specified column
            embeddings = np.array([np.array(val) for val in df[embedding_method]])
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
                embeddings = embedding_method(compositions, **(embedding_kwargs or {}))
            # Use built-in embedding methods
            elif embedding_method == "one-hot":
                embeddings = one_hot_encode(compositions, **(embedding_kwargs or {}))
            elif embedding_method in get_args(EmbeddingMethod):
                embeddings = matminer_featurize(
                    compositions, preset=embedding_method, **(embedding_kwargs or {})
                )
            else:
                raise ValueError(
                    f"{embedding_method=} must be in {get_args(EmbeddingMethod)}, "
                    f"a callable, or a valid column name in the DataFrame"
                )

        # Project embeddings
        if callable(projection):
            # Use custom projection function
            projected = projection(
                embeddings,
                n_components=n_components,
                **projection_kwargs,  # type: ignore[call-arg]
            )
            projector = None
        elif projection in get_args(ProjectionMethod):
            # Use built-in projection methods
            projected, projector = project_vectors(
                embeddings,
                method=projection,  # type: ignore[arg-type]
                n_components=n_components,
                **projection_kwargs,
            )
        else:
            raise ValueError(
                f"{projection=} must be in {get_args(ProjectionMethod)}, "
                f"a callable, or column name in the DataFrame"
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

    df_plot = pd.DataFrame()
    df_plot["composition"] = comp_strs

    # Determine label for projection columns
    if isinstance(projection, str):
        # If using pre-computed coordinates from df column, use generic name
        # If using built-in projection method, use that name
        proj_name = "coordinates" if using_precomputed_coords else projection
    else:  # For custom projection functions, use generic name
        proj_name = "coordinates"

    df_plot[(x_name := f"{proj_name}1")] = projected[:, 0]
    df_plot[(y_name := f"{proj_name}2")] = projected[:, 1]
    z_name = f"{proj_name}3"
    if n_components == 3:
        df_plot[z_name] = projected[:, 2]

    if chem_systems is not None:
        df_plot["chem_system"] = chem_systems

    # Add properties or configure chemical system coloring
    prop_values: list[float] | None = None
    if properties is not None:
        prop_values = properties.tolist()
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
            if embeddings is not None:
                embeddings = embeddings[sort_indices]
            # Also update comp_strs to maintain alignment
            comp_strs = [comp_strs[i] for i in sort_indices]
            # Update other array-like data if present
            if chem_systems is not None:
                chem_systems = [chem_systems[idx] for idx in sort_indices]
            if prop_values is not None:
                prop_values = [prop_values[idx] for idx in sort_indices]
        else:
            raise TypeError(f"Invalid sort parameter type: {type(sort).__name__}")

        if not callable(sort):
            # Sort by property values
            sort_indices = np.argsort(prop_values)
            if sort_direction < 0:
                sort_indices = sort_indices[::-1]  # Reverse for descending order
            df_plot = df_plot.iloc[sort_indices]
            projected = projected[sort_indices]
            if embeddings is not None:
                embeddings = embeddings[sort_indices]
            # Also update comp_strs to maintain alignment
            comp_strs = [comp_strs[idx] for idx in sort_indices]
            # Update other array-like data if present
            if chem_systems is not None:
                chem_systems = [chem_systems[idx] for idx in sort_indices]
            if prop_values is not None:
                prop_values = [prop_values[idx] for idx in sort_indices]

    # Create hover text template
    hover_template: list[str] = []

    # Determine the method label for hover text
    if using_precomputed_coords:  # For pre-computed coordinates, use generic label
        method_label = "Component"
    elif isinstance(projection, str):
        # For built-in projection methods, use standardized labels
        method_label = method_labels.get(projection, "Component")  # type: ignore[call-overload]
    else:  # For custom projection funcs, use func name or generic fallback if unnamed
        method_label = getattr(projection, "__name__", "Component")
        if method_label in ("<lambda>", "lambda", "", " ", None):
            method_label = "Component"

    for idx, comp in enumerate(comp_strs):
        hover_text = f"Composition: {comp}<br>"

        # Add projected coordinates
        coord_fmt = (
            f"{{:.{hover_format[1:]}}}"
            if hover_format.startswith(".")
            else hover_format
        )
        hover_text += f"{method_label} 1: {coord_fmt.format(projected[idx, 0])}<br>"
        hover_text += f"{method_label} 2: {coord_fmt.format(projected[idx, 1])}<br>"
        if n_components == 3:
            hover_text += f"{method_label} 3: {coord_fmt.format(projected[idx, 2])}<br>"

        # Add property or chemical system
        if prop_values is not None:
            prop_fmt = (
                f"{{:.{hover_format[1:]}}}"
                if hover_format.startswith(".")
                else hover_format
            )
            try:
                hover_text += f"{prop_name}: {prop_fmt.format(prop_values[idx])}"
            except Exception:  # noqa: BLE001
                hover_text += f"{prop_name}: {prop_values[idx]}"
        elif chem_systems is not None:
            hover_text += f"Chemical System: {chem_systems[idx]}"

        hover_template.append(hover_text)

    df_plot["hover_text"] = hover_template

    # Calculate projection statistics
    projection_stats: str | None = None
    if show_projection_stats and not using_precomputed_coords:
        if projection == "pca" and projector is not None:
            # Get explained variance ratios from PCA object
            var_explained_ratio = projector.explained_variance_ratio_[:n_components]
            cum_var_explained = np.cumsum(var_explained_ratio)

            # Create variance stats text
            stats_text: list[str] = []
            for idx, (var, cum_var) in enumerate(
                zip(var_explained_ratio, cum_var_explained, strict=True)
            ):
                stats_text.append(f"PC{idx + 1}: {var:.1%} (cumulative: {cum_var:.1%})")
            projection_stats = "<br>".join(stats_text)
        elif projection == "tsne":
            # For t-SNE, show perplexity and learning rate
            perplexity = projection_kwargs.get("perplexity", 30)
            learning_rate = projection_kwargs.get("learning_rate", 200)
            stats_text = [f"{perplexity = }", f"{learning_rate = }"]
            projection_stats = "<br>".join(stats_text)
        elif projection == "umap":
            # For UMAP, show n_neighbors and min_dist
            n_neighbors = projection_kwargs.get("n_neighbors", 15)
            min_dist = projection_kwargs.get("min_dist", 0.1)
            stats_text = [f"{n_neighbors = }", f"{min_dist = }"]
            projection_stats = "<br>".join(stats_text)
        elif projection == "isomap":
            # For Isomap, show n_neighbors and metric
            n_neighbors = projection_kwargs.get("n_neighbors", 15)
            metric = projection_kwargs.get("metric", "euclidean")
            stats_text = [f"{n_neighbors = }", f"{metric = }"]
            projection_stats = "<br>".join(stats_text)
        elif projection == "kernel_pca":
            # For Kernel PCA, show kernel type and parameters
            kernel = projection_kwargs.get("kernel", "linear")
            gamma = projection_kwargs.get("gamma", 1.0)
            stats_text = [f"{kernel = }", f"{gamma = }"]
            projection_stats = "<br>".join(stats_text)

    # Create the plot
    plot_func = px.scatter if n_components == 2 else px.scatter_3d
    plot_kwargs: dict[str, Any] = {
        "x": x_name,
        "y": y_name,
        "custom_data": ["composition", "hover_text"],
    }
    if n_components == 3:
        plot_kwargs["z"] = z_name

    # Define helper functions to reduce redundancy
    def create_shape_df() -> pd.DataFrame:
        """Create a dataframe for shape-based visualization."""
        df_shape = pd.DataFrame()
        df_shape["composition"] = df_plot["composition"]
        df_shape[x_name] = df_plot[x_name]
        df_shape[y_name] = df_plot[y_name]
        if n_components == 3:
            df_shape[z_name] = df_plot[z_name]
        df_shape["hover_text"] = df_plot["hover_text"]
        if chem_systems is not None:
            df_shape["chem_system"] = chem_systems
        # Add properties for coloring if available
        if prop_values is not None:
            df_shape[color_column] = prop_values
        return df_shape

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

    # Set up color mapping
    def configure_color_options(kwargs: dict[str, Any]) -> None:
        """Configure color options based on properties and chemical systems."""
        if prop_values is not None:
            kwargs.update(color=color_column, color_continuous_scale=heatmap_colorscale)
        elif color_column is not None:
            kwargs.update(color=color_column, color_discrete_map=color_discrete_map)

    # Handle different visualization modes
    symbol_map = {
        system: valid_symbols[idx % len(valid_symbols)]
        for idx, system in enumerate(sorted(uniq_chem_sys))
    }
    if show_chem_sys == "shape":
        # Create dataframe for shape-based visualization
        df_shape = create_shape_df()

        # Set up plot kwargs
        shape_kwargs = plot_kwargs.copy()

        configure_color_options(shape_kwargs)

        fig = plot_func(df_shape, **shape_kwargs | kwargs)

        # Apply symbols
        apply_symbol_mapping(fig, symbol_map)

    elif show_chem_sys == "color+shape":
        # For color+shape mode, we need to ensure a single trace with both color and
        # shape info
        df_shape = create_shape_df()

        # Set up plot kwargs without using color for grouping
        shape_kwargs = plot_kwargs.copy()

        # If we have properties, use them for coloring
        if prop_values is not None:
            shape_kwargs["color"] = color_column
            shape_kwargs["color_continuous_scale"] = heatmap_colorscale

        # Do not group by chem_system to avoid multiple traces
        fig = plot_func(df_shape, **shape_kwargs | kwargs)

        # Apply the symbols
        symbols = [symbol_map[cs] for cs in chem_systems]
        fig.data[0].marker.symbol = symbols

        # If we don't have properties to color by, but need to color by chemical system
        # we need to manually set the colors
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

    elif show_chem_sys == "color":
        # For color mode, we create a plot with color by chemical system
        plot_kwargs.update(hover_data={"chem_system": True})

        configure_color_options(plot_kwargs)

        fig = plot_func(df_plot, **plot_kwargs | kwargs)

    else:  # No chemical system visualization
        configure_color_options(plot_kwargs)

        fig = plot_func(df_plot, **plot_kwargs | kwargs)

    fig.update_traces(marker_size=marker_size / 3 if n_components == 3 else marker_size)

    fig.update_traces(hovertemplate="%{customdata[1]}<extra></extra>")

    # Use the already defined method_label for axis labels
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
            title=dict(text=prop_name, side="top"),
        )
        fig.update_layout(coloraxis_colorbar=color_bar)

    # Attach projector and embeddings as metadata dict
    # since Plotly doesn't allow arbitrary attributes on Figures
    fig._pymatviz_meta = {"projector": projector}
    if embeddings is not None:
        fig._pymatviz_meta["embeddings"] = embeddings

    return fig
