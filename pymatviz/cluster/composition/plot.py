"""Plotting functions for chemical composition clustering."""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, get_args

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.validator_cache import ValidatorCache
from pymatgen.core import Composition
from sklearn.decomposition import PCA

from pymatviz.cluster.composition.embed import matminer_featurize, one_hot_encode
from pymatviz.cluster.composition.project import project_vectors
from pymatviz.enums import LabelEnum


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from sklearn.decomposition import KernelPCA
    from sklearn.manifold import TSNE, Isomap

    from pymatviz.typing import ColorType

symbol_validator = ValidatorCache.get_validator("scatter.marker", "symbol")
symbol_3d_validator = ValidatorCache.get_validator("scatter3d.marker", "symbol")


class ClusterFigure(go.Figure):
    """A Plotly Figure with typed metadata for clustering visualizations."""

    projector: PCA | TSNE | Isomap | KernelPCA | None
    embeddings: np.ndarray | None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the figure with optional metadata."""
        super().__init__(*args, **kwargs)
        # Use object.__setattr__ to bypass Plotly's attribute validation
        object.__setattr__(self, "projector", None)
        object.__setattr__(self, "embeddings", None)


class ProjectionCallable(Protocol):
    """Protocol for custom projection functions."""

    def __call__(
        self, embeddings: np.ndarray, n_components: int, **kwargs: Any
    ) -> np.ndarray:
        """Project embeddings to lower dimensions."""
        ...


class EmbeddingCallable(Protocol):
    """Protocol for custom embedding functions."""

    def __call__(self, compositions: Sequence[str], **kwargs: Any) -> np.ndarray:
        """Convert compositions to embeddings."""
        ...


class EmbeddingMethod(LabelEnum):
    """Supported embedding methods for composition vectorization."""

    one_hot = "one-hot", "One-hot Encoding"
    magpie = "magpie", "Matminer MagPie"
    deml = "deml", "Matminer DEML"
    matminer = "matminer", "Matminer ElementProperty"
    matscholar_el = "matscholar_el", "Matminer Matscholar Element"
    megnet_el = "megnet_el", "Matminer MEGNet Element"


class ProjectionMethod(LabelEnum):
    """Supported projection methods for dimensionality reduction."""

    pca = "pca", "Principal Component"
    tsne = "tsne", "t-SNE Component"
    umap = "umap", "UMAP Component"
    isomap = "isomap", "Isomap Component"
    kernel_pca = "kernel_pca", "Kernel PCA Component"


ShowChemSys = Literal["color", "shape", "color+shape"]
ColorScale = Literal["linear", "log", "arcsinh"]


def _arcsinh_transform(
    color_scale: ColorScale | dict[str, Any],
) -> Callable[[float], float]:
    """Use the same arcsinh scaling for property colors and colorbar positions."""
    config = color_scale if isinstance(color_scale, dict) else {}
    scale_factor = config.get("scale_factor", 2.0)
    lin_thresh, lin_scale = config.get("lin_thresh"), config.get("lin_scale")
    if lin_thresh is not None and lin_scale is not None:
        return lambda value: (
            np.arcsinh((value * lin_scale) / lin_thresh) * lin_thresh / scale_factor
        )
    return lambda value: np.arcsinh(value / scale_factor) * scale_factor


def _format_tick_label(val: float) -> str:
    """Format a colorbar tick value: scientific notation for very large/small
    magnitudes, integer/decimal notation otherwise.
    """
    if val == 0:
        return "0"
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1e4 or abs_val <= 1e-3:
        exp = math.floor(math.log10(abs_val))
        mantissa = abs_val / 10**exp
        if math.isclose(mantissa, 1, rel_tol=1e-10):
            return f"{sign}10^{exp}"
        return f"{sign}{mantissa:.1f}x10^{exp}"
    if math.isclose(val, round(val), rel_tol=1e-10, abs_tol=1e-10):
        return f"{round(val)}"
    if abs_val >= 100:
        return f"{val:.0f}"
    if abs_val >= 10:
        return f"{val:.1f}"
    return f"{val:.2f}"


def _generate_colorbar_ticks(
    color_scale: ColorScale | dict[str, Any], df_plot: pd.DataFrame
) -> tuple[list[float] | None, list[str] | None]:
    """Generate custom tick values and text for color bars.

    This function could become redundant if plotly ever adds native support for
    arcsinh or symlog colorbar scales. Tracked in https://github.com/plotly/plotly.js/issues/221.

    Args:
        color_scale (ColorScale | dict[str, Any]): The color scale type ("linear",
            "log", "arcsinh") or a dictionary with custom scale configuration.
        df_plot (pd.DataFrame): DataFrame containing plot data with property values.

    Returns:
        tuple[list[float] | None, list[str] | None]: Lists of tick values and their
            label texts. Both can be None if no custom ticks are needed.
    """
    if color_scale == "linear":
        return None, None

    # Create custom ticks that reflect the original data values
    if color_scale == "log" or (
        isinstance(color_scale, dict) and color_scale["type"] == "log"
    ):
        # For log scale, generate ticks at powers of 10
        min_val = min(val for val in df_plot["original_property"] if val > 0)
        max_val = max(val for val in df_plot["original_property"] if val > 0)

        # Find appropriate powers of 10 for the range
        min_exp = math.floor(math.log10(min_val))
        max_exp = math.ceil(math.log10(max_val))

        tick_vals = []
        tick_text = []

        # Create a list of preferred tick values (1, 2, 5 sequence)
        preferred_ticks = []
        for exp in range(min_exp - 1, max_exp + 2):  # Extend range slightly
            preferred_ticks.extend([10**exp, 2 * 10**exp, 5 * 10**exp])

        # Filter to values within or slightly outside our data range
        preferred_ticks = [
            val for val in preferred_ticks if min_val / 2 <= val <= max_val * 2
        ]
        preferred_ticks.sort()

        # If we have fewer than 5 ticks, add intermediate values
        if len(preferred_ticks) < 5:
            extra_ticks = []
            for exp in range(min_exp - 1, max_exp + 1):
                extra_ticks += [
                    1.5 * 10**exp,
                    3 * 10**exp,
                    4 * 10**exp,
                    6 * 10**exp,
                    7 * 10**exp,
                    8 * 10**exp,
                    9 * 10**exp,
                ]

            extra_ticks = [
                val for val in extra_ticks if min_val / 2 <= val <= max_val * 2
            ]

            # Combine and sort all ticks
            preferred_ticks = sorted(preferred_ticks + extra_ticks)

        # If we still don't have enough ticks, include all values
        if len(preferred_ticks) < 5:
            # Add more values as needed
            more_ticks = []
            for exp in range(min_exp - 1, max_exp + 1):
                more_ticks += [val * 10**exp for val in range(1, 10)]

            more_ticks = [
                val
                for val in more_ticks
                if min_val / 2 <= val <= max_val * 2 and val not in preferred_ticks
            ]

            preferred_ticks = sorted(preferred_ticks + more_ticks)

        # If we have too many ticks, trim to a reasonable number
        if len(preferred_ticks) > 10:
            # Use a step size to reduce number of ticks
            step = len(preferred_ticks) // 8  # Aim for around 8 ticks
            preferred_ticks = preferred_ticks[::step]

            # Make sure we include min and max values
            if preferred_ticks[0] > min_val:
                preferred_ticks.insert(0, min_val)
            if preferred_ticks[-1] < max_val:
                preferred_ticks.append(max_val)

        # Create tick values and labels
        for val in preferred_ticks:
            tick_vals.append(math.log10(val))
            tick_text.append(_format_tick_label(val))

        return tick_vals, tick_text

    if color_scale == "arcsinh" or (
        isinstance(color_scale, dict) and color_scale["type"] == "arcsinh"
    ):
        # For arcsinh scale, generate ticks that match the original data values
        min_val = min(df_plot["original_property"])
        max_val = max(df_plot["original_property"])

        transform_func = _arcsinh_transform(color_scale)

        # Generate nice tick values based on data range
        tick_vals, tick_text = [], []
        # Create preferred tick vals: include +ve and -ve in 1,2,5 sequence
        pos_ticks, neg_ticks = [], []
        zero_tick = False

        # Determine max exponent needed for both positive and negative sides
        pos_exp_max = neg_exp_max = -float("inf")

        if max_val > 0:
            pos_exp_max = math.ceil(math.log10(max_val))
        if min_val < 0:
            neg_exp_max = math.ceil(math.log10(-min_val))

        # Create all possible tick values in 1,2,5 sequence
        exp_range = max(pos_exp_max, neg_exp_max) + 1

        # Build positive ticks
        if max_val > 0:  # Start from 0.001
            for exp in range(-3, round(exp_range)):
                for base in [1, 2, 5]:
                    val = base * 10**exp
                    if 0 < val <= max_val * 1.1:
                        pos_ticks.append(val)

        # Build negative ticks
        if min_val < 0:  # Start from -0.001
            for exp in range(-3, round(exp_range)):
                for base in [1, 2, 5]:
                    val = -base * 10**exp
                    if min_val * 1.1 <= val < 0:
                        neg_ticks.append(val)

        # Include zero if the range crosses it
        if min_val <= 0 <= max_val:
            zero_tick = True

        # Combine all ticks
        all_ticks = sorted(neg_ticks + ([0] if zero_tick else []) + pos_ticks)

        # If we have less than 5 ticks, add more intermediate values
        if len(all_ticks) < 5:
            extra_ticks = []

            if max_val > 0:
                for exp in range(-3, round(exp_range)):
                    for base in (3, 4, 6, 7, 8, 9):
                        val = base * 10**exp
                        if 0 < val <= max_val * 1.1:
                            extra_ticks.append(val)

            if min_val < 0:
                for exp in range(-3, round(exp_range)):
                    for base in (3, 4, 6, 7, 8, 9):
                        val = -base * 10**exp
                        if min_val * 1.1 <= val < 0:
                            extra_ticks.append(val)

            all_ticks = sorted(all_ticks + extra_ticks)

        # If we have too many ticks, reduce them
        if len(all_ticks) > 10:
            # Use a step size to reduce number of ticks
            step = len(all_ticks) // 8  # Aim for around 8 ticks
            reduced_ticks = all_ticks[::step]

            # Make sure we include zero if in range
            if min_val <= 0 <= max_val and 0 not in reduced_ticks:
                # Find the index where 0 would be inserted to maintain order
                for idx, val in enumerate(reduced_ticks):
                    if val > 0:
                        reduced_ticks.insert(idx, 0)
                        break
                else:
                    reduced_ticks.append(0)

            all_ticks = reduced_ticks

        # Transform ticks and format labels
        for val in all_ticks:
            tick_vals.append(transform_func(val))
            tick_text.append(_format_tick_label(val))

        if tick_vals:
            return tick_vals, tick_text

    return None, None


def _composition_strings(
    compositions: Iterable[Any], *, strict: bool = True
) -> list[str]:
    """Convert compositions to a list of formula strings."""
    comp_strs: list[str] = []
    for comp in compositions:
        if isinstance(comp, str):
            comp_strs.append(comp)
        elif isinstance(comp, Composition):
            comp_strs.append(comp.formula)
        elif strict:
            raise TypeError(f"Expected str or Composition, got {comp=}")
        else:
            comp_strs.append(str(comp))
    return comp_strs


def cluster_compositions(
    df_in: pd.DataFrame,
    composition_col: str = "composition",
    *,
    prop_name: str | None = None,
    embedding_method: str
    | EmbeddingMethod
    | EmbeddingCallable = EmbeddingMethod.magpie,
    projection: ProjectionMethod | ProjectionCallable | str,
    n_components: int = 2,
    hover_format: str = ".2f",
    heatmap_colorscale: str = "Viridis",
    marker_size: int = 8,
    show_chem_sys: ShowChemSys | None = None,
    color_discrete_map: Mapping[str, ColorType] | None = None,
    embedding_kwargs: dict[str, Any] | None = None,
    projection_kwargs: dict[str, Any] | None = None,
    sort: bool | int | Callable[[np.ndarray], np.ndarray] = True,
    show_projection_stats: bool | dict[str, Any] = True,
    color_scale: ColorScale | dict[str, Any] = "linear",
    annotate_points: Callable[[pd.Series], str | dict[str, Any] | None] | None = None,
    **kwargs: Any,
) -> ClusterFigure:
    """Plot chemical composition clusters with optional property coloring.

    Gives a 2D or 3D scatter plot of chemical compositions, using various
    embedding and dimensionality reduction techniques to visualize the relationships
    between different materials.

    Args:
        df_in (pd.DataFrame): DataFrame containing composition data and optionally
            properties and/or pre-computed embeddings.
        composition_col (str): Name of the column containing one of:
            - Chemical formulas (as strings)
            - pymatgen Composition objects
            - Pre-computed embeddings (as numpy arrays or lists)
            Default is "composition".
        prop_name (str | None): Name of the column to use for coloring points.
            If provided, the values in this column will be used to color the points.
            (default: None)
        embedding_method (str | EmbeddingMethod | Callable[[list[str], Any], ndarray]):
            Method to convert compositions to vectors (default: "magpie"). Options:
            - "one-hot": One-hot encoding of element fractions
            - "magpie": Matminer's MagPie featurization
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
        color_scale (str | dict): Method for scaling property values for coloring points
            (default: "linear"):
            - "linear": Linear scale (default)
            - "log": Logarithmic scale for positive values
            - "arcsinh": Inverse hyperbolic sine scale, handles both large positive
              and negative values
            - dict: Custom scale configuration with required "type" key:
                - {"type": "arcsinh", "scale_factor": float}: Custom scale factor
                  (default: 2)
                - {"type": "arcsinh", "lin_thresh": float, "lin_scale": float}:
                  Custom linearity threshold and scale (for fine-tuning the
                  transition between linear and logarithmic regions)
        annotate_points (Callable[[pd.Series], str | dict[str, Any] | None] | None):
            Function to generate text annotations for each point. Takes a pandas Series
            representing a row in the input DataFrame with added rows for the projected
            coordinates (e.g., 'pca1', 'pca2'). Should return a string to use as label
            or a dict with valid keys for fig.add_annotation().
        **kwargs: Passed to px.scatter or px.scatter_3d (depending on n_components)

    Returns:
        ClusterFigure: Plotly figure object with typed metadata attributes:
            - fig.projector: Fitted projection object (PCA, TSNE, etc.) or None
              for custom projection functions
            - fig.embeddings: Embeddings used for projection
    """
    if n_components not in (2, 3):  # Validate inputs
        raise ValueError(f"{n_components=} must be 2 or 3")

    if not isinstance(show_projection_stats, (bool, dict)):
        raise TypeError(f"{show_projection_stats=} must be bool or dict")

    if composition_col not in df_in:
        columns = df_in.columns.tolist()
        raise ValueError(f"{composition_col=} not found in DataFrame {columns=}")

    if prop_name is not None and prop_name not in df_in:
        raise ValueError(
            f"{prop_name=} not found in DataFrame columns: {df_in.columns.tolist()}"
        )

    if projection is None:
        raise ValueError(
            "projection must be specified. Choose from: "
            f"{list(ProjectionMethod)} or provide a custom function or column name."
        )

    # Validate color_scale parameter
    valid_scales = get_args(ColorScale)
    if isinstance(color_scale, str):
        if color_scale not in valid_scales:
            raise ValueError(f"{color_scale=} must be one of {valid_scales} or dict")
    elif isinstance(color_scale, dict):
        if "type" not in color_scale:
            raise ValueError("When color_scale is a dict, 'type' key must be provided")
        scale_type = color_scale["type"]
        if scale_type not in valid_scales:
            raise ValueError(
                f"color_scale dict 'type'='{scale_type}' must be one of {valid_scales}"
            )
        # Validate arcsinh parameters if provided
        if scale_type == "arcsinh":
            scale_factor = color_scale.get("scale_factor", 2.0)
            if scale_factor <= 0:
                raise ValueError(f"{scale_factor=} must be positive for arcsinh scale")
    else:
        raise TypeError(
            f"color_scale must be a string or a dict, got {type(color_scale).__name__}"
        )

    # Check if projection is a column name in the DataFrame
    using_precomputed_coords = projection in df_in and projection not in list(
        ProjectionMethod
    )

    if using_precomputed_coords:
        # Validate the coordinates
        first_coords = df_in[projection].iloc[0]
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
    compositions = df_in[composition_col]
    properties = df_in[prop_name] if prop_name is not None else None

    # Check if pre-computed coordinates are provided in a DataFrame column
    if using_precomputed_coords:
        # Use pre-computed coordinates from the specified column
        projected = np.array([np.array(coords) for coords in df_in[projection]])
        projector = None
        embeddings = None  # We don't need to calculate embeddings

        # For hover text, we still need composition strings
        comp_strs = _composition_strings(compositions, strict=False)
    else:  # No pre-computed coordinates, follow normal embedding and projection flow
        # Handle embeddings based on the type of data in the composition column
        first_val = compositions.iloc[0]

        if isinstance(first_val, (list, np.ndarray)):
            # Direct pre-computed embeddings in composition_col
            comp_strs = df_in.index.tolist()  # Use DataFrame index for compositions
            embeddings = np.array([np.array(val) for val in compositions])
        elif isinstance(embedding_method, str) and embedding_method in df_in:
            # Using a specified column for embeddings
            comp_strs = _composition_strings(compositions)

            # Get embeddings from the specified column
            embeddings = np.array([np.array(val) for val in df_in[embedding_method]])
        else:
            # Convert compositions to strings for consistent handling
            comp_strs = _composition_strings(compositions)

            # Create embeddings
            if callable(embedding_method):
                # Use custom embedding function
                embedding_fn = cast("EmbeddingCallable", embedding_method)
                embeddings = embedding_fn(compositions, **(embedding_kwargs or {}))
            # Use built-in embedding methods
            elif embedding_method == "one-hot":
                embeddings = one_hot_encode(compositions, **(embedding_kwargs or {}))
            else:
                try:
                    preset = EmbeddingMethod(embedding_method)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{embedding_method=} must be in {list(EmbeddingMethod)}, "
                        "a callable, or a valid column name in the DataFrame"
                    ) from exc
                embeddings = matminer_featurize(
                    compositions,
                    preset=preset.value,  # ty: ignore[invalid-argument-type]
                    **(embedding_kwargs or {}),
                )

        # Project embeddings
        if callable(projection):
            # Use custom projection function
            projection_fn = cast("ProjectionCallable", projection)
            projected = projection_fn(
                embeddings,
                n_components=n_components,
                **projection_kwargs,
            )
            projector = None
        else:
            # Use built-in projection methods
            try:
                projection_method = ProjectionMethod(projection)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{projection=} must be in {list(ProjectionMethod)}, "
                    f"a callable, or column name in the DataFrame"
                ) from exc
            projected, projector = project_vectors(
                embeddings,
                method=projection_method,
                n_components=n_components,
                **projection_kwargs,
            )

    # Extract chemical systems if needed
    chem_systems = None
    uniq_chem_sys = set()
    valid_symbols = []

    if show_chem_sys in ("shape", "color", "color+shape"):
        chem_systems = []
        symbol_filter = lambda x: (
            isinstance(x, str)
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
            all_symbols = symbol_3d_validator.values  # noqa: PD011
            valid_symbols = list(filter(symbol_filter, all_symbols))
        else:
            # For 2D plots, we can use the full set of markers
            all_symbols = symbol_validator.values  # noqa: PD011
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

    df_plot = pd.DataFrame(index=df_in.index)
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
    colorbar_title = None  # Initialize to handle all code paths

    if properties is not None:
        prop_values = properties.tolist()

        # Apply color scale transformations if needed
        if color_scale != "linear" and prop_name is not None:
            # Create a copy to avoid modifying the original data
            original_prop_values = prop_values.copy()

            # Apply the selected transformation
            if color_scale == "log" or (
                isinstance(color_scale, dict) and color_scale["type"] == "log"
            ):
                # For log scale, replace negative or zero values with NaN
                prop_values = [
                    math.log10(val) if val > 0 else float("nan") for val in prop_values
                ]
                colorbar_title = f"{prop_name} (log scale)"

            elif color_scale == "arcsinh" or (
                isinstance(color_scale, dict) and color_scale["type"] == "arcsinh"
            ):
                transform = _arcsinh_transform(color_scale)
                prop_values = [transform(value) for value in prop_values]
                colorbar_title = f"{prop_name} (arcsinh scale)"
            else:
                # For linear scale, no transformation needed
                colorbar_title = prop_name

            # Update the DataFrame with transformed values for plotting
            df_plot[prop_name] = prop_values

            # Store original values for hover text
            df_plot["original_property"] = original_prop_values
        else:
            # For linear scale, no transformation needed
            df_plot[prop_name] = prop_values
            colorbar_title = prop_name

        color_column = prop_name
    elif chem_systems is not None and show_chem_sys in ("color", "color+shape"):
        # Only color by chem_system for "color" or "color+shape" modes
        color_column = "chem_system"
    else:
        color_column = None

    if sort and prop_values is not None:
        if isinstance(sort, int):  # Includes True; False is handled above.
            sort_indices = np.argsort(prop_values)
            if sort < 0:
                sort_indices = sort_indices[::-1]
        elif callable(sort):
            sort_indices = sort(np.asarray(prop_values))
        else:
            raise TypeError(f"Invalid sort parameter type: {type(sort).__name__}")

        df_plot = df_plot.iloc[sort_indices]
        projected = projected[sort_indices]
        if embeddings is not None:
            embeddings = embeddings[sort_indices]
        comp_strs = [comp_strs[idx] for idx in sort_indices]
        if chem_systems is not None:
            chem_systems = [chem_systems[idx] for idx in sort_indices]
        prop_values = [prop_values[idx] for idx in sort_indices]

    # Create hover text template
    hover_template: list[str] = []

    # Determine the method label for hover text
    if using_precomputed_coords:  # For pre-computed coordinates, use generic label
        method_label = "Component"
    elif projection in list(ProjectionMethod):
        # For built-in projection methods, use standardized labels
        method_label = ProjectionMethod(projection).label
    else:  # For custom projection functions, use func.__name__
        # or generic fallback if unnamed
        method_label = getattr(projection, "__name__", "Component")
        if method_label in ("<lambda>", "lambda", "", " ", None):
            method_label = "Component"

    # Format string for projected coordinates and property values in hover text
    coord_fmt = (
        f"{{:.{hover_format[1:]}}}" if hover_format.startswith(".") else hover_format
    )

    for idx, comp in enumerate(comp_strs):
        hover_text = f"Composition: {comp}<br>"

        # Add projected coordinates
        hover_text += f"{method_label} 1: {coord_fmt.format(projected[idx, 0])}<br>"
        hover_text += f"{method_label} 2: {coord_fmt.format(projected[idx, 1])}<br>"
        if n_components == 3:
            hover_text += f"{method_label} 3: {coord_fmt.format(projected[idx, 2])}<br>"

        # Add property or chemical system
        if prop_values is not None:
            prop_fmt = coord_fmt

            try:  # Show the property value
                if color_scale != "linear" and "original_property" in df_plot:
                    # For non-linear scales, show original value only
                    original_val = df_plot["original_property"].iloc[idx]
                    if not np.isnan(original_val):
                        hover_text += f"{prop_name}: {prop_fmt.format(original_val)}"
                    else:
                        hover_text += f"{prop_name}: NaN"
                else:  # For linear scale, just show the property value
                    hover_text += f"{prop_name}: {prop_fmt.format(prop_values[idx])}"
            except Exception:  # noqa: BLE001
                hover_text += f"{prop_name}: {prop_values[idx]}"
        elif chem_systems is not None:
            hover_text += f"Chemical System: {chem_systems[idx]}"

        hover_template.append(hover_text)

    # Store hover text in the dataframe
    df_plot["hover_text"] = hover_template

    # Calculate projection statistics
    projection_stats: str | None = None
    if show_projection_stats and not using_precomputed_coords:
        if projection == "pca" and isinstance(projector, PCA):
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
        elif projector is not None and isinstance(projection, str):
            stat_params = {
                "tsne": ("perplexity", "learning_rate"),
                "umap": ("n_neighbors", "min_dist"),
                "isomap": ("n_neighbors", "metric"),
                "kernel_pca": ("kernel", "gamma"),
            }[projection]
            projection_stats = "<br>".join(
                f"{param} = {getattr(projector, param)!r}" for param in stat_params
            )

    # Create the plot
    plot_func = px.scatter if n_components == 2 else px.scatter_3d
    plot_kwargs: dict[str, Any] = {
        "x": x_name,
        "y": y_name,
        "custom_data": ["composition", "hover_text"],
    }
    if n_components == 3:
        plot_kwargs["z"] = z_name

    if prop_values is not None:
        plot_kwargs.update(
            color=color_column, color_continuous_scale=heatmap_colorscale
        )
    elif color_column is not None and show_chem_sys != "color+shape":
        plot_kwargs.update(color=color_column, color_discrete_map=color_discrete_map)
    if show_chem_sys == "color":
        plot_kwargs["hover_data"] = {"chem_system": True}

    fig = plot_func(df_plot, **plot_kwargs | kwargs)

    if show_chem_sys in ("shape", "color+shape") and chem_systems is not None:
        symbol_map = {
            system: valid_symbols[idx % len(valid_symbols)]
            for idx, system in enumerate(sorted(uniq_chem_sys))
        }
        fig.data[0].marker.symbol = [symbol_map[system] for system in chem_systems]

        # Keep color+shape in one trace, assigning category colors per point.
        if show_chem_sys == "color+shape" and prop_values is None:
            default_colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
            color_map = {
                system: default_colors[idx % len(default_colors)]
                for idx, system in enumerate(sorted(uniq_chem_sys))
            } | dict(color_discrete_map or {})
            fig.data[0].marker.color = [color_map[system] for system in chem_systems]

    fig.update_traces(marker_size=marker_size / 3 if n_components == 3 else marker_size)

    fig.update_traces(hovertemplate="%{customdata[1]}<extra></extra>")

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
        tick_vals, tick_text = _generate_colorbar_ticks(color_scale, df_plot)
        color_bar = dict(
            orientation="h",
            yanchor="bottom",
            y=0,
            xanchor="right",
            x=0.99,
            thickness=12,
            len=350,
            lenmode="pixels",
            title=dict(text=colorbar_title, side="top"),
            tickvals=tick_vals,
            ticktext=tick_text,
        )

        fig.layout.coloraxis.colorbar.update(**color_bar)

    if callable(annotate_points):
        df_plot[df_in.columns] = df_in
        # Batch assignment avoids revalidating the entire array for every point.
        annotations = []
        for _idx, row in df_plot.iterrows():
            row_annotation = annotate_points(row)
            if isinstance(row_annotation, str):
                row_annotation = {"text": row_annotation}
            if not row_annotation:
                continue
            annotation = {
                "x": row[x_name],
                "y": row[y_name],
                **({"z": row[z_name]} if n_components == 3 else {}),
                "showarrow": False,
                "yshift": 10,
                "font": {"size": 10},
                **row_annotation,
            }
            annotations.append(
                go.layout.scene.Annotation(annotation)
                if n_components == 3
                else annotation
            )
        if n_components == 3:
            fig.update_layout(scene_annotations=annotations)
        else:
            fig.layout.annotations = list(fig.layout.annotations) + annotations

    # Convert to ClusterFigure and attach metadata
    cluster_fig = ClusterFigure(fig)
    object.__setattr__(cluster_fig, "projector", projector)
    if embeddings is not None:
        object.__setattr__(cluster_fig, "embeddings", embeddings)

    return cluster_fig
