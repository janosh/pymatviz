"""Plotting functions for chemical composition clustering."""

from __future__ import annotations

import itertools
import math
import numbers
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
    """Format a colorbar tick value with the fewest significant digits that represent
    it (to 1e-9 relative precision, so distinct ticks get distinct labels): positional
    notation for magnitudes in [1e-3, 1e4), scientific (e.g. 2x10^-4, 10^5) otherwise.
    """
    if val == 0:
        return "0"
    sig_figs = next(
        n_digits
        for n_digits in range(1, 18)
        if math.isclose(float(f"{val:.{n_digits - 1}e}"), val, rel_tol=1e-9)
    )
    if 1e-3 <= abs(val) < 1e4:
        return np.format_float_positional(
            val, precision=sig_figs, unique=False, fractional=False, trim="-"
        )
    mantissa, exp = f"{abs(val):.{sig_figs - 1}e}".split("e")
    sign = "-" if val < 0 else ""
    if mantissa == "1":
        return f"{sign}10^{int(exp)}"
    return f"{sign}{mantissa}x10^{int(exp)}"


def _thin_ticks(ticks: list[float], keep: float | None = None) -> list[float]:
    """Thin more than 10 ticks to the densest subset of 5-10 ticks whose mantissas
    are in (1, 2, 5), (1, 5) or (1,) (i.e. powers of ten). If there's no such subset,
    use every n-th power of ten (if more than 10) or every 2nd tick in phase with
    the powers of ten. keep (e.g. 0) is always retained.
    """
    if len(ticks) <= 10:
        return ticks

    def mantissa(tick: float) -> float:
        """Leading digits of tick in scientific notation, e.g. 2.5 for -2.5e-3."""
        return float(f"{abs(tick):.6e}".split("e")[0])

    subset = ticks
    for mantissas in ((1, 2, 5), (1, 5), (1,)):
        subset = [
            tick
            for tick in ticks
            if tick == keep or (tick != 0 and mantissa(tick) in mantissas)
        ]
        if 5 <= len(subset) <= 10:
            return subset

    if len(subset) > 10:  # many decades: every n-th power of ten
        thinned = subset[:: math.ceil(len(subset) / 10)]
    else:  # few decades: every 2nd tick, in phase with the powers of ten
        phase = next(
            (idx for idx, tick in enumerate(ticks) if tick in subset and tick != keep),
            0,
        )
        thinned = ticks[phase % 2 :: 2]
    if keep is not None and keep in ticks and keep not in thinned:
        thinned = sorted([*thinned, keep])
    return thinned


def _scale_type(color_scale: ColorScale | dict[str, Any]) -> str:
    """Return the scale type of a color_scale name or config dict."""
    return color_scale["type"] if isinstance(color_scale, dict) else color_scale


def _nice_ticks(min_val: float, max_val: float, min_mag: float) -> list[float]:
    """Nice ticks for a log-like colorbar spanning [min_val, max_val].

    Ticks are signed base * 10^exp values in a 1, 2, 5 sequence (densified with
    1.5, 3, 4, 6, 7, 8, 9 if that gives fewer than 5 ticks) within the data range
    (ticks beyond it aren't shown), plus 0 if the range crosses it. More than 10
    ticks are thinned with _thin_ticks. Ranges too narrow for 5 such ticks get 5-10
    evenly spaced ticks at a 1, 2, 2.5, 5 step instead.

    Args:
        min_val (float): Smallest finite data value.
        max_val (float): Largest finite data value.
        min_mag (float): Magnitude of the lowest tick decade.

    Returns:
        list[float]: Sorted tick values.
    """
    max_exp = math.ceil(math.log10(max(abs(min_val), abs(max_val))))
    exps = range(math.floor(math.log10(min_mag)), max_exp + 1)
    # Tolerance keeps ticks like 3 * 10**-1 = 0.30000000000000004 for data max 0.3
    lower, upper = min_val - 1e-9 * abs(min_val), max_val + 1e-9 * abs(max_val)
    crosses_zero = min_val <= 0 <= max_val

    def signed_ticks(bases: Iterable[float]) -> list[float]:
        """Positive and negative base * 10^exp ticks within the tick range."""
        mags = [base * 10**exp for exp in exps for base in bases]
        return [tick for mag in mags for tick in (mag, -mag) if lower <= tick <= upper]

    ticks = sorted(signed_ticks((1, 2, 5)) + ([0] if crosses_zero else []))
    if len(ticks) < 5:
        ticks = sorted(ticks + signed_ticks((1.5, 3, 4, 6, 7, 8, 9)))

    if len(ticks) < 5 and max_val > min_val:
        # Range narrower than a decade: evenly spaced ticks at the largest nice step
        # giving at least 5 ticks. Consecutive steps differ by at most 2x, so that
        # gives at most 10 ticks, and a step <= tick range / 10 always gives >= 5.
        top_exp = math.floor(math.log10(upper - lower))
        for exp, base in itertools.product((top_exp, top_exp - 1), (5, 2.5, 2, 1)):
            step = base * 10**exp
            tick_idx = range(math.ceil(lower / step), math.floor(upper / step) + 1)
            if len(tick_idx) >= 5:
                return [round(idx * step, max(0, 1 - exp)) for idx in tick_idx]

    return _thin_ticks(ticks, keep=0 if crosses_zero else None)


def _generate_colorbar_ticks(
    color_scale: ColorScale | dict[str, Any], values: Sequence[float]
) -> tuple[list[float] | None, list[str] | None]:
    """Generate custom tick values and text for color bars.

    This function could become redundant if plotly ever adds native support for
    arcsinh or symlog colorbar scales. Tracked in https://github.com/plotly/plotly.js/issues/221.

    Args:
        color_scale (ColorScale | dict[str, Any]): The color scale type ("linear",
            "log", "arcsinh") or a dictionary with custom scale configuration.
        values (Sequence[float]): Original (untransformed) property values.

    Returns:
        tuple[list[float] | None, list[str] | None]: Tick positions (in transformed
            color space) and their label texts. Both are None for linear scales or
            when the data has no finite values the scale can place ticks for.
    """
    scale_type = _scale_type(color_scale)
    if scale_type == "linear":  # plotly's default ticks already show the raw values
        return None, None

    finite_vals = [val for val in values if math.isfinite(val)]
    if scale_type == "log":
        positive_vals = [val for val in finite_vals if val > 0]
        if not positive_vals:
            return None, None
        min_val = min(positive_vals)
        ticks = _nice_ticks(min_val, max(positive_vals), min_mag=min_val)
        tick_vals = [math.log10(val) for val in ticks]
    elif scale_type == "arcsinh":
        nonzero_mags = [abs(val) for val in finite_vals if val != 0]
        if not nonzero_mags:
            return None, None
        # Tick decades reach at most 5 decades below the largest magnitude since
        # smaller ticks would all sit on top of 0 on the colorbar
        max_exp = math.ceil(math.log10(max(nonzero_mags)))
        min_mag = max(min(nonzero_mags), 10.0 ** (max_exp - 5))
        ticks = _nice_ticks(min(finite_vals), max(finite_vals), min_mag)
        transform_func = _arcsinh_transform(color_scale)
        tick_vals = [float(transform_func(val)) for val in ticks]
    else:
        raise ValueError(f"Unknown {scale_type=} for {color_scale=}")

    return tick_vals, list(map(_format_tick_label, ticks))


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
        hover_format (str): Python format spec (as in format(value, spec), e.g.
            ".2f", ".3g", ",.1e") for projected coordinates and numeric property
            values in hover text. Non-numeric (categorical) property values are shown
            as-is. Raises ValueError for invalid specs like printf-style "%.2f".
            (default: ".2f")
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

    try:
        format(0.0, hover_format)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{hover_format=} is not a valid Python format spec for floats, use e.g. "
            "'.2f', '.3g' or ',.1e'"
        ) from exc

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
    embedding_kwargs = embedding_kwargs or {}

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
                embeddings = embedding_fn(compositions, **embedding_kwargs)
            # Use built-in embedding methods
            elif embedding_method == "one-hot":
                embeddings = one_hot_encode(compositions, **embedding_kwargs)
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
                    **embedding_kwargs,
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
            if not isinstance(comp_str, str):
                raise TypeError(f"Expected str, got {type(comp_str)} for {comp_str}")
            # chemical_system is already alphabetically sorted
            chem_systems.append(Composition(comp_str).chemical_system)

        uniq_chem_sys = set(chem_systems)

        # 3D plots support a much more limited set of marker symbols than 2D
        validator = symbol_3d_validator if n_components == 3 else symbol_validator
        valid_symbols = list(filter(symbol_filter, validator.values))

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

    # Built-in projection methods name their columns, pre-computed coordinates and
    # custom projection functions use a generic name
    if isinstance(projection, str) and not using_precomputed_coords:
        proj_name = projection
    else:
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
    # Untransformed property values for hover text and colorbar tick labels
    hover_prop_values: list[Any] = []
    colorbar_title = None  # Initialize to handle all code paths
    scale_type = _scale_type(color_scale)

    if properties is not None:
        prop_values = hover_prop_values = properties.tolist()
        colorbar_title = prop_name

        # Apply color scale transformations if needed
        if scale_type == "log":
            if not any(val > 0 and math.isfinite(val) for val in prop_values):
                raise ValueError(
                    "color_scale='log' needs at least one positive finite value in "
                    f"{prop_name=}, got {prop_values=}"
                )
            # For log scale, replace negative or zero values with NaN
            prop_values = [
                math.log10(val) if val > 0 else float("nan") for val in prop_values
            ]
            colorbar_title = f"{prop_name} (log scale)"
        elif scale_type == "arcsinh":
            transform = _arcsinh_transform(color_scale)
            prop_values = [transform(value) for value in prop_values]
            colorbar_title = f"{prop_name} (arcsinh scale)"

        df_plot[prop_name] = prop_values  # plot transformed values

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
        hover_prop_values = [hover_prop_values[idx] for idx in sort_indices]

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

    def fmt_prop(val: Any) -> str:
        """Format numeric property values with hover_format, others as strings."""
        return format(val, hover_format) if isinstance(val, numbers.Real) else str(val)

    for idx, comp in enumerate(comp_strs):
        hover_text = f"Composition: {comp}<br>" + "".join(
            f"{method_label} {dim + 1}: {format(projected[idx, dim], hover_format)}<br>"
            for dim in range(n_components)
        )

        # Add untransformed property value or chemical system
        if prop_values is not None:
            hover_text += f"{prop_name}: {fmt_prop(hover_prop_values[idx])}"
        elif chem_systems is not None:
            hover_text += f"Chemical System: {chem_systems[idx]}"

        hover_template.append(hover_text)

    # Store hover text in the dataframe
    df_plot["hover_text"] = hover_template

    # Calculate projection statistics
    projection_stats: str | None = None
    if show_projection_stats and not using_precomputed_coords:
        if projection == "pca" and isinstance(projector, PCA):
            # Variance explained by each principal component
            var_explained_ratio = projector.explained_variance_ratio_[:n_components]
            cum_var_explained = np.cumsum(var_explained_ratio)
            projection_stats = "<br>".join(
                f"PC{idx}: {var:.1%} (cumulative: {cum_var:.1%})"
                for idx, (var, cum_var) in enumerate(
                    zip(var_explained_ratio, cum_var_explained, strict=True), start=1
                )
            )
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

    fig.update_traces(
        marker_size=marker_size / 3 if n_components == 3 else marker_size,
        hovertemplate="%{customdata[1]}<extra></extra>",
    )

    axis_titles = {
        f"{axis}axis_title": f"{method_label} {idx}"
        for idx, axis in enumerate("xyz"[:n_components], start=1)
    }
    if n_components == 2:
        fig.update_layout(**axis_titles)
    else:
        fig.update_scenes(**axis_titles)

    # Add projection stats annotation if available
    if projection_stats is not None:
        stats_kwargs = (
            show_projection_stats if isinstance(show_projection_stats, dict) else {}
        )
        stats_defaults = {
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
        fig.add_annotation(**stats_defaults | stats_kwargs)

    if prop_values is not None:
        tick_vals, tick_text = _generate_colorbar_ticks(color_scale, hover_prop_values)
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
    object.__setattr__(cluster_fig, "embeddings", embeddings)

    return cluster_fig
