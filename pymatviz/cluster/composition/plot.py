"""Plotting functions for chemical composition clustering."""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import numpy as np
import pandas as pd
from kemio.interop import is_composition_like
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from pymatviz.cluster.composition.embed import magpie_featurize, one_hot_encode
from pymatviz.cluster.composition.project import project_vectors
from pymatviz.composition import to_composition
from pymatviz.enums import LabelEnum
from pymatviz.widgets._normalize import with_defaults
from pymatviz.widgets.scatter_plot import ScatterPlotWidget
from pymatviz.widgets.scatter_plot_3d import ScatterPlot3DWidget


# Category colors for show_chem_sys="color+shape" without a property
CHEM_SYS_COLORS = ("#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A")
# matterviz (d3) marker symbols for show_chem_sys="shape" (2D only)
CHEM_SYS_SYMBOLS = (
    "Circle", "Square", "Triangle", "Diamond", "Cross", "Star", "Wye", "Plus",
    "Times", "Triangle2", "Asterisk", "Square2", "Diamond2",
)  # fmt: skip


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from sklearn.decomposition import PCA, KernelPCA
    from sklearn.manifold import TSNE, Isomap

    from pymatviz.typing import ColorType

type Projector = PCA | TSNE | Isomap | KernelPCA | None


class ClusterScatterPlot(ScatterPlotWidget):
    """2D composition cluster widget carrying the fitted ``projector`` (None for
    custom or pre-computed projections) and the ``embeddings`` that were projected.
    ``embeddings`` rows follow ``df`` order.
    """

    projector: Projector = None
    embeddings: np.ndarray | None = None


class ClusterScatterPlot3D(ScatterPlot3DWidget):
    """3D composition cluster widget with ``projector`` and ``embeddings`` (see
    ClusterScatterPlot).
    """

    projector: Projector = None
    embeddings: np.ndarray | None = None


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
    magpie = "magpie", "Magpie"


class ProjectionMethod(LabelEnum):
    """Supported projection methods for dimensionality reduction."""

    pca = "pca", "Principal Component"
    tsne = "tsne", "t-SNE Component"
    umap = "umap", "UMAP Component"
    isomap = "isomap", "Isomap Component"
    kernel_pca = "kernel_pca", "Kernel PCA Component"


ShowChemSys = Literal["color", "shape", "color+shape"]
# matterviz ColorScale: a d3 interpolator name or a {scheme, type, ...} dict
type ColorScale = str | dict[str, Any]


def _composition_strings(
    compositions: Iterable[Any], *, strict: bool = True
) -> list[str]:
    """Convert compositions to a list of formula strings."""
    comp_strs: list[str] = []
    for comp in compositions:
        if is_composition_like(comp):
            comp_strs.append(
                comp if isinstance(comp, str) else to_composition(comp).formula
            )
        elif strict:
            raise TypeError(f"Expected str or Composition, got {comp=}")
        else:
            comp_strs.append(str(comp))
    return comp_strs


def cluster_compositions(
    df: pd.DataFrame,
    *,
    composition_col: str = "composition",
    prop_name: str | None = None,
    embedding_method: str
    | EmbeddingMethod
    | EmbeddingCallable = EmbeddingMethod.magpie,
    projection: ProjectionMethod | ProjectionCallable | str,
    n_components: int = 2,
    marker_size: float = 8,
    show_chem_sys: ShowChemSys | None = None,
    color_discrete_map: Mapping[str, ColorType] | None = None,
    embedding_kwargs: dict[str, Any] | None = None,
    projection_kwargs: dict[str, Any] | None = None,
    sort: bool | int | Callable[[np.ndarray], np.ndarray] = True,
    show_projection_stats: bool = True,
    log: bool = False,
    color_scale: ColorScale = "interpolateViridis",
    annotate_points: Callable[[pd.Series], str | dict[str, Any] | None] | None = None,
    **kwargs: Any,
) -> ClusterScatterPlot | ClusterScatterPlot3D:
    """Plot chemical composition clusters with optional property coloring.

    Gives a 2D or 3D scatter plot of chemical compositions, using various
    embedding and dimensionality reduction techniques to visualize the relationships
    between different materials.

    Args:
        df (pd.DataFrame): DataFrame containing composition data and optionally
            properties and/or pre-computed embeddings.
        composition_col (str): Name of the column containing one of:
            - Chemical formulas (as strings)
            - kemio or pymatgen Composition objects
            - Pre-computed embeddings (as numpy arrays or lists)
            Default is "composition".
        prop_name (str | None): Name of the column to use for coloring points.
            If provided, the values in this column will be used to color the points.
            (default: None)
        embedding_method (str | EmbeddingMethod | Callable[[list[str], Any], ndarray]):
            Method to convert compositions to vectors (default: "magpie"). Options:
            - "one-hot": One-hot encoding of element fractions
            - "magpie": Magpie element-property features (kemio, as matminer's)
            - Callable: Custom embedding function (any other featurization, e.g.
              matminer presets) that takes (N, ) array of compositions and returns
              (N, D) array of values where D is the number of dimensions in the
              embedding space.
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
        marker_size (float): Marker diameter in px (default: 8)
        show_chem_sys (ShowChemSys | None): How to visualize chemical systems
            (one legend series per system):
            - "color": Color points by chemical system (if no property)
            - None: Don't use chemical system visualization
            - "shape": Use different marker shapes for different chemical systems
              (2D only, works best with <= 13 different chemical systems)
            - "color+shape": Use both colors and shapes to distinguish chemical
              systems (2D only)
            (default: None)
        color_discrete_map (dict[str, ColorType] | None): Optional mapping of
            categorical property values (or chemical systems, without a categorical
            property) to colors (default: None)
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
        show_projection_stats (bool): Whether to show projection statistics
            (default: True). PCA shows the variance explained by each component in
            the axis titles. In 2D, t-SNE (perplexity, learning rate), UMAP
            (n_neighbors, min_dist), Isomap (n_neighbors, metric) and Kernel PCA
            (kernel, gamma) parameters are shown in an annotation badge.
        log (bool): Log-scale property colors, leaving non-positive values
            uncolored (default: False).
        color_scale (str | dict): matterviz ColorScale for numeric property colors:
            a d3 interpolator name or a dict with keys scheme and type ("linear",
            "log", "arcsinh" for large positive and negative values, linear for
            |x| < 1, or {"type": "arcsinh", "threshold": float} to set where it turns
            logarithmic) (default: "interpolateViridis").
        annotate_points (Callable[[pd.Series], str | dict[str, Any] | None] | None):
            2D only. Function returning a text label for each point (or a matterviz
            LabelStyle dict like {"text": ..., "font_size": "10px"}, or None for no
            label). Takes a pandas Series of a df row plus the projected
            coordinates (e.g., 'pca1', 'pca2').
        **kwargs: Passed to ScatterPlotWidget or ScatterPlot3DWidget
            (depending on n_components)

    Returns:
        ClusterScatterPlot | ClusterScatterPlot3D: Scatter widget (2D or 3D) with
            attributes:
            - projector: Fitted projection object (PCA, TSNE, etc.) or None for
              custom projection functions and pre-computed coordinates
            - embeddings: Embeddings used for projection (df row order)
    """
    if n_components not in (2, 3):  # Validate inputs
        raise ValueError(f"{n_components=} must be 2 or 3")

    if show_chem_sys not in ("color", "shape", "color+shape", None):
        raise ValueError(
            f"{show_chem_sys=} must be 'color', 'shape', 'color+shape' or None"
        )
    if n_components == 3 and show_chem_sys in ("shape", "color+shape"):
        raise ValueError(
            f"{show_chem_sys=} needs marker symbols, which ScatterPlot3D lacks. Use "
            "show_chem_sys='color' or n_components=2"
        )
    if n_components == 3 and annotate_points is not None:
        raise ValueError("annotate_points needs n_components=2 (no 3D point labels)")

    if composition_col not in df:
        columns = df.columns.tolist()
        raise ValueError(f"{composition_col=} not found in DataFrame {columns=}")

    if prop_name is not None and prop_name not in df:
        raise ValueError(
            f"{prop_name=} not found in DataFrame columns: {df.columns.tolist()}"
        )

    if projection is None:
        raise ValueError(
            "projection must be specified. Choose from: "
            f"{[method.value for method in ProjectionMethod]} or provide a custom "
            "function or column name."
        )

    scale = {"scheme": color_scale} if isinstance(color_scale, str) else color_scale
    scale = {"scheme": "interpolateViridis", "type": "log" if log else "linear"} | scale
    scale_type = (
        scale["type"]["type"] if isinstance(scale["type"], dict) else scale["type"]
    )
    if scale_type not in ("linear", "log", "arcsinh") or (log and scale_type != "log"):
        raise ValueError(
            f"color scale type {scale['type']!r} must be 'linear', 'log' (or omitted "
            f"with {log=}), 'arcsinh' or a dict like {{'type': 'arcsinh', "
            "'threshold': 1}"
        )

    # Check if projection is a column name in the DataFrame
    is_method_name = isinstance(projection, str) and projection in list(
        ProjectionMethod
    )
    if is_method_name and projection in df:
        raise ValueError(
            f"df has a column named {projection!r}, which is also a projection "
            "method name; rename the column to plot its pre-computed coordinates"
        )
    using_precomputed_coords = not is_method_name and projection in df

    if using_precomputed_coords:
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
    embedding_kwargs = embedding_kwargs or {}

    compositions = df[composition_col]
    properties = None if prop_name is None else df[prop_name].infer_objects()
    # Numbers with None/pd.NA gaps (object or nullable dtypes) become float NaN
    if (
        properties is not None
        and is_numeric_dtype(properties)
        and not is_bool_dtype(properties)
    ):
        properties = properties.astype(float)

    projector: Projector = None
    embeddings: np.ndarray | None = None
    if using_precomputed_coords:
        projected = np.array([np.array(coords) for coords in df[projection]])
        comp_strs = _composition_strings(compositions, strict=False)
    else:
        if isinstance(compositions.iloc[0], (list, np.ndarray)):
            # composition_col holds pre-computed embeddings, label points by index
            comp_strs = df.index.tolist()
            embeddings = np.array([np.array(val) for val in compositions])
        else:
            comp_strs = _composition_strings(compositions)
            if isinstance(embedding_method, str) and embedding_method in df:
                embeddings = np.array([np.array(val) for val in df[embedding_method]])
            elif callable(embedding_method):
                embedding_fn = cast("EmbeddingCallable", embedding_method)
                embeddings = embedding_fn(compositions, **embedding_kwargs)
            elif embedding_method == EmbeddingMethod.one_hot:
                embeddings = one_hot_encode(compositions, **embedding_kwargs)
            elif embedding_method == EmbeddingMethod.magpie:
                embeddings = magpie_featurize(compositions, **embedding_kwargs)
            else:
                raise ValueError(
                    f"{embedding_method=} must be in "
                    f"{[method.value for method in EmbeddingMethod]}, a callable, a "
                    "column name in the DataFrame or precomputed embeddings"
                )

        if callable(projection):
            projection_fn = cast("ProjectionCallable", projection)
            projected = projection_fn(
                embeddings, n_components=n_components, **projection_kwargs
            )
        else:
            try:
                projection_method = ProjectionMethod(projection)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{projection=} must be in "
                    f"{[method.value for method in ProjectionMethod]}, "
                    f"a callable, or column name in the DataFrame"
                ) from exc
            projected, projector = project_vectors(
                embeddings,
                method=projection_method,
                n_components=n_components,
                **projection_kwargs,
            )

    proj_name = projection if is_method_name else "coordinates"
    coord_names = [f"{proj_name}{dim}" for dim in range(1, n_components + 1)]
    if using_precomputed_coords:  # For pre-computed coordinates, use generic label
        method_label = "Component"
    elif is_method_name:  # For built-in projection methods, use standardized labels
        method_label = ProjectionMethod(projection).label
    else:  # custom projection functions are labeled by name
        method_label = getattr(projection, "__name__", "Component")
        if method_label == "<lambda>":
            method_label = "Component"
    axis_labels = [f"{method_label} {dim}" for dim in range(1, n_components + 1)]

    stats_lines: list[str] = []
    if show_projection_stats and not using_precomputed_coords:
        from sklearn.decomposition import PCA

        if projection == "pca" and isinstance(projector, PCA):
            ratios = projector.explained_variance_ratio_[:n_components]
            axis_labels = [
                f"{label} ({ratio:.1%}, cumulative {cum_ratio:.1%})"
                for label, ratio, cum_ratio in zip(
                    axis_labels, ratios, np.cumsum(ratios), strict=True
                )
            ]
        elif projector is not None and isinstance(projection, str):
            stat_params = {
                "tsne": ("perplexity", "learning_rate"),
                "umap": ("n_neighbors", "min_dist"),
                "isomap": ("n_neighbors", "metric"),
                "kernel_pca": ("kernel", "gamma"),
            }[projection]
            for param in stat_params:
                # Prefer fitted values (e.g. learning_rate_ for learning_rate="auto")
                value = getattr(projector, f"{param}_", getattr(projector, param))
                stats_lines.append(
                    f"{param} = {value!r}"
                    if isinstance(value, str)
                    else f"{param} = {value:.4g}"
                )

    # Chemical systems for color/shape grouping and point metadata
    chem_systems: list[str] | None = None
    if show_chem_sys is not None:
        for comp_str in comp_strs:
            if not isinstance(comp_str, str):
                raise TypeError(f"Expected str, got {type(comp_str)} for {comp_str}")
        # chemical_system is already alphabetically sorted
        chem_systems = [to_composition(comp).chemical_system for comp in comp_strs]
        n_systems = len(set(chem_systems))
        if "shape" in show_chem_sys and n_systems > len(CHEM_SYS_SYMBOLS):
            warnings.warn(
                f"Number of unique chemical systems ({n_systems}) exceeds available "
                f"marker symbols ({len(CHEM_SYS_SYMBOLS)}). Some systems will share a "
                "symbol.",
                UserWarning,
                stacklevel=2,
            )

    # Numeric properties color points on a continuous scale, others split points
    # into one series per category
    numeric_props: np.ndarray | None = None
    categories: list[str] | None = None
    if properties is not None:
        if is_numeric_dtype(properties) and not is_bool_dtype(properties):
            numeric_props = properties.to_numpy(dtype=float)
            if scale_type == "log" and not np.any(
                (numeric_props > 0) & np.isfinite(numeric_props)
            ):
                raise ValueError(
                    "Log color scale needs at least one positive finite value in "
                    f"{prop_name=}, got {numeric_props.tolist()}"
                )
        else:
            # one "nan" category for missing values, whichever missing marker
            # (None, NaN, pd.NA) the pandas version stores in object/str columns
            categories = ["nan" if pd.isna(val) else str(val) for val in properties]

    plot_order = np.arange(len(df))  # df position of each plotted point
    if sort and numeric_props is not None:
        if callable(sort):
            plot_order = np.asarray(sort(numeric_props))
            if plot_order.dtype.kind not in "iu" or not np.array_equal(
                np.sort(plot_order), np.arange(len(df))
            ):
                raise ValueError(
                    f"sort callable must return a permutation of indices, got "
                    f"{plot_order.tolist()}"
                )
        elif isinstance(sort, int):  # Includes True; False is handled above.
            plot_order = np.argsort(numeric_props, kind="stable")
            if sort < 0:
                plot_order = plot_order[::-1]
        else:
            raise TypeError(f"Invalid sort parameter type: {type(sort).__name__}")

    def finite_or_none(value: Any) -> Any:
        """JSON-safe point values: non-finite floats and missing markers as None."""
        if value is pd.NA or (
            isinstance(value, float | np.floating) and not math.isfinite(value)
        ):
            return None
        return value.item() if isinstance(value, np.generic) else value

    def point_metadata(pos: int) -> dict[str, Any]:
        """Tooltip metadata of the point at df position pos."""
        meta: dict[str, Any] = {"composition": str(comp_strs[pos])}
        if properties is not None:
            meta[str(prop_name)] = finite_or_none(properties.iloc[pos])
        if chem_systems is not None:
            meta["chem_system"] = chem_systems[pos]
        return meta

    point_labels: dict[int, dict[str, Any]] = {}
    if callable(annotate_points):
        # Rows are df's (raw values, original index label in row.name) plus the
        # projected coordinates
        df_rows = df.copy()
        for dim, name in enumerate(coord_names):
            df_rows[name] = projected[:, dim]
        for pos in range(len(df)):
            label = annotate_points(df_rows.iloc[pos])
            if label:
                point_labels[pos] = (
                    {"text": label} if isinstance(label, str) else dict(label)
                )

    # Group points (in plot order) into series by category and/or chemical system
    by_system = chem_systems is not None and (
        "shape" in str(show_chem_sys)
        or (show_chem_sys == "color" and categories is None and numeric_props is None)
    )
    groups: dict[tuple[str | None, str | None], list[int]] = {}
    for pos in plot_order:
        category = None if categories is None else categories[pos]
        system = chem_systems[pos] if by_system else None
        groups.setdefault((category, system), []).append(int(pos))
    ordered_systems = sorted(set(chem_systems or []))
    # series colors keyed by category (categorical property) or chemical system
    color_keys: list[str] = []
    if categories is not None:
        color_keys = sorted(set(categories))
    elif show_chem_sys == "color+shape" and properties is None:
        color_keys = ordered_systems
    color_map = {
        key: CHEM_SYS_COLORS[idx % len(CHEM_SYS_COLORS)]
        for idx, key in enumerate(color_keys)
    } | dict(color_discrete_map or {})

    series: list[dict[str, Any]] = []
    for (category, system), positions in sorted(
        groups.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))
    ):
        point_style: dict[str, Any] = {"radius": marker_size / 2}
        if system is not None and "shape" in str(show_chem_sys):
            point_style["symbol_type"] = CHEM_SYS_SYMBOLS[
                ordered_systems.index(system) % len(CHEM_SYS_SYMBOLS)
            ]
        srs: dict[str, Any] = {
            axis: projected[positions, dim]
            for dim, axis in enumerate("xyz"[:n_components])
        } | {
            "markers": "points",
            "point_style": point_style,
            "metadata": [point_metadata(pos) for pos in positions],
        }
        if label := " · ".join(part for part in (category, system) if part):
            srs["label"] = label
        if (color_key := category or system) in color_map:
            srs["color"] = str(color_map[color_key])
        if numeric_props is not None:
            values = numeric_props[positions]
            if scale_type == "log":
                values = np.where(values > 0, values, np.nan)
            srs["color_values"] = [finite_or_none(val) for val in values.tolist()]
        if point_labels:
            srs["point_label"] = [point_labels.get(pos, {}) for pos in positions]
        series.append(srs)

    axes = {
        f"{axis}_axis": {"label": label}
        for axis, label in zip("xyz", axis_labels, strict=False)
    }
    options: dict[str, Any] = axes | {"show_legend": len(series) > 1}
    if numeric_props is not None:
        options["color_scale"] = scale
    if n_components == 2:
        if numeric_props is not None:
            options["color_bar"] = {"title": prop_name}
        if stats_lines:
            options["annotation"] = "\n".join(stats_lines)
        widget: ClusterScatterPlot | ClusterScatterPlot3D = ClusterScatterPlot(
            series=series, **with_defaults(options, kwargs)
        )
    else:
        widget = ClusterScatterPlot3D(series=series, **with_defaults(options, kwargs))
    widget.projector = projector
    widget.embeddings = embeddings
    return widget
