"""Cluster material compositions: Magpie or one-hot embeddings, projected to 2D/3D
and plotted.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import numpy as np
import pandas as pd
from kemio.interop import is_composition_like
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from pymatviz.compositions import to_composition
from pymatviz.enums import LabelEnum
from pymatviz.widgets._normalize import with_defaults
from pymatviz.widgets.scatter_plot import ScatterPlotWidget
from pymatviz.widgets.scatter_plot_3d import ScatterPlot3DWidget


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from kemio.interop import CompositionLike
    from numpy.typing import ArrayLike
    from sklearn.decomposition import PCA, KernelPCA
    from sklearn.manifold import TSNE, Isomap


def magpie_featurize(
    compositions: Sequence[CompositionLike] | pd.Series,
    *,
    normalize: bool = False,
    feature_subset: Sequence[str] | None = None,
) -> np.ndarray:
    """Magpie element-property features of chemical formulas or compositions
    (matminer's ElementProperty "magpie" preset, computed by
    kemio.composition_features). For other featurizations, pass a feature matrix
    or callable to cluster_compositions.

    Args:
        compositions (Sequence[CompositionLike] | pd.Series): chemical
            formulas as strings or kemio/pymatgen Composition objects.
        normalize (bool): If True, L2-normalize each feature vector (default: False).
            Magpie-style features mix units (e.g. kelvin and eV), so a row norm is
            dominated by the largest-magnitude ones (melting temperature) and divides
            every feature by it; column standardization in project_vectors already
            puts features on a common scale.
        feature_subset (Sequence[str] | None): Feature labels to keep, in this order
            (see kemio.composition_feature_labels). Defaults to all 132.

    Returns:
        np.ndarray: Array of shape (n_compositions, n_features), missing values as 0.

    Raises:
        ValueError: For unknown feature labels.
    """
    import kemio

    features = kemio.composition_features(
        [to_composition(comp) for comp in compositions]
    )
    if feature_subset is not None:
        labels = kemio.composition_feature_labels()
        if unknown := [label for label in feature_subset if label not in labels]:
            raise ValueError(f"Unknown Magpie features {unknown}, expected {labels}")
        features = features[:, [labels.index(label) for label in feature_subset]]
    features = np.nan_to_num(features)
    if normalize:
        import sklearn.preprocessing

        features = sklearn.preprocessing.normalize(features, norm="l2", axis=1)
    return features


def one_hot_encode(
    compositions: Sequence[CompositionLike] | pd.Series,
    *,
    normalize: bool = True,
    elements: Sequence[str] | None = None,
) -> np.ndarray:
    """One-hot encode chemical formulas or compositions.

    Args:
        compositions (Sequence[CompositionLike] | pd.Series): chemical
            formulas as strings or kemio/pymatgen Composition objects.
        normalize (bool): Whether to normalize the feature vectors to unit norm
            (default: True)
        elements (Sequence[str] | None): Element symbols to include in the one-hot
            encoding. If None (default), all elements in the periodic table will be
            used. Compositions with elements outside this list raise ValueError.

    Returns:
        np.ndarray: Array of shape (n_compositions, n_elements) with one-hot encoded
            compositions where each row corresponds to a composition and each column
            to an element's fraction in that composition.
    """
    if elements is None:
        from pymatviz.utils import df_ptable

        elements = list(df_ptable.index)
    elements = list(elements)
    column_of = {element: idx for idx, element in enumerate(elements)}
    if len(column_of) != len(elements):
        raise ValueError(f"duplicate entries in {elements=}")

    comp_objs = [to_composition(comp) for comp in compositions]
    features = np.zeros((len(comp_objs), len(elements)))

    for idx, comp in enumerate(comp_objs):
        comp_dict = comp.fractional_composition.as_dict()
        if missing := sorted(set(comp_dict) - set(column_of)):
            raise ValueError(
                f"{comp.formula} contains {missing}, which are not in {elements=}"
            )
        for element, fraction in comp_dict.items():
            features[idx, column_of[element]] = fraction

    if normalize:
        import sklearn.preprocessing

        features = sklearn.preprocessing.normalize(features, norm="l2", axis=1)

    return features


def project_vectors(
    data: ArrayLike,
    *,
    method: ProjectionMethod | str = "pca",
    n_components: int = 2,
    random_state: int | None = 42,
    scale_data: bool = True,
    **kwargs: Any,
) -> tuple[np.ndarray, PCA | TSNE | Isomap | KernelPCA | Any]:
    """Project high-dimensional data to lower dimensions using various methods.

    Args:
        data (ArrayLike): Input data of shape (n_samples, n_features)
        method ("pca" | "tsne" | "umap" | "isomap" | "kernel_pca"): Projection
            method to use (see ProjectionMethod enum, default: "pca")
        n_components (int): Projection dimensions (2 or 3) (default: 2)
        random_state (int | None): Random seed for reproducibility
        scale_data (bool): Whether to standardize input columns before projection
            and rescale the projected coordinates to zero mean and unit overall
            variance (one shared factor for all axes, so aspect ratios are kept)
        **kwargs: Additional arguments passed to the projection method

    Returns:
        tuple[np.array, PCA | TSNE | Isomap | KernelPCA | Any]: A tuple containing:
            - Projected data array of shape (n_samples, n_components)
            - The fitted projection object (PCA, TSNE, UMAP, Isomap, or KernelPCA)

    Raises:
        ValueError: If method is invalid or n_components is too small
        ImportError: If UMAP is requested but not installed
    """
    from sklearn.decomposition import PCA, KernelPCA
    from sklearn.manifold import TSNE, Isomap

    if n_components < 2:
        raise ValueError("n_components must be at least 2")
    data = np.asarray(data, dtype=float)

    n_samples = data.shape[0]
    if n_samples < n_components:
        raise ValueError(f"Need at least {n_components=} samples, got {n_samples}")

    if scale_data:  # standardize columns (std=0 kept as is), NaNs become 0
        std = np.nanstd(data, axis=0)
        data = (data - np.nanmean(data, axis=0)) / np.where(std == 0, 1, std)
        data = np.nan_to_num(data, nan=0.0)

    if method == "pca":
        # Exact SVD by default: svd_solver="auto" silently switches to randomized SVD
        # for e.g. 636x118 inputs, which rotates near-degenerate components.
        pca_kwargs = {"svd_solver": "full"} | kwargs
        reducer = PCA(
            n_components=n_components, random_state=random_state, **pca_kwargs
        )

    elif method == "tsne":
        if n_components > 3:
            raise ValueError("t-SNE visualization typically uses 2 or 3 components")
        # Adjust perplexity for small datasets
        default_perplexity = min(30, n_samples / 3)  # t-SNE default is 30
        tsne_kwargs = {
            "n_components": n_components,
            "perplexity": default_perplexity,
            "learning_rate": "auto",
            "init": "pca",
            "random_state": random_state,
        } | kwargs
        reducer = TSNE(**tsne_kwargs)

    elif method == "umap":
        if n_samples < n_components + 2:
            raise ValueError(
                f"UMAP needs at least n_components + 2 = {n_components + 2} samples, "
                f"got {n_samples}"
            )
        try:
            from umap import UMAP
        except ImportError:
            raise ImportError(
                "UMAP requires the 'umap-learn' package: pip install umap-learn"
            ) from None
        reducer = UMAP(
            n_components=n_components,
            random_state=random_state,
            **kwargs,
        )

    elif method == "isomap":
        # Adjust n_neighbors for small datasets
        n_neighbors = kwargs.pop("n_neighbors", 5)
        if n_samples < 10 and n_neighbors > n_samples / 2:
            # Isomap needs n_neighbors < n_samples
            n_neighbors = min(max(2, int(n_samples / 2)), n_samples - 1)
            warnings.warn(
                f"Isomap: adjusted to {n_neighbors=} for {n_samples=}",
                UserWarning,
                stacklevel=2,
            )
        reducer = Isomap(
            n_components=n_components,
            n_neighbors=n_neighbors,
            **kwargs,
        )

    elif method == "kernel_pca":
        reducer = KernelPCA(
            n_components=n_components,
            kernel=kwargs.pop("kernel", "rbf"),
            random_state=random_state,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown projection {method=}")

    projected_data = reducer.fit_transform(data)
    if scale_data:
        # One scale for all axes: per-axis standardization would stretch e.g. PC2 to
        # look as spread out as PC1.
        projected_data = projected_data - projected_data.mean(axis=0)
        scale = np.sqrt(np.mean(projected_data**2))
        projected_data = projected_data / (scale if scale > 0 else 1)
    return projected_data, reducer


# Category colors for show_chem_sys="color+shape" without a property
CHEM_SYS_COLORS = ("#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A")
# matterviz (d3) marker symbols for show_chem_sys="shape" (2D only)
CHEM_SYS_SYMBOLS = (
    "Circle", "Square", "Triangle", "Diamond", "Cross", "Star", "Wye", "Plus",
    "Times", "Triangle2", "Asterisk", "Square2", "Diamond2",
)  # fmt: skip

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
    color_discrete_map: Mapping[str, str] | None = None,
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
        color_discrete_map (dict[str, str] | None): Optional mapping of categorical
            property values (or chemical systems, without a categorical property) to
            CSS colors (default: None)
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
    if df.empty:
        raise ValueError("df has no rows to cluster")

    if prop_name is not None and prop_name not in df:
        raise ValueError(
            f"{prop_name=} not found in DataFrame columns: {df.columns.tolist()}"
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
    method_names = list(ProjectionMethod)
    is_method_name = isinstance(projection, str) and projection in method_names
    if is_method_name and projection in df:
        raise ValueError(
            f"df has a column named {projection!r}, which is also a projection "
            "method name; rename the column to plot its pre-computed coordinates"
        )
    using_precomputed_coords = not is_method_name and projection in df

    if using_precomputed_coords and (
        bad_coords := [
            coords
            for coords in df[projection]
            if not isinstance(coords, list | tuple | np.ndarray)
            or len(coords) != n_components
        ]
    ):
        raise ValueError(
            f"Column {projection!r} must hold a list or array of {n_components} "
            f"coordinates per row, got {len(bad_coords)} rows like {bad_coords[0]!r}"
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
            positive_finite = (numeric_props > 0) & np.isfinite(numeric_props)
            if scale_type == "log" and not positive_finite.any():
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
            is_int = plot_order.dtype.kind in "iu"
            if not (is_int and np.array_equal(np.sort(plot_order), np.arange(len(df)))):
                raise ValueError(
                    f"sort callable must return a permutation of indices, got "
                    f"{plot_order.tolist()}"
                )
        elif isinstance(sort, int | np.integer | np.bool_):  # False skipped above
            plot_order = np.argsort(numeric_props, kind="stable")
            if sort < 0:
                plot_order = plot_order[::-1]
        else:
            raise TypeError(f"Invalid sort parameter type: {type(sort).__name__}")

    def finite_or_none(value: Any) -> Any:
        """JSON-safe point values: non-finite floats and missing markers as None."""
        non_finite = isinstance(value, float | np.floating) and not math.isfinite(value)
        if value is pd.NA or non_finite:
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
    no_color_prop = categories is None and numeric_props is None
    color_by_system = show_chem_sys == "color" and no_color_prop
    shape_by_system = "shape" in str(show_chem_sys)
    by_system = chem_systems is not None and (shape_by_system or color_by_system)
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
    n_colors = len(CHEM_SYS_COLORS)
    cycled_colors = {
        key: CHEM_SYS_COLORS[idx % n_colors] for idx, key in enumerate(color_keys)
    }
    color_map = cycled_colors | dict(color_discrete_map or {})

    series: list[dict[str, Any]] = []
    for (category, system), positions in sorted(
        groups.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))
    ):
        point_style: dict[str, Any] = {"radius": marker_size / 2}
        if system is not None and shape_by_system:
            symbol_idx = ordered_systems.index(system) % len(CHEM_SYS_SYMBOLS)
            point_style["symbol_type"] = CHEM_SYS_SYMBOLS[symbol_idx]

        xyz_axes = "xyz"[:n_components]
        coords = {axis: projected[positions, dim] for dim, axis in enumerate(xyz_axes)}
        metadata = [point_metadata(pos) for pos in positions]
        srs: dict[str, Any] = {
            **coords,
            "markers": "points",
            "point_style": point_style,
            "metadata": metadata,
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
