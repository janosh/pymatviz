"""Unit tests for composition cluster plots."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytest
from pymatgen.core import Composition
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, Isomap

import pymatviz as pmv
from pymatviz.cluster.composition.plot import ClusterFigure, ColorScale


if TYPE_CHECKING:
    from collections.abc import Callable

    from pymatviz.cluster.composition import ProjectionCallable, ProjectionMethod
    from pymatviz.cluster.composition.plot import ShowChemSys

np_rng = np.random.default_rng(seed=0)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame with compositions and properties."""
    df_sample = pd.DataFrame(
        {"composition": ["Fe0.5Co0.5", "Ni0.7Cu0.3", "Zr0.6Ti0.4"]}
    )
    df_sample["property"] = np.array([1, 10, 100])
    return df_sample


@pytest.fixture
def df_comp() -> pd.DataFrame:
    """Base DataFrame with only compositions for testing."""
    return pd.DataFrame({"composition": ["Fe2O3", "Al2O3", "Cu", "Au", "Ag"]})


@pytest.fixture
def df_prop(df_comp: pd.DataFrame) -> pd.DataFrame:
    """DataFrame with compositions and property values for testing."""
    df_prop = df_comp.copy()
    df_prop["property"] = [1.0, 2.0, 3.0, 4.0, 5.0]
    return df_prop


@pytest.fixture
def sort_df(df_comp: pd.DataFrame) -> pd.DataFrame:
    """DataFrame with compositions and property values for sort testing."""
    df_sort = df_comp.copy()
    df_sort["property"] = [3.0, 1.0, 4.0, 2.0, 5.0]
    return df_sort


@pytest.fixture
def hover_alignment_df(df_comp: pd.DataFrame) -> pd.DataFrame:
    """DataFrame for testing hover text alignment."""
    df_hover = df_comp.copy()
    df_hover["property"] = [30.0, 10.0, 50.0, 20.0, 40.0]
    return df_hover


@pytest.fixture
def df_with_embeddings(hover_alignment_df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame with pre-computed embeddings."""
    df_embed = hover_alignment_df.copy()
    df_embed["my_embeddings"] = list(np_rng.random((len(df_embed), 10)))
    return df_embed


@pytest.fixture
def custom_coords_2d(df_prop: pd.DataFrame) -> np.ndarray:
    """Pre-computed 2D coordinates for testing."""
    return np_rng.random((len(df_prop), 2))


@pytest.fixture
def custom_coords_3d(df_prop: pd.DataFrame) -> np.ndarray:
    """Pre-computed 3D coordinates for testing."""
    return np_rng.random((len(df_prop), 3))


def hover_texts_by_comp(fig: go.Figure) -> dict[str, str]:
    """Map each point's composition to its hover text across all traces."""
    return {data[0]: data[1] for trace in fig.data for data in trace.customdata}


@pytest.mark.parametrize(
    ("show_chem_sys", "prop_name", "expected_traces", "check_symbol"),
    [
        (None, None, 1, False),  # No chem system, no property
        (None, "property", 1, False),  # No chem system, with property
        ("shape", None, 1, True),  # Shape mode, no property
        ("shape", "property", 1, True),  # Shape mode, with property
        ("color", None, 3, False),  # Color mode, no property
        # Color mode, with property (overrides color mode)
        ("color", "property", 1, False),
        ("color+shape", None, 1, True),  # Color+shape mode, no property
        ("color+shape", "property", 1, True),  # Color+shape mode, with property
    ],
)
def test_chemical_system_visualization(
    sample_df: pd.DataFrame,
    show_chem_sys: ShowChemSys | None,
    prop_name: str | None,
    expected_traces: int,
    check_symbol: bool,
) -> None:
    """Test different chemical system viz options with and without properties."""
    fig = pmv.cluster_compositions(
        df_in=sample_df,
        embedding_method="one-hot",
        projection="pca",
        show_chem_sys=show_chem_sys,
        prop_name=prop_name,
    )

    assert isinstance(fig, ClusterFigure)
    assert len(fig.data) == expected_traces
    assert {trace.type for trace in fig.data} == {"scatter"}
    assert sum(len(trace.x) for trace in fig.data) == len(sample_df)

    # Get data dictionary for the first trace
    data_dict = fig.to_dict()["data"][0]

    # Check if property coloring is applied when prop_name is provided
    if prop_name:
        assert data_dict["marker"]["coloraxis"] == "coloraxis"
        assert fig.layout.coloraxis.colorbar.title.text == prop_name

    # Check if symbols are properly applied
    if check_symbol:
        symbols = data_dict["marker"]["symbol"]
        assert isinstance(symbols, list)
        assert len(set(symbols)) >= min(len(sample_df), 3)

    # Check specific behavior for color mode without properties
    if show_chem_sys == "color" and not prop_name:
        # Each trace should have distinct color
        colors = [trace.marker.color for trace in fig.data]
        assert len(set(colors)) == 3  # Three unique colors

        # Each trace should contain 1 point (one composition per chem system)
        for trace in fig.data:
            assert len(trace.x) == 1
            assert len(trace.y) == 1

        # Ensure colors follow standard plotly palette (case-insensitive)
        for color in colors:
            assert color.upper() in px.colors.qualitative.Plotly[:3]

    # Check for color+shape mode without properties
    if show_chem_sys == "color+shape" and not prop_name:
        # Check that marker has multiple colors (one per chemical system)
        assert isinstance(fig.data[0].marker.color, (list, tuple))
        # Each chemical system should have a different color
        assert len(fig.data[0].marker.color) == len(sample_df)

        # The colors should be standard Plotly colors (case-insensitive)
        for color in fig.data[0].marker.color:
            assert color in px.colors.qualitative.Plotly[:5]


@pytest.mark.parametrize(
    ("sort_value", "expected_order"),
    [
        (True, [1, 3, 0, 2, 4]),
        (1, [1, 3, 0, 2, 4]),
        (False, [0, 1, 2, 3, 4]),
        (0, [0, 1, 2, 3, 4]),
        (-1, [4, 2, 0, 3, 1]),
        pytest.param(
            lambda _values: np.array([2, 0, 4, 1, 3]),
            [2, 0, 4, 1, 3],
            id="custom_sort_func",
        ),
    ],
)
def test_sorting_options(
    sort_df: pd.DataFrame,
    sort_value: bool | int | Callable[[np.ndarray], np.ndarray],
    expected_order: list[int],
) -> None:
    """Sort options reorder points while hover text stays aligned with each row."""
    fig = pmv.cluster_compositions(
        df_in=sort_df,
        prop_name="property",
        embedding_method="one-hot",
        projection="pca",
        sort=sort_value,
    )

    compositions = [comp for comp, _hover in fig.data[0].customdata]
    assert compositions == sort_df["composition"].iloc[expected_order].tolist()
    assert (
        list(fig.data[0].marker.color)
        == sort_df["property"].iloc[expected_order].tolist()
    )

    prop_by_comp = dict(zip(sort_df["composition"], sort_df["property"], strict=True))
    for comp, hover_text in fig.data[0].customdata:
        assert hover_text.startswith(f"Composition: {comp}<br>")
        assert hover_text.endswith(f"<br>property: {prop_by_comp[comp]:.2f}")


@pytest.mark.parametrize("show_chem_sys", ["color", "color+shape"])
def test_color_discrete_map(
    sample_df: pd.DataFrame, show_chem_sys: ShowChemSys
) -> None:
    """Custom color_discrete_map colors chemical systems in color(+shape) modes."""
    custom_colors = {"Co-Fe": "#ff0000", "Cu-Ni": "#00ff00", "Ti-Zr": "#0000ff"}
    fig = pmv.cluster_compositions(
        df_in=sample_df[["composition"]],
        embedding_method="one-hot",
        projection="pca",
        show_chem_sys=show_chem_sys,
        color_discrete_map=custom_colors,
    )

    if show_chem_sys == "color":  # one trace per chemical system
        assert {trace.name: trace.marker.color for trace in fig.data} == custom_colors
    else:  # single trace with per-point colors in input order
        assert list(fig.data[0].marker.color) == list(custom_colors.values())


def test_default_marker_color(sample_df: pd.DataFrame) -> None:
    """Without property or chemical system coloring, use plotly's default color."""
    fig = pmv.cluster_compositions(
        df_in=sample_df[["composition"]],
        embedding_method="one-hot",
        projection="pca",
        show_chem_sys=None,
    )
    assert len(fig.data) == 1
    assert fig.data[0].marker.color == "#636efa"


@pytest.mark.parametrize(
    ("projection", "projection_kwargs", "expected_texts"),
    [
        ("pca", {}, ["PC1", "cumulative"]),
        (
            "tsne",
            {"perplexity": 1.0, "learning_rate": "auto"},
            ["perplexity = 1.0", "learning_rate = 'auto'"],
        ),
        (
            "isomap",
            {"n_neighbors": 2, "metric": "euclidean"},
            ["n_neighbors = 2", "metric = 'euclidean'"],
        ),
        (
            "kernel_pca",
            {"kernel": "rbf", "gamma": 0.1},
            ["kernel = 'rbf'", "gamma = 0.1"],
        ),
        pytest.param(
            "tsne",
            {},
            ["perplexity = 1.6666666666666667", "learning_rate = 'auto'"],
            id="tsne-defaults",
        ),
        pytest.param(
            "isomap",
            {},
            ["n_neighbors = 2", "metric = 'minkowski'"],
            id="isomap-defaults",
        ),
        pytest.param(
            "isomap",
            {"n_neighbors": 4},
            ["n_neighbors = 2"],
            id="isomap-adjusted-neighbors",
        ),
        pytest.param(
            "kernel_pca",
            {},
            ["kernel = 'rbf'", "gamma = None"],
            id="kernel-pca-defaults",
        ),
        ("umap", {"n_neighbors": 15, "min_dist": 0.1}, ["n_neighbors", "min_dist"]),
    ],
)
def test_projection_stats(
    df_prop: pd.DataFrame,
    projection: ProjectionMethod,
    projection_kwargs: dict[str, Any],
    expected_texts: list[str],
) -> None:
    """Test projection statistics display for different methods."""
    if projection == "umap":
        pytest.importorskip("umap")

    fig = pmv.cluster_compositions(
        df_in=df_prop,
        embedding_method="one-hot",
        projection=projection,
        projection_kwargs=projection_kwargs,
        show_projection_stats=True,
    )

    # Check that all expected texts are present in at least one annotation
    for expected_text in expected_texts:
        assert any(expected_text in ann.text for ann in fig.layout.annotations)

    # For PCA, stats must report the fitted projector's explained variance
    if projection == "pca":
        assert isinstance(fig.projector, PCA)
        pc1, pc2 = fig.projector.explained_variance_ratio_
        assert pc1 >= pc2
        expected_stats = (
            f"PC1: {pc1:.1%} (cumulative: {pc1:.1%})<br>"
            f"PC2: {pc2:.1%} (cumulative: {pc1 + pc2:.1%})"
        )
        assert fig.layout.annotations[0].text == expected_stats


@pytest.mark.parametrize(
    ("kwargs", "err_type", "match"),
    [
        ({"n_components": 4}, ValueError, "n_components=4 must be 2 or 3"),
        ({"embedding_method": "invalid_method"}, ValueError, "embedding_method="),
        (
            {"df_in": pd.DataFrame({"composition": [object(), object()]})},
            TypeError,
            "Expected str or Composition",
        ),
        *[
            (
                {"show_projection_stats": val},
                TypeError,
                re.escape(f"show_projection_stats={val!r} must be bool or dict"),
            )
            for val in (1, (1,), "foo")
        ],
        *[
            (
                {"hover_format": fmt},
                ValueError,
                re.escape(f"hover_format={fmt!r} is not a valid Python format spec"),
            )
            for fmt in ("%.2f", "$%.2f", "{:.2f}", ",.3~s", "d")
        ],
        (
            {
                "df_in": pd.DataFrame({"composition": ["Fe", "Cu"], "prop": [-1, 0]}),
                "prop_name": "prop",
                "color_scale": {"type": "log"},
            },
            ValueError,
            "color_scale='log' needs at least one positive finite value",
        ),
    ],
)
def test_cluster_compositions_invalid_inputs(
    df_comp: pd.DataFrame,
    kwargs: dict[str, Any],
    err_type: type[Exception],
    match: str,
) -> None:
    """Invalid arguments raise informative errors."""
    defaults: dict[str, Any] = dict(
        df_in=df_comp, embedding_method="one-hot", projection="pca"
    )
    with pytest.raises(err_type, match=match):
        pmv.cluster_compositions(**defaults | kwargs)


@pytest.mark.parametrize(
    ("embedding_method", "projection", "expected_label"),
    [
        ("magpie", "pca", "Principal Component"),
        ("one-hot", "tsne", "t-SNE Component"),
        ("deml", "pca", "Principal Component"),
        ("matminer", "isomap", "Isomap Component"),
        ("matscholar_el", "kernel_pca", "Kernel PCA Component"),
    ],
)
def test_cluster_compositions_methods(
    sample_df: pd.DataFrame,
    embedding_method: str,
    projection: str,
    expected_label: str,
) -> None:
    """Test combinations of embedding and projection methods in 2D and 3D."""
    for n_components in (2, 3):
        fig = pmv.cluster_compositions(
            df_in=sample_df,
            prop_name="property",
            embedding_method=embedding_method,
            projection=projection,
            n_components=n_components,
        )

        assert isinstance(fig, ClusterFigure)
        assert len(fig.data) == 1
        trace = fig.data[0]
        axis_layout = fig.layout if n_components == 2 else fig.layout.scene
        assert trace.type == ("scatter" if n_components == 2 else "scatter3d")
        for idx, axis in enumerate("xyz"[:n_components], start=1):
            assert getattr(trace, axis).shape == (len(sample_df),)
            axis_title = getattr(axis_layout, f"{axis}axis").title.text
            assert axis_title == f"{expected_label} {idx}"

    # with only 3 samples, non-PCA methods may collapse the 3rd dimension
    if projection == "pca":
        assert np.var(fig.data[0].z) > 0


@pytest.mark.parametrize(
    ("custom_embedding", "custom_projection", "with_kwargs"),
    [
        (True, False, False),  # Only custom embedding
        (False, True, False),  # Only custom projection
        (True, True, False),  # Both custom functions
        (True, True, True),  # Both with custom kwargs
    ],
)
def test_custom_embedding_projection(
    sample_df: pd.DataFrame,
    custom_embedding: bool,
    custom_projection: bool,
    with_kwargs: bool,
) -> None:
    """Custom embedding/projection functions receive kwargs and set fig metadata."""
    emb_scale, proj_scale = (2.0, 3.0) if with_kwargs else (1.0, 1.0)

    def custom_embedding_func(
        compositions: list[str], scale: float = 1.0, **_kwargs: Any
    ) -> np.ndarray:
        """Custom embedding returning a deterministic scaled array."""
        return np.arange(len(compositions) * 4, dtype=float).reshape(-1, 4) ** 2 * scale

    def custom_projection_func(
        embeddings: np.ndarray, n_components: int, scale: float = 1.0, **_kwargs: Any
    ) -> np.ndarray:
        """Custom projection returning the scaled first n_components features."""
        return embeddings[:, :n_components] * scale

    fig = pmv.cluster_compositions(
        df_in=sample_df,
        prop_name="property",
        embedding_method=custom_embedding_func if custom_embedding else "one-hot",  # ty: ignore[invalid-argument-type]
        projection=custom_projection_func if custom_projection else "pca",
        embedding_kwargs={"scale": emb_scale} if with_kwargs else {},
        projection_kwargs={"scale": proj_scale} if with_kwargs else {},
    )

    assert isinstance(fig, ClusterFigure)
    assert len(fig.data) == 1
    assert fig.data[0].x.shape == (3,)
    assert isinstance(fig.embeddings, np.ndarray)
    if custom_embedding:
        expected_emb = custom_embedding_func(sample_df["composition"], scale=emb_scale)
        np.testing.assert_array_equal(fig.embeddings, expected_emb)
    if custom_projection:
        assert fig.projector is None  # custom projectors store None
        assert fig.layout.xaxis.title.text == "custom_projection_func 1"
        np.testing.assert_array_equal(fig.data[0].x, fig.embeddings[:, 0] * proj_scale)
        np.testing.assert_array_equal(fig.data[0].y, fig.embeddings[:, 1] * proj_scale)
    else:
        assert isinstance(fig.projector, PCA)


@pytest.mark.parametrize("n_components", [2, 3])
@pytest.mark.parametrize("show_chem_sys", [None, "color", "shape", "color+shape"])
def test_marker_size_and_title(
    sample_df: pd.DataFrame, n_components: int, show_chem_sys: ShowChemSys | None
) -> None:
    """Marker size is scaled down for 3D plots and extra kwargs reach px.scatter."""
    marker_size = 12
    fig = pmv.cluster_compositions(
        df_in=sample_df,
        prop_name="property",
        embedding_method="one-hot",
        projection="pca",
        n_components=n_components,
        marker_size=marker_size,
        show_chem_sys=show_chem_sys,
        title="Custom Plot",
    )
    assert fig.layout.title.text == "Custom Plot"
    expected_size = marker_size / (3 if n_components == 3 else 1)
    for trace in fig.data:
        assert trace.marker.size == expected_size


@pytest.mark.parametrize(
    ("fixture_name", "sort_option", "viz_mode", "prop_name", "embedding_method"),
    [
        ("hover_alignment_df", True, "color", None, "one-hot"),
        ("df_with_embeddings", True, None, "property", "my_embeddings"),
        ("df_with_embeddings", False, "shape", "property", "my_embeddings"),
        ("df_with_embeddings", True, "color+shape", "property", "my_embeddings"),
    ],
)
def test_hover_text_alignment(
    request: pytest.FixtureRequest,
    fixture_name: str,
    sort_option: bool,
    viz_mode: ShowChemSys | None,
    prop_name: str | None,
    embedding_method: str,
) -> None:
    """Hover text matches its point for chem-sys modes and embedding columns."""
    df_data = request.getfixturevalue(fixture_name)
    fig = pmv.cluster_compositions(
        df_in=df_data,
        prop_name=prop_name,
        embedding_method=embedding_method,
        projection="pca",
        sort=sort_option,
        show_chem_sys=viz_mode,
    )

    hover_by_comp = hover_texts_by_comp(fig)
    assert set(hover_by_comp) == set(df_data["composition"])
    prop_by_comp = dict(zip(df_data["composition"], df_data["property"], strict=True))
    for comp, hover_text in hover_by_comp.items():
        assert hover_text.startswith(f"Composition: {comp}<br>")
        if prop_name:
            assert hover_text.endswith(f"<br>property: {prop_by_comp[comp]:.2f}")
        else:  # chemical system info instead of property
            chem_sys = Composition(comp).chemical_system
            assert hover_text.endswith(f"<br>Chemical System: {chem_sys}")


@pytest.mark.parametrize(
    "color_scale",
    ["linear", "log", "arcsinh", {"type": "arcsinh", "scale_factor": 1.5}],
)
@pytest.mark.parametrize("hover_format", [None, ".1f", ".4f", ".1e", "g", ",.3g", ""])
def test_hover_format(
    color_scale: ColorScale | dict[str, Any], hover_format: str | None
) -> None:
    """Hover text formats coordinates and original (untransformed) property values."""
    props = [0.123, 1.2345, 12.345, 123.456, 1234.567]
    df_test = pd.DataFrame(
        {"composition": ["Fe2O3", "Al2O3", "Cu", "SiO2", "TiO2"], "property": props}
    )
    fmt_kwargs: dict[str, Any] = {}
    if hover_format is not None:
        fmt_kwargs["hover_format"] = hover_format
    fmt = ".2f" if hover_format is None else hover_format

    fig = pmv.cluster_compositions(
        df_in=df_test,
        prop_name="property",
        embedding_method="one-hot",
        projection="pca",
        color_scale=color_scale,
        **fmt_kwargs,
    )

    # ascending property values, so default sorting keeps the input order
    trace = fig.data[0]
    for idx, (_comp, hover_text) in enumerate(trace.customdata):
        assert f"Principal Component 1: {trace.x[idx]:{fmt}}<br>" in hover_text
        assert f"Principal Component 2: {trace.y[idx]:{fmt}}<br>" in hover_text
        assert hover_text.endswith(f"<br>property: {props[idx]:{fmt}}")


@pytest.mark.parametrize(
    ("values", "color_scale", "expected", "n_traces"),
    [
        pytest.param(
            [0.5, np.nan, np.inf, -np.inf],
            "linear",
            ["0.50", "nan", "inf", "-inf"],
            1,
            id="nan-inf-linear",
        ),
        pytest.param(
            [0.5, np.nan, np.inf, -np.inf],
            {"type": "linear"},
            ["0.50", "nan", "inf", "-inf"],
            1,
            id="nan-inf-linear-dict",
        ),
        pytest.param(
            [0.5, np.nan, -1.0, 2.0],
            "log",
            ["0.50", "nan", "-1.00", "2.00"],
            1,
            id="nan-non-positive-log",
        ),
        pytest.param(
            [0.5, np.nan, -1.0, 2.0],
            "arcsinh",
            ["0.50", "nan", "-1.00", "2.00"],
            1,
            id="nan-arcsinh",
        ),
        pytest.param(
            *(["low", "medium", "high"], "linear", ["low", "medium", "high"], 3),
            id="strings",
        ),
        pytest.param(
            [True, False, True], "linear", ["1.00", "0.00", "1.00"], 2, id="booleans"
        ),
    ],
)
def test_hover_special_property_values(
    values: list[Any],
    color_scale: ColorScale | dict[str, Any],
    expected: list[str],
    n_traces: int,
) -> None:
    """NaN/inf, non-positive, string and boolean properties are plotted and hovered."""
    compositions = ["Fe2O3", "Al2O3", "Cu", "SiO2"][: len(values)]
    df_test = pd.DataFrame({"composition": compositions, "prop": values})
    fig = pmv.cluster_compositions(
        df_in=df_test,
        prop_name="prop",
        embedding_method="one-hot",
        projection="pca",
        color_scale=color_scale,
    )
    # all points plotted (incl. NaN), one trace (with its own color) per category
    assert len(fig.data) == n_traces
    assert sum(len(trace.x) for trace in fig.data) == len(values)
    assert len({str(trace.marker.color) for trace in fig.data}) == n_traces
    hover_props = {
        comp: hover.split("<br>")[-1]
        for comp, hover in hover_texts_by_comp(fig).items()
    }
    assert hover_props == {
        comp: f"prop: {val}" for comp, val in zip(compositions, expected, strict=True)
    }


@pytest.mark.parametrize(
    ("projection", "expected_projector_type"),
    [("pca", PCA), ("tsne", TSNE), ("isomap", Isomap), ("kernel_pca", KernelPCA)],
)
def test_attached_projector_and_embeddings(
    sample_df: pd.DataFrame,
    projection: ProjectionMethod,
    expected_projector_type: type,
) -> None:
    """Test that projector and embeddings attributes are attached to the figure."""
    fig = pmv.cluster_compositions(
        df_in=sample_df,
        embedding_method="one-hot",
        projection=projection,
    )

    assert isinstance(fig, ClusterFigure)
    assert isinstance(fig.projector, expected_projector_type)
    # when present, ensure 2D projection
    if n_components := getattr(fig.projector, "n_components", None):
        assert n_components == 2
    assert isinstance(fig.embeddings, np.ndarray)
    np.testing.assert_array_equal(
        fig.embeddings, pmv.cluster.composition.one_hot_encode(sample_df["composition"])
    )


def test_mixed_composition_types() -> None:
    """Formula strings and Composition objects can be mixed in one column."""
    df_mixed = pd.DataFrame(
        {"composition": ["Fe2O3", Composition("Al2O3"), "Cu"], "prop": [1, 2, 3]}
    )
    fig = pmv.cluster_compositions(
        df_in=df_mixed, prop_name="prop", embedding_method="one-hot", projection="pca"
    )
    assert [comp for comp, _ in fig.data[0].customdata] == ["Fe2O3", "Al2 O3", "Cu"]


@pytest.mark.parametrize("n_components", [2, 3])
def test_precomputed_coordinates(
    df_with_embeddings: pd.DataFrame, n_components: int
) -> None:
    """Pre-computed coordinates are plotted as-is and take priority over embeddings."""
    coords = np_rng.random((len(df_with_embeddings), n_components))
    df_with_coords = df_with_embeddings.assign(coords=list(coords))

    fig = pmv.cluster_compositions(
        df_in=df_with_coords,
        prop_name="property",
        embedding_method="my_embeddings",  # ignored in favor of coords column
        projection="coords",
        n_components=n_components,
    )

    assert isinstance(fig, ClusterFigure)
    assert len(fig.data) == 1
    # points are sorted by property value
    sort_idx = np.argsort(df_with_coords["property"])
    for dim, axis in enumerate("xyz"[:n_components]):
        np.testing.assert_array_equal(getattr(fig.data[0], axis), coords[sort_idx, dim])
    # no projector fitted and no embeddings computed when coordinates are provided
    assert fig.projector is None
    assert fig.embeddings is None

    # Test with invalid coordinates shape (wrong number of components)
    wrong_dim = 5 - n_components  # 3D coords for 2D plot and vice versa
    df_wrong_shape = df_with_embeddings.assign(
        coords=list(np_rng.random((len(df_with_embeddings), wrong_dim)))
    )
    with pytest.raises(ValueError, match=f"must have length {n_components}"):
        pmv.cluster_compositions(
            df_in=df_wrong_shape,
            embedding_method="one-hot",
            projection="coords",
            n_components=n_components,
        )

    # Test with wrong type of data in coordinates column
    df_wrong_type = df_with_embeddings.assign(coords="not_array")
    with pytest.raises(ValueError, match="must contain arrays or lists"):
        pmv.cluster_compositions(
            df_in=df_wrong_type,
            embedding_method="one-hot",
            projection="coords",
        )


def test_coordinates_with_categorical_property(
    df_comp: pd.DataFrame, custom_coords_2d: np.ndarray
) -> None:
    """Categorical properties with pre-computed coordinates give a trace per class."""
    df_with_coords = df_comp.assign(
        property=["A", "B", "A", "C", "B"], coords=list(custom_coords_2d)
    )
    fig = pmv.cluster_compositions(
        df_in=df_with_coords,
        prop_name="property",
        embedding_method="one-hot",
        projection="coords",
    )
    assert {trace.name: len(trace.x) for trace in fig.data} == {"A": 2, "B": 2, "C": 1}
    assert fig.embeddings is None


@pytest.mark.parametrize("show_chem_sys", ["color", "shape", "color+shape"])
def test_coordinates_with_show_chem_sys(
    df_comp: pd.DataFrame, custom_coords_2d: np.ndarray, show_chem_sys: ShowChemSys
) -> None:
    """Test that coordinates work correctly with show_chem_sys."""
    df_with_coords = df_comp.assign(coords=list(custom_coords_2d))
    fig = pmv.cluster_compositions(
        df_in=df_with_coords,
        embedding_method="one-hot",
        projection="coords",
        show_chem_sys=show_chem_sys,
    )

    assert all(
        "<br>Chemical System: " in hover for hover in hover_texts_by_comp(fig).values()
    )
    if "shape" in show_chem_sys:  # 5 compositions with distinct chemical systems
        assert len(set(fig.data[0].marker.symbol)) == len(df_comp)
    else:
        assert len(fig.data) == len(df_comp)


def test_precomputed_embeddings_in_composition_col() -> None:
    """Test using pre-computed embeddings directly in composition column."""
    embeddings = np_rng.random((5, 10))  # 5 samples, 10 features
    df_emb = pd.DataFrame(
        {"composition": list(embeddings), "property": [3.0, 1.0, 4.0, 2.0, 5.0]}
    )

    fig = pmv.cluster_compositions(
        df_in=df_emb,
        prop_name="property",
        embedding_method="one-hot",  # This should be ignored since we have embeddings
        projection="pca",
    )

    assert len(fig.data) == 1
    assert len(fig.data[0].x) == len(df_emb)
    # embeddings are reordered along with the points sorted by property
    assert fig.embeddings is not None
    np.testing.assert_array_equal(
        fig.embeddings, embeddings[np.argsort(df_emb["property"])]
    )


def test_too_many_chem_systems() -> None:
    """Test warning when there are too many chemical systems for shape markers."""
    # 3D scatter only supports 5 marker symbols
    df_chem_sys = pd.DataFrame(
        {"composition": ["Fe2O3", "Al2O3", "Cu", "Au", "Ag", "SiO2"]}
    )
    msg = "Number of unique chemical systems .* exceeds available marker symbols"
    with pytest.warns(UserWarning, match=msg):
        pmv.cluster_compositions(
            df_in=df_chem_sys,
            embedding_method="one-hot",
            projection="pca",
            n_components=3,
            show_chem_sys="shape",
        )


def test_embeddings_from_column(df_with_embeddings: pd.DataFrame) -> None:
    """Test using embeddings from a DataFrame column."""
    embeddings = np.array(df_with_embeddings["my_embeddings"].tolist())

    fig = pmv.cluster_compositions(
        df_in=df_with_embeddings,
        prop_name="property",
        embedding_method="my_embeddings",  # Use this column for embeddings
        projection="pca",
    )

    assert len(fig.data) == 1
    assert len(fig.data[0].x) == len(df_with_embeddings)
    assert isinstance(fig.projector, PCA)
    # embeddings are reordered along with the points sorted by property
    assert fig.embeddings is not None
    np.testing.assert_array_equal(
        fig.embeddings, embeddings[np.argsort(df_with_embeddings["property"])]
    )


@pytest.mark.parametrize(
    ("projection", "component_label"),
    [
        ("pca", "Principal Component"),
        ("tsne", "t-SNE Component"),
        ("isomap", "Isomap Component"),
        ("kernel_pca", "Kernel PCA Component"),
        # Custom function should use the function name or 'Component' if no proper name
        (
            lambda x, n_components, **_: np_rng.random((x.shape[0], n_components)),
            "Component",
        ),
        # Column name of coordinates should use generic "Component"
        ("coords", "Component"),
    ],
)
def test_coordinates_with_labels(
    df_prop: pd.DataFrame,
    custom_coords_2d: np.ndarray,
    custom_coords_3d: np.ndarray,
    projection: ProjectionMethod | ProjectionCallable,
    component_label: str,
) -> None:
    """Axis titles and hover text use the projection's component label."""
    df_2d = df_prop.copy()
    df_3d = df_prop.copy()

    # Add coordinates to DataFrames if testing column of coordinates
    if projection == "coords":
        df_2d["coords"] = list(custom_coords_2d)
        df_3d["coords"] = list(custom_coords_3d)

    fig = pmv.cluster_compositions(
        df_in=df_2d,
        prop_name="property",
        embedding_method="one-hot",
        projection=projection,
        n_components=2,
    )

    # Check axis labels are correctly set based on projection
    assert fig.layout.xaxis.title.text == f"{component_label} 1"
    assert fig.layout.yaxis.title.text == f"{component_label} 2"
    for hover_text in hover_texts_by_comp(fig).values():
        assert f"<br>{component_label} 1: " in hover_text
        assert f"<br>{component_label} 2: " in hover_text

    if projection == "coords":
        fig_3d = pmv.cluster_compositions(
            df_in=df_3d,
            prop_name="property",
            embedding_method="one-hot",
            projection=projection,
            n_components=3,
        )

        # Check 3D axis labels
        assert fig_3d.layout.scene.xaxis.title.text == f"{component_label} 1"
        assert fig_3d.layout.scene.yaxis.title.text == f"{component_label} 2"
        assert fig_3d.layout.scene.zaxis.title.text == f"{component_label} 3"

        # Verify embeddings are not calculated when using column
        assert fig_3d.embeddings is None


@pytest.mark.parametrize(
    ("color_scale", "expected_title"),
    [
        ("linear", "property"),
        ({"type": "linear"}, "property"),
        ("log", "property (log scale)"),
        ("arcsinh", "property (arcsinh scale)"),
        ({"type": "arcsinh", "scale_factor": 1.5}, "property (arcsinh scale)"),
        *[
            (
                {"type": "arcsinh", "lin_thresh": lin_thresh, "lin_scale": lin_scale},
                "property (arcsinh scale)",
            )
            for lin_thresh, lin_scale in ((0.1, 0.5), (2.0, 1.0), (6.0, 2.0))
        ],
    ],
)
def test_color_scale_options(
    sample_df: pd.DataFrame,
    color_scale: ColorScale | dict[str, Any],
    expected_title: str,
) -> None:
    """Color scales set the colorbar title, layout and (non-linear) tick labels."""
    df_test = sample_df.assign(property=[0.1, 10, 1000])
    fig = pmv.cluster_compositions(
        df_in=df_test,
        prop_name="property",
        embedding_method="one-hot",
        projection="pca",
        color_scale=color_scale,
    )

    colorbar = fig.layout.coloraxis.colorbar
    assert colorbar.title.text == expected_title
    assert colorbar.orientation == "h"
    assert colorbar.y == 0
    if expected_title == "property":  # raw values as colors, plotly's default ticks
        assert list(fig.data[0].marker.color) == [0.1, 10, 1000]
        assert colorbar.ticktext is None
    else:  # tick labels cover the data range
        assert {"0.1", "10", "1000"} <= set(colorbar.ticktext)


@pytest.mark.parametrize(
    ("test_case", "expected_error"),
    [
        (
            {"color_scale": "invalid"},
            "color_scale='invalid' must be one of ('linear', 'log', 'arcsinh') or dict",
        ),
        (
            {"color_scale": {"lin_thresh": 0.5}},  # Missing 'type'
            "When color_scale is a dict, 'type' key must be provided",
        ),
        (
            {
                "color_scale": {"type": "arcsinh", "scale_factor": -1.0}
            },  # Negative scale_factor
            "scale_factor=-1.0 must be positive for arcsinh scale",
        ),
    ],
)
def test_color_scale_errors(
    sample_df: pd.DataFrame, test_case: dict[str, Any], expected_error: str
) -> None:
    """Test error handling for various invalid color scale configurations."""
    with pytest.raises(ValueError, match=re.escape(expected_error)):
        pmv.cluster_compositions(
            df_in=sample_df,
            composition_col="composition",
            prop_name="property",
            projection="pca",
            **test_case,
        )


wide_range = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
signed_range = [-10.0, -5.0, -0.5, 0.5, 5.0, 10.0]


@pytest.mark.parametrize(
    ("values", "color_scale", "nice_labels"),
    [
        pytest.param(wide_range, "log", {"0.01", "10", "1000"}, id="wide-log"),
        pytest.param(wide_range, "arcsinh", {"0.01", "10", "1000"}, id="wide-arcsinh"),
        pytest.param(
            wide_range,
            {"type": "arcsinh", "lin_thresh": 0.5, "lin_scale": 0.8},
            {"0.01", "10", "1000"},
            id="wide-arcsinh-lin-thresh",
        ),
        pytest.param(
            wide_range,
            {"type": "arcsinh", "scale_factor": 1.5},
            {"0.01", "10", "1000"},
            id="wide-arcsinh-scale-factor",
        ),
        pytest.param(
            [0.95, 1.0, 1.05, 1.1, 1.15], "arcsinh", {"1"}, id="narrow-arcsinh"
        ),
        pytest.param(
            [0.001, 0.1, 10, 1000], "log", {"1", "100"}, id="multi-decade-log"
        ),
        pytest.param(
            [0.001, 0.1, 10, 1000], "arcsinh", {"10", "1000"}, id="multi-decade-arcsinh"
        ),
        pytest.param(
            [-100, -50, -10, -1], "arcsinh", {"-100", "-1"}, id="negative-arcsinh"
        ),
        pytest.param(
            [-100.0, -10.0, -1.0, -0.1, 0, 0.1, 1.0, 10.0, 100.0],
            "arcsinh",
            {"-100", "0", "100"},
            id="symmetric-arcsinh",
        ),
        pytest.param([1.2, 2.3, 3.4], "log", {"1.5", "2", "3"}, id="small-range-log"),
        pytest.param(
            [-100.0, -1.0, -0.1, 0.1, 1.0, 100.0],
            "arcsinh",
            {"-100", "0", "100"},
            id="zero-crossing-arcsinh",
        ),
        *[
            pytest.param(
                signed_range,
                {"type": "arcsinh", "lin_thresh": lin_thresh, "lin_scale": lin_scale},
                {"-10", "0", "10"},
                id=f"lin-thresh-{lin_thresh}",
            )
            for lin_thresh, lin_scale in ((0.1, 0.5), (2.0, 1.0), (6.0, 2.0))
        ],
        pytest.param(
            [0.0001, 0.01, 1.0, 100.0, 10000.0],
            "arcsinh",
            {"0.1", "10", "1000", "10^4"},
            id="very-wide-arcsinh",
        ),
        pytest.param(
            [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
            "arcsinh",
            {"10^-9", "10^-7", "10^-5"},
            id="very-small-arcsinh",
        ),
        pytest.param(
            [0.001, 0.002, 0.003, 0.004, 0.005],
            "arcsinh",
            {"0.001", "0.002", "0.003", "0.004", "0.005"},
            id="milli-arcsinh",
        ),
        pytest.param(
            [0.001, 0.002, 0.003, 0.004, 0.005],
            "log",
            {"0.001", "0.002", "0.003", "0.004", "0.005"},
            id="milli-log",
        ),
        pytest.param(
            [50.0, 60.0, 70.0, 80.0, 100.0],
            "arcsinh",
            {"50", "60", "70", "80", "100"},
            id="far-from-zero-arcsinh",
        ),
        pytest.param(
            [1e5, 1e6, 1e7, 1e8, 1e9],
            "arcsinh",
            {"10^5", "10^9"},
            id="very-large-arcsinh",
        ),
        pytest.param(
            [-1000.0, -10.0, -0.1, 0.0, 0.1, 10.0, 1000.0],
            "arcsinh",
            {"-1000", "0", "10"},
            id="mixed-signs-arcsinh",
        ),
        pytest.param(
            [-0.01, -0.001, -0.0001, 0.0, 0.0001, 0.001, 0.01],
            "arcsinh",
            {"-0.001", "0", "0.001", "10^-4"},
            id="near-zero-arcsinh",
        ),
        pytest.param(
            [-10000.0, -1000.0, -100.0, -10.0, -1.0],
            "arcsinh",
            {"-10^4", "-100", "-1"},
            id="negative-large-arcsinh",
        ),
        pytest.param(
            [-10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0],
            {"type": "arcsinh", "lin_thresh": 0.01, "lin_scale": 0.5},
            {"-10", "-1", "0"},
            id="tiny-lin-thresh",
        ),
        pytest.param(
            [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0],
            {"type": "arcsinh", "lin_thresh": 50.0, "lin_scale": 2.0},
            {"-100", "0", "100"},
            id="huge-lin-thresh",
        ),
        pytest.param(
            [1.0, 2.0, 3.0, 5.0, 8.0, 10.0],
            "arcsinh",
            {"1", "2", "4", "10"},
            id="one-decade-arcsinh",
        ),
        pytest.param(
            [1.0, 10.0, 100.0, -1.0, -10.0],
            "arcsinh",
            {"-10", "-1", "0", "1", "100"},
            id="mixed-sign-arcsinh",
        ),
        pytest.param(
            [0.1, 1.0, 10.0, 100.0, 1000.0],
            "log",
            {"0.1", "1", "10", "100", "1000"},
            id="decades-log",
        ),
        pytest.param(
            [1.0, 5.0, 10.0, 50.0, 100.0, 1000.0],
            "log",
            {"1", "5", "10", "50", "100", "1000"},
            id="whole-numbers-log",
        ),
        pytest.param(
            [1.0, 5.0, 10.0, 50.0, 100.0, 1000.0],
            "arcsinh",
            {"10", "50", "1000"},
            id="whole-numbers-arcsinh",
        ),
    ],
)
def test_colorbar_ticks(
    values: list[float],
    color_scale: ColorScale | dict[str, Any],
    nice_labels: set[str],
) -> None:
    """Log/arcsinh colorbars get >= 5 increasing ticks with unique, nice round labels
    covering each sign present in the data.
    """
    compositions = "Fe2O3 Al2O3 Cu SiO2 TiO2 ZnO MgO CaO Na2O K2O".split()  # noqa: SIM905
    df_test = pd.DataFrame(
        {"composition": compositions[: len(values)], "property": values}
    )
    fig = pmv.cluster_compositions(
        df_in=df_test,
        prop_name="property",
        embedding_method="one-hot",
        projection="pca",
        color_scale=color_scale,
    )

    colorbar = fig.layout.coloraxis.colorbar
    tick_text, tick_vals = list(colorbar.ticktext), list(colorbar.tickvals)
    assert len(tick_text) == len(tick_vals) >= 5
    assert len(set(tick_text)) == len(tick_text), f"duplicate labels {tick_text=}"
    assert tick_vals == sorted(set(tick_vals))  # strictly increasing
    assert nice_labels <= set(tick_text), f"{tick_text=}"
    # whole numbers are labeled without decimals
    assert not any(re.fullmatch(r"-?[1-9]\d*\.0+", label) for label in tick_text)
    assert any(label.startswith("-") for label in tick_text) == (min(values) < 0)
    has_positive = any(
        not label.startswith("-") and label != "0" for label in tick_text
    )
    assert has_positive == (max(values) > 0)


@pytest.mark.parametrize(
    "color_scale",
    [
        "log",
        "arcsinh",
        {"type": "arcsinh", "scale_factor": 1.5},
        {"type": "arcsinh", "lin_thresh": 2.0, "lin_scale": 1.0},
    ],
)
def test_marker_colors_match_colorbar_ticks(
    color_scale: ColorScale | dict[str, Any],
) -> None:
    """Points whose value equals a tick label sit exactly at that tick's position."""
    values = [1.87, 2.0, 10.0, 50.0, 200.0]
    df_prop = pd.DataFrame(
        {"composition": ["P4H12", "SiO2", "ZnO", "Fe2O3", "TiO2"], "property": values}
    )
    fig = pmv.cluster_compositions(
        df_in=df_prop,
        prop_name="property",
        embedding_method="one-hot",
        projection="pca",
        color_scale=color_scale,
    )

    colorbar = fig.layout.coloraxis.colorbar
    tick_pos = dict(zip(colorbar.ticktext, colorbar.tickvals, strict=True))
    colors = fig.data[0].marker.color  # values ascending, so input order is kept
    assert list(colors) == sorted(colors)  # monotonic, incl. 1.87 < 2.0
    n_matched = 0
    for value, color in zip(values, colors, strict=True):
        if (label := f"{value:g}") in tick_pos:
            assert color == tick_pos[label]
            n_matched += 1
    assert n_matched >= 3


def test_log_scale_with_negative_values(sample_df: pd.DataFrame) -> None:
    """Non-positive properties are omitted from logarithmic coloring."""
    df_test = sample_df.assign(property=[1.0, -1.0, 2.0])
    fig = pmv.cluster_compositions(
        df_in=df_test,
        composition_col="composition",
        prop_name="property",
        projection="pca",
        color_scale="log",
    )

    assert fig.layout.coloraxis.colorbar.title.text == "property (log scale)"
    assert np.isnan(fig.data[0].marker.color).sum() == 1


def get_point_annotations(fig: go.Figure, n_components: int) -> tuple[Any, ...]:
    """Return the point annotations of a 2D or 3D figure."""
    return (fig.layout if n_components == 2 else fig.layout.scene).annotations


@pytest.mark.parametrize("n_components", [2, 3])
def test_annotate_points_basic(
    sample_df: pd.DataFrame, n_components: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Annotation functions see original and projected columns, and annotations are
    batch-assigned.

    Calling fig.add_annotation() per point is O(n^2) (plotly re-validates the whole
    annotations array on each call), so both the 2D and 3D branches must collect dicts
    and assign fig.layout.annotations / scene_annotations once, not one at a time.
    """
    df_extra = sample_df[["composition"]].assign(
        extra_info=["info_0", "info_1", "info_2"]
    )

    def annotate_func(row: pd.Series) -> str:
        """Annotate with composition, an original column and a projected coord."""
        return f"{row['composition']} {row['extra_info']} {row['pca1']:.3f}"

    # spy on per-point add_annotation to guard against O(n^2) reintroduction
    n_calls = {"count": 0}
    orig_add_annotation = go.Figure.add_annotation

    def spy(self: go.Figure, *args: object, **kwargs: object) -> object:
        n_calls["count"] += 1
        return orig_add_annotation(self, *args, **kwargs)

    monkeypatch.setattr(go.Figure, "add_annotation", spy)

    fig = pmv.cluster_compositions(
        df_in=df_extra,
        embedding_method="one-hot",
        projection="pca",
        n_components=n_components,
        annotate_points=annotate_func,
        show_projection_stats=False,  # Disable stats for simpler counting
    )

    point_annotations = get_point_annotations(fig, n_components)
    assert [anno.text for anno in point_annotations] == [
        f"{comp} {extra} {pca1:.3f}"
        for comp, extra, pca1 in zip(
            df_extra["composition"], df_extra["extra_info"], fig.data[0].x, strict=True
        )
    ]

    # annotations were batch-assigned, not added via per-point add_annotation calls
    assert n_calls["count"] == 0, (
        f"per-point fig.add_annotation called {n_calls['count']}x (should be batched)"
    )


@pytest.mark.parametrize("n_components", [2, 3])
def test_annotate_points_dict(sample_df: pd.DataFrame, n_components: int) -> None:
    """Returned dicts override default annotation params and keep the others."""

    def annotate_func(row: pd.Series) -> dict[str, Any]:
        """Annotation function returning a dictionary."""
        return {
            "text": f"ID: {row.name}",
            "yshift": 25,  # Override default 10
            "showarrow": True,  # Override default False
            "arrowhead": 2,
        }

    fig = pmv.cluster_compositions(
        df_in=sample_df,
        embedding_method="one-hot",
        projection="pca",
        n_components=n_components,
        annotate_points=annotate_func,
        show_projection_stats=False,  # Disable stats for simpler counting
    )

    point_annotations = get_point_annotations(fig, n_components)
    assert len(point_annotations) == len(sample_df)
    for idx, annotation in enumerate(point_annotations):
        assert annotation.text == f"ID: {sample_df.index[idx]}"
        assert annotation.yshift == 25
        assert annotation.showarrow is True
        assert annotation.arrowhead == 2
        assert annotation.font.size == 10  # default font size retained


@pytest.mark.parametrize("n_components", [2, 3])
def test_annotate_points_conditional(
    sample_df: pd.DataFrame, n_components: int
) -> None:
    """Test conditional annotation where some points are annotated."""
    target_composition = sample_df["composition"].iloc[0]

    def annotate_func(row: pd.Series) -> str | None:
        """Annotate only the first composition."""
        if row["composition"] == target_composition:
            return f"Target: {row['composition']}"
        return None

    fig = pmv.cluster_compositions(
        df_in=sample_df,
        prop_name="property",
        embedding_method="one-hot",
        projection="pca",
        n_components=n_components,
        annotate_points=annotate_func,
        show_projection_stats=False,  # Disable stats for simpler counting
    )

    point_annotations = get_point_annotations(fig, n_components)
    assert len(point_annotations) == 1

    annotation = point_annotations[0]
    assert annotation.text == f"Target: {target_composition}"
    # Check if annotation is placed at the correct point
    assert annotation.x == fig.data[0].x[0]
    assert annotation.y == fig.data[0].y[0]


def test_annotate_points_with_coords(sample_df: pd.DataFrame) -> None:
    """Test annotation when using pre-computed coordinates."""
    coords = np_rng.random((len(sample_df), 2))
    df_coords = sample_df.assign(coords=list(coords))

    fig = pmv.cluster_compositions(
        df_in=df_coords,
        embedding_method="one-hot",
        projection="coords",
        annotate_points=lambda row: f"Row {row.name}",
    )

    # For precomputed coords, no stats annotation is added by default
    # so we expect exactly the number of points annotated.
    actual_annotations = fig.layout.annotations
    assert len(actual_annotations) == len(sample_df)

    for idx, annotation in enumerate(actual_annotations):
        assert annotation.text == f"Row {sample_df.index[idx]}"
        # Check annotation position matches the precomputed coordinate
        assert annotation.x == coords[idx, 0]
        assert annotation.y == coords[idx, 1]
