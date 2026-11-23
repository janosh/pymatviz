"""Tests for pymatviz.cluster: embeddings, projections and cluster_compositions."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import kemio
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, Isomap

import pymatviz as pmv
from pymatviz.cluster import (
    CHEM_SYS_COLORS,
    CHEM_SYS_SYMBOLS,
    ClusterScatterPlot,
    ClusterScatterPlot3D,
    ProjectionMethod,
    magpie_featurize,
    one_hot_encode,
    project_vectors,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pymatviz.cluster import ColorScale, ShowChemSys


@pytest.mark.parametrize(
    "compositions",
    [
        pytest.param(["H2O", "CO2", "NaCl"], id="strings"),
        pytest.param(
            [kemio.Composition(comp) for comp in ["H2O", "CO2", "NaCl"]],
            id="compositions",
        ),
        pytest.param(pd.Series(["H2O", "CO2", "NaCl"]), id="series"),
    ],
)
def test_one_hot_encode(
    compositions: list[str | kemio.Composition] | pd.Series,
) -> None:
    """Encode supported inputs with all elements or a selected subset."""
    elements = ["H", "C", "O", "Na", "Cl"]
    # unit-norm element fractions: H2O and CO2 are (2, 1)/sqrt(5), NaCl (1, 1)/sqrt(2)
    two_one, one_one = np.array([2, 1]) / np.sqrt(5), np.array([1, 1]) / np.sqrt(2)
    expected = np.zeros((3, 5))
    expected[0, [0, 2]] = two_one  # H, O
    expected[1, [2, 1]] = two_one  # O, C
    expected[2, [3, 4]] = one_one  # Na, Cl
    result_custom = one_hot_encode(compositions, elements=elements)
    np.testing.assert_allclose(result_custom, expected, rtol=1e-14, atol=0)

    # all 118 elements by default, the same nonzero entries in Z-order columns
    result = one_hot_encode(compositions)
    assert result.shape == (3, 118)
    z_columns = [0, 5, 7, 10, 16]  # H, C, O, Na, Cl
    # 118-wide L2 norms sum in another order: observed rel. diff 1.2e-16 (< f64 eps)
    np.testing.assert_allclose(
        result[:, z_columns], result_custom, rtol=4.4e-16, atol=0
    )
    assert np.count_nonzero(result) == 6


@pytest.mark.parametrize("normalize", [False, True])
def test_one_hot_encode_oxidation_states(normalize: bool) -> None:
    """Oxidation states do not change elemental fractions or normalized embeddings."""
    result = one_hot_encode(
        ["Fe2O3", kemio.Composition({"Fe3+": 2, "O2-": 3})],
        elements=["Fe", "O"],
        normalize=normalize,
    )
    np.testing.assert_array_equal(result[0], result[1])
    if not normalize:
        np.testing.assert_array_equal(result, [[0.4, 0.6], [0.4, 0.6]])
    with pytest.raises(ValueError, match=r"contains \['Al'\]"):
        one_hot_encode(["Fe2O3Al"], elements=["Fe", "O"])
    with pytest.raises(ValueError, match="duplicate entries"):
        one_hot_encode(["FeO"], elements=["Fe", "O", "Fe"])


@pytest.mark.parametrize("embed_func", [one_hot_encode, magpie_featurize])
@pytest.mark.parametrize("compositions", [[1, 2, 3], [["H2O"]]])
def test_embed_invalid_input(
    embed_func: Callable[..., np.ndarray], compositions: list[Any]
) -> None:
    """Non-composition inputs raise a TypeError naming the expected types."""
    with pytest.raises(TypeError, match="Expected a formula string, mapping"):
        embed_func(compositions)


def test_magpie_featurize() -> None:
    """Featurize a Series, normalized or not, with and without a feature subset."""
    compositions = pd.Series(["H2O", "CO2", "NaCl", "Fe2O3"])
    labels = kemio.composition_feature_labels()

    raw = magpie_featurize(compositions)
    assert raw.shape == (len(compositions), len(labels))
    assert not np.isnan(raw).any()
    result = magpie_featurize(compositions, normalize=True)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.0, rtol=1e-14, atol=0)
    np.testing.assert_allclose(
        raw / np.linalg.norm(raw, axis=1)[:, None], result, rtol=1e-14, atol=0
    )
    subset = magpie_featurize(compositions, feature_subset=labels[1::50])
    np.testing.assert_array_equal(subset, raw[:, 1::50])

    with pytest.raises(ValueError, match=r"Unknown Magpie features \['Nonexistent"):
        magpie_featurize(compositions, feature_subset=["NonexistentFeature"])


@pytest.fixture
def sample_data() -> np.ndarray:
    """Generate sample data for testing projection methods."""
    xs, _ = make_blobs(  # Create synthetic data with 3 clusters in 10 dimensions
        n_samples=100, n_features=10, centers=3, random_state=0, cluster_std=1.0
    )
    return xs


@pytest.mark.parametrize(
    ("method", "kwargs", "expected_obj_type", "extra_attrs"),
    [
        ("pca", {"whiten": True}, PCA, {"n_components_": 2, "svd_solver": "full"}),
        ("pca", {"svd_solver": "arpack"}, PCA, {}),
        (
            "tsne",
            {"perplexity": 10.0, "learning_rate": 100.0, "max_iter": 500},
            TSNE,
            {"n_components": 2},
        ),
        ("umap", {"n_neighbors": 5, "min_dist": 0.2, "spread": 2}, None, {}),
        (
            "isomap",
            {"n_neighbors": 5, "metric": "euclidean", "path_method": "D"},
            Isomap,
            {},
        ),
        (
            "kernel_pca",
            {"kernel": "rbf", "gamma": 0.1, "fit_inverse_transform": True},
            KernelPCA,
            {"n_components": 2},
        ),
    ],
)
def test_project_vectors_methods(
    sample_data: np.ndarray,
    method: ProjectionMethod,
    kwargs: dict[str, Any],
    expected_obj_type: type,
    extra_attrs: dict[str, Any],
) -> None:
    """Each method returns its fitted projector with the given kwargs set."""
    if method == ProjectionMethod.umap:
        pytest.importorskip("umap")
        from umap import UMAP

        expected_obj_type = UMAP

    result, proj_obj = project_vectors(sample_data, method=method, **kwargs)
    assert result.shape == (100, 2)
    assert isinstance(proj_obj, expected_obj_type)
    for attr, value in (kwargs | extra_attrs).items():
        assert getattr(proj_obj, attr) == value

    # scale_data=True centers the projection and scales it to unit RMS (t-SNE
    # returns float32: eps 1.2e-7, observed mean ~4e-8)
    tol = 1e-12 if result.dtype == np.float64 else 1e-6
    np.testing.assert_allclose(result.mean(axis=0), 0, rtol=0, atol=tol)
    assert np.sqrt(np.mean(result**2)) == pytest.approx(1, rel=tol)
    assert np.linalg.matrix_rank(result) == 2


@pytest.mark.parametrize("n_components", [2, 3])
def test_project_vectors_components(sample_data: np.ndarray, n_components: int) -> None:
    """Default PCA projection to different numbers of components."""
    result, proj_obj = project_vectors(sample_data, n_components=n_components)

    assert result.shape == (100, n_components)
    assert isinstance(proj_obj, PCA)
    assert proj_obj.n_components_ == n_components
    assert proj_obj.explained_variance_ratio_.shape == (n_components,)
    # nested lists project exactly like arrays
    list_result, _ = project_vectors(sample_data.tolist(), n_components=n_components)
    np.testing.assert_array_equal(list_result, result)

    # centered, unit overall RMS, components ordered by decreasing spread
    np.testing.assert_allclose(result.mean(axis=0), 0, rtol=0, atol=1e-12)
    assert np.sqrt(np.mean(result**2)) == pytest.approx(1, rel=1e-12)
    spread = result.std(axis=0)
    assert np.all(np.diff(spread) < 0), f"{spread=}"


def test_project_vectors_with_scaling(sample_data: np.ndarray) -> None:
    """Projection with (default) and without data scaling."""
    result_with_scaling, proj_obj1 = project_vectors(sample_data, scale_data=True)
    result_without_scaling, proj_obj2 = project_vectors(sample_data, scale_data=False)
    assert isinstance(proj_obj1, PCA)
    assert isinstance(proj_obj2, PCA)
    assert proj_obj1.n_components_ == proj_obj2.n_components_ == 2
    assert proj_obj1.explained_variance_ratio_.mean() == pytest.approx(0.4348426)
    assert proj_obj2.explained_variance_ratio_.mean() == pytest.approx(0.4691580)

    # scale_data=False returns raw PCA scores (sklearn centers, nothing rescales)
    raw = PCA(n_components=2, svd_solver="full").fit_transform(sample_data)
    np.testing.assert_allclose(
        np.abs(result_without_scaling), np.abs(raw), rtol=0, atol=1e-10
    )
    assert not np.allclose(
        np.abs(result_with_scaling), np.abs(result_without_scaling), atol=1e-3
    )
    assert np.sqrt(np.mean(result_with_scaling**2)) == pytest.approx(1, rel=1e-12)


def test_project_vectors_random_state(sample_data: np.ndarray) -> None:
    """UMAP is reproducible for a fixed random state and varies across states."""
    pytest.importorskip("umap")
    from umap import UMAP

    (result1, umap_obj1), (result2, umap_obj2), (result3, umap_obj3) = (
        project_vectors(sample_data, method=ProjectionMethod.umap, random_state=seed)
        for seed in (0, 0, 1)
    )
    assert isinstance(umap_obj1, UMAP)
    assert isinstance(umap_obj3, UMAP)
    assert (umap_obj1.random_state, umap_obj3.random_state) == (0, 1)
    assert umap_obj1 is not umap_obj2
    assert np.allclose(result1, result2, atol=1e-10)
    assert not np.allclose(result1, result3, atol=1e-10)
    for result in (result1, result3):
        assert np.std(result) == pytest.approx(1.0, rel=0.5)


@pytest.mark.parametrize(
    ("n_samples", "kwargs", "match"),
    [
        (100, {"method": "invalid_method"}, "Unknown projection method="),
        (100, {"n_components": 1}, "n_components must be at least 2"),
        (1, {"n_components": 2}, "Need at least n_components=2 samples, got 1"),
        (100, {"method": ProjectionMethod.tsne, "n_components": 4}, ""),  # sklearn
        (3, {"method": ProjectionMethod.umap}, "UMAP needs at least n_components"),
    ],
)
def test_project_vectors_invalid_inputs(
    sample_data: np.ndarray, n_samples: int, kwargs: dict[str, Any], match: str
) -> None:
    """Invalid methods, component counts and too few samples raise ValueError."""
    with pytest.raises(ValueError, match=match):
        project_vectors(sample_data[:n_samples], **kwargs)


@pytest.mark.parametrize(
    ("method", "expected_attrs"),
    [(ProjectionMethod.tsne, {"perplexity": 20 / 3}), ("umap", {"n_neighbors": 15})],
)
def test_project_vectors_small_dataset(
    method: ProjectionMethod, expected_attrs: dict[str, float]
) -> None:
    """t-SNE adjusts perplexity to small datasets, UMAP keeps default n_neighbors."""
    if method == "umap":
        pytest.importorskip("umap")
    xs, _ = make_blobs(n_samples=20, n_features=5, centers=2, random_state=0)

    result, proj_obj = project_vectors(xs, method=method)
    assert result.shape == (len(xs), 2)
    for attr, value in expected_attrs.items():
        assert getattr(proj_obj, attr) == pytest.approx(value)
    assert not np.allclose(result, 0, atol=1e-10)
    assert np.isfinite(result).all()
    assert np.std(result) == pytest.approx(1.0, rel=0.5)


@pytest.mark.parametrize(
    "data",
    [
        np.array([[1.0, 2.0, 3.0]] * 4),
        np.array([[1.0, 2.0], [np.nan, 0.5], [3.0, -1.0], [0.0, 4.0]]),
    ],
    ids=["zero-variance", "nan-entry"],
)
def test_project_vectors_degenerate_data_stays_finite(data: np.ndarray) -> None:
    """Constant columns and NaN entries give finite coords (regression: the
    post-projection standardization divided by a zero std).
    """
    projected, _ = project_vectors(data, n_components=2)
    assert projected.shape == (4, 2)
    assert np.isfinite(projected).all()


def test_project_vectors_pca_is_exact_and_keeps_aspect_ratio() -> None:
    """PCA uses an exact SVD and one shared output scale for all axes."""
    rng = np.random.default_rng(seed=0)
    # 636x118 is a shape where sklearn's svd_solver="auto" picks randomized SVD.
    data = rng.normal(size=(636, 118)) * np.linspace(1, 3, 118)
    projected, pca = project_vectors(data, n_components=3)
    assert isinstance(pca, PCA)
    standardized = (data - data.mean(axis=0)) / data.std(axis=0)
    exact = PCA(n_components=3, svd_solver="full").fit_transform(standardized)
    exact /= np.sqrt(np.mean(exact**2))
    np.testing.assert_allclose(np.abs(projected), np.abs(exact), rtol=0, atol=1e-10)
    spread = projected.std(axis=0)
    variance_ratio = pca.explained_variance_ / pca.explained_variance_[0]
    np.testing.assert_allclose(spread / spread[0], np.sqrt(variance_ratio), rtol=1e-10)
    assert spread[0] > spread[1] > spread[2]


def test_project_vectors_isomap_two_samples() -> None:
    """Isomap clamps n_neighbors below n_samples for tiny inputs."""
    data = np.random.default_rng(seed=0).normal(size=(2, 4))
    with pytest.warns(UserWarning, match="adjusted to n_neighbors=1"):
        projected, _ = project_vectors(data, method=ProjectionMethod.isomap)
    assert projected.shape == (2, 2)


BASE_KWARGS: dict[str, Any] = dict(embedding_method="one-hot", projection="pca")


@pytest.fixture
def df_prop() -> pd.DataFrame:
    """Compositions from 4 chemical systems with a numeric and categorical column."""
    return pd.DataFrame(
        {
            "composition": ["Fe2O3", "FeO", "Al2O3", "Cu", "Au", "Fe3O4"],
            "property": [3.0, 1.0, 4.0, -2.0, 5.0, 0.0],
            "category": ["a", "b", "a", "b", "a", "a"],
        }
    )


def points(widget: ClusterScatterPlot | ClusterScatterPlot3D) -> pd.DataFrame:
    """Flatten all series into one row per point with coords, metadata and style."""
    rows = []
    for srs in widget.series:
        for idx, meta in enumerate(srs["metadata"]):
            row = {axis: srs[axis][idx] for axis in "xyz" if axis in srs} | meta
            row |= {"label": srs.get("label"), "color": srs.get("color")}
            row["symbol"] = srs["point_style"].get("symbol_type")
            if "color_values" in srs:
                row["color_value"] = srs["color_values"][idx]
            if "point_label" in srs:
                row["point_label"] = srs["point_label"][idx].get("text")
            rows.append(row)
    return pd.DataFrame(rows).set_index("composition")


@pytest.mark.parametrize("n_components", [2, 3])
@pytest.mark.parametrize("sort", [True, False, -1, np.int64(-1), np.True_, np.False_])
def test_cluster_compositions_property(
    df_prop: pd.DataFrame, n_components: int, sort: bool | int
) -> None:
    """Numeric properties color one series of points, matching the PCA projection."""
    widget = pmv.cluster_compositions(
        df_prop,
        prop_name="property",
        n_components=n_components,
        sort=sort,
        **BASE_KWARGS,
    )
    widget_cls = ClusterScatterPlot if n_components == 2 else ClusterScatterPlot3D
    assert isinstance(widget, widget_cls)
    assert isinstance(widget.projector, PCA)
    (srs,) = widget.series
    assert widget.embeddings is not None
    expected_coords, _ = project_vectors(
        widget.embeddings, method=ProjectionMethod.pca, n_components=n_components
    )
    df_points = points(widget).loc[df_prop["composition"]]
    for dim, axis in enumerate("xyz"[:n_components]):
        np.testing.assert_allclose(
            df_points[axis], expected_coords[:, dim], rtol=1e-12, atol=1e-12
        )
    np.testing.assert_array_equal(df_points["property"], df_prop["property"])
    np.testing.assert_array_equal(df_points["color_value"], df_prop["property"])
    if sort:
        assert srs["color_values"] == sorted(srs["color_values"], reverse=sort < 0)
    else:
        assert srs["color_values"] == df_prop["property"].tolist()
    assert widget.color_scale == {"scheme": "interpolateViridis", "type": "linear"}
    ratios = widget.projector.explained_variance_ratio_
    assert widget.x_axis == {
        "label": f"Principal Component 1 ({ratios[0]:.1%}, cumulative {ratios[0]:.1%})"
    }
    assert widget.show_legend is False
    if isinstance(widget, ClusterScatterPlot):
        assert widget.color_bar == {"title": "property"}
        assert srs["point_style"] == {"radius": 4}


@pytest.mark.parametrize(
    ("log", "color_scale", "expected_type", "expected_color_values"),
    [
        (True, "interpolateViridis", "log", [3.0, 1.0, 4.0, None, 5.0, None]),
        (False, {"type": "log"}, "log", [3.0, 1.0, 4.0, None, 5.0, None]),
        (False, {"type": "arcsinh"}, "arcsinh", [3.0, 1.0, 4.0, -2.0, 5.0, 0.0]),
        (
            False,
            {"type": {"type": "arcsinh", "threshold": 0.1}},
            {"type": "arcsinh", "threshold": 0.1},
            [3.0, 1.0, 4.0, -2.0, 5.0, 0.0],
        ),
    ],
)
def test_color_scale(
    df_prop: pd.DataFrame,
    log: bool,
    color_scale: ColorScale,
    expected_type: str | dict[str, Any],
    expected_color_values: list[float | None],
) -> None:
    """Scale types go to matterviz; log leaves non-positive values uncolored."""
    widget = pmv.cluster_compositions(
        df_prop,
        prop_name="property",
        log=log,
        color_scale=color_scale,
        sort=False,
        **BASE_KWARGS,
    )
    assert widget.color_scale == {"scheme": "interpolateViridis", "type": expected_type}
    assert widget.series[0]["color_values"] == expected_color_values


@pytest.mark.parametrize("show_chem_sys", ["color", "shape", "color+shape"])
@pytest.mark.parametrize("prop_name", [None, "property", "category"])
def test_chemical_system_visualization(
    df_prop: pd.DataFrame, show_chem_sys: ShowChemSys, prop_name: str | None
) -> None:
    """Chemical systems split points into series styled by color and/or shape."""
    widget = pmv.cluster_compositions(
        df_prop, prop_name=prop_name, show_chem_sys=show_chem_sys, **BASE_KWARGS
    )
    df_points = points(widget)
    systems = {
        comp: kemio.Composition(comp).chemical_system for comp in df_prop.composition
    }
    assert df_points["chem_system"].to_dict() == systems
    ordered_systems = sorted(set(systems.values()))
    has_shape = "shape" in show_chem_sys
    split_by_system = has_shape or prop_name is None
    for comp, row in df_points.iterrows():
        system_idx = ordered_systems.index(systems[comp])
        expected_symbol = CHEM_SYS_SYMBOLS[system_idx] if has_shape else None
        assert row["symbol"] == expected_symbol
        if show_chem_sys == "color+shape" and prop_name is None:
            assert row["color"] == CHEM_SYS_COLORS[system_idx]
        if split_by_system:
            assert systems[comp] in row["label"]
        if prop_name == "category":
            category = df_prop.set_index("composition").loc[comp, "category"]
            assert row["label"].startswith(category)
    n_series = len(widget.series)
    assert widget.show_legend == (n_series > 1)
    if prop_name == "property" and not has_shape:
        assert n_series == 1


def test_color_discrete_map(df_prop: pd.DataFrame) -> None:
    """color_discrete_map overrides chemical system colors."""
    widget = pmv.cluster_compositions(
        df_prop,
        show_chem_sys="color",
        color_discrete_map={"Fe-O": "red"},
        **BASE_KWARGS,
    )
    colors = {srs["label"]: srs.get("color") for srs in widget.series}
    assert colors == {"Al-O": None, "Au": None, "Cu": None, "Fe-O": "red"}


@pytest.mark.parametrize(
    ("dtype", "missing"), [(object, None), (object, np.nan), ("string", pd.NA)]
)
def test_categorical_property(df_prop: pd.DataFrame, dtype: Any, missing: Any) -> None:
    """Non-numeric properties (missing values included, None, NaN or pd.NA) split
    points into one series per category, colored per category also when split by
    system shape.
    """
    df_prop["category"] = df_prop["category"].astype(dtype)
    df_prop.loc[5, "category"] = missing
    widget = pmv.cluster_compositions(df_prop, prop_name="category", **BASE_KWARGS)
    assert [srs["label"] for srs in widget.series] == ["a", "b", "nan"]
    assert widget.series[-1]["metadata"][0]["category"] is None
    shapes = pmv.cluster_compositions(
        df_prop, prop_name="category", show_chem_sys="shape", **BASE_KWARGS
    )
    colors = {srs["label"]: srs["color"] for srs in shapes.series}
    assert colors["a · Fe-O"] == colors["a · Al-O"] != colors["b · Fe-O"]
    assert widget.color_scale is None
    assert all("color_values" not in srs for srs in widget.series)


@pytest.mark.parametrize(
    ("projection", "projection_kwargs", "expected_stats"),
    [
        (
            "tsne",
            dict(perplexity=2, learning_rate=50),
            "perplexity = 2\nlearning_rate = 50",
        ),
        ("isomap", dict(n_neighbors=3), "n_neighbors = 3\nmetric = 'minkowski'"),
        ("kernel_pca", dict(gamma=0.1), "kernel = 'rbf'\ngamma = 0.1"),
    ],
)
def test_projection_stats(
    df_prop: pd.DataFrame,
    projection: str,
    projection_kwargs: dict[str, Any],
    expected_stats: str,
) -> None:
    """Non-PCA projection parameters show in the 2D annotation badge."""
    for show_stats, expected in ((True, expected_stats), (False, None)):
        widget = pmv.cluster_compositions(
            df_prop,
            embedding_method="one-hot",
            projection=projection,
            projection_kwargs=projection_kwargs,
            show_projection_stats=show_stats,
        )
        assert isinstance(widget, ClusterScatterPlot)
        assert widget.annotation == expected


@pytest.mark.parametrize(
    ("kwargs", "err_type", "match"),
    [
        ({"n_components": 4}, ValueError, "n_components=4 must be 2 or 3"),
        ({"embedding_method": "invalid_method"}, ValueError, "embedding_method="),
        ({"embedding_method": None}, ValueError, "embedding_method=None must be in"),
        ({"projection": None}, ValueError, "projection=None must be in"),
        ({"prop_name": "property", "sort": "asc"}, TypeError, "Invalid sort"),
        (
            {"df": pd.DataFrame({"composition": [object(), object()]})},
            TypeError,
            "Expected a formula string",
        ),
        (
            {"n_components": 3, "show_chem_sys": "shape"},
            ValueError,
            "needs marker symbols, which ScatterPlot3D lacks",
        ),
        (
            {"n_components": 3, "annotate_points": str},
            ValueError,
            "annotate_points needs n_components=2",
        ),
        ({"show_chem_sys": "size"}, ValueError, "show_chem_sys='size' must be"),
        ({"color_scale": {"type": "sqrt"}}, ValueError, "type 'sqrt' must be"),
        (
            {"log": True, "color_scale": {"type": "arcsinh"}},
            ValueError,
            "type 'arcsinh' must be 'linear', 'log' \\(or omitted with log=True\\)",
        ),
        (
            {
                "df": pd.DataFrame({"composition": ["Fe", "Cu"], "prop": [-1, 0]}),
                "prop_name": "prop",
                "log": True,
            },
            ValueError,
            "Log color scale needs at least one positive finite value",
        ),
        ({"prop_name": "missing"}, ValueError, "prop_name='missing' not found"),
        (
            {"prop_name": "property", "sort": np.sort},
            ValueError,
            "sort callable must return a permutation",
        ),
        (
            {"prop_name": "property", "sort": lambda vals: np.argsort(vals)[:1]},
            ValueError,
            "sort callable must return a permutation",
        ),
        (
            {"df": pd.DataFrame({"composition": []})},
            ValueError,
            "df has no rows to cluster",
        ),
        *(  # every row of a pre-computed coordinates column is checked
            (
                {
                    "df": pd.DataFrame(
                        {"composition": ["Fe", "Cu"], "xy": [[1, 2], coords]}
                    ),
                    "projection": "xy",
                },
                ValueError,
                f"2 coordinates per row, got 1 rows like {re.escape(repr(coords))}",
            )
            for coords in ([3], "ab", [1, 2, 3])
        ),
    ],
)
def test_cluster_compositions_invalid_inputs(
    df_prop: pd.DataFrame, kwargs: dict[str, Any], err_type: type[Exception], match: str
) -> None:
    """Invalid arguments raise informative errors."""
    call_kwargs: dict[str, Any] = {"df": df_prop, **BASE_KWARGS, **kwargs}
    with pytest.raises(err_type, match=match):
        pmv.cluster_compositions(**call_kwargs)


@pytest.mark.parametrize("n_components", [2, 3])
def test_precomputed_coordinates(df_prop: pd.DataFrame, n_components: int) -> None:
    """A column of coordinates is plotted as is, without projector or embeddings."""
    coords = np.random.default_rng(seed=0).random((len(df_prop), n_components))
    df_prop["coords"] = list(coords)
    widget = pmv.cluster_compositions(
        df_prop, projection="coords", n_components=n_components
    )
    assert widget.projector is None
    assert widget.embeddings is None
    df_points = points(widget).loc[df_prop["composition"]]
    for dim, axis in enumerate("xyz"[:n_components]):
        np.testing.assert_array_equal(df_points[axis], coords[:, dim])
    assert widget.x_axis == {"label": "Component 1"}
    # partial axis options merge over the default axis label instead of dropping it
    widget = pmv.cluster_compositions(
        df_prop,
        projection="coords",
        n_components=n_components,
        x_axis={"range": [-3, 3]},
    )
    assert widget.x_axis == {"label": "Component 1", "range": [-3, 3]}


def test_custom_embedding_and_projection(df_prop: pd.DataFrame) -> None:
    """Custom callables get their kwargs and label axes by function name."""

    def my_embed(
        compositions: Sequence[str], scale: float = 1, **_kwargs: Any
    ) -> np.ndarray:
        """Deterministic scaled embedding."""
        return np.arange(len(compositions) * 4, dtype=float).reshape(-1, 4) * scale

    def my_project(
        embeddings: np.ndarray, n_components: int, **_kwargs: Any
    ) -> np.ndarray:
        """First n_components features."""
        return embeddings[:, :n_components]

    widget = pmv.cluster_compositions(
        df_prop,
        embedding_method=my_embed,
        projection=my_project,
        embedding_kwargs={"scale": 2},
    )
    assert widget.embeddings is not None
    np.testing.assert_array_equal(widget.embeddings, my_embed(df_prop.composition, 2))
    assert widget.projector is None
    assert widget.x_axis == {"label": "my_project 1"}
    np.testing.assert_array_equal(points(widget)["x"], widget.embeddings[:, 0])
    widget = pmv.cluster_compositions(
        df_prop, embedding_method=my_embed, projection=lambda emb, **_: emb[:, :2]
    )
    assert widget.x_axis == {"label": "Component 1"}  # lambdas get a generic label


def test_annotate_points(df_prop: pd.DataFrame) -> None:
    """annotate_points labels get each df row plus projected coordinates."""
    rows: list[pd.Series] = []

    def label(row: pd.Series) -> str | dict[str, str] | None:
        """Label positive-property points, with a dict for the largest."""
        rows.append(row)
        if row["property"] == 5:
            return {"text": "max", "font_size": "8px"}
        return row["composition"] if row["property"] > 0 else None

    widget = pmv.cluster_compositions(
        df_prop, prop_name="property", annotate_points=label, **BASE_KWARGS
    )
    assert {"pca1", "pca2", "composition", "category"} <= set(rows[0].index)
    labels = {
        comp: None if pd.isna(text) else text
        for comp, text in points(widget)["point_label"].items()
    }
    assert labels == {
        "Fe2O3": "Fe2O3",
        "FeO": "FeO",
        "Al2O3": "Al2O3",
        "Cu": None,
        "Au": "max",
        "Fe3O4": None,
    }


def test_cluster_compositions_kwargs(df_prop: pd.DataFrame) -> None:
    """Extra kwargs go to the widget, overriding defaults; NaN properties become
    None in metadata and colors.
    """
    df_prop.loc[1, "property"] = np.nan
    widget = pmv.cluster_compositions(
        df_prop,
        prop_name="property",
        sort=False,
        color_scale="interpolatePlasma",
        marker_size=12,
        color_bar={"title": "E (eV)"},
        **BASE_KWARGS,
    )
    assert isinstance(widget, ClusterScatterPlot)
    assert widget.color_scale["scheme"] == "interpolatePlasma"
    (srs,) = widget.series
    assert srs["point_style"] == {"radius": 6}
    assert widget.color_bar == {"title": "E (eV)"}
    assert srs["metadata"][1] == {"composition": "FeO", "property": None}
    assert srs["color_values"][1] is None
