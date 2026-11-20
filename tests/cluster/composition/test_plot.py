"""Unit tests for composition cluster plots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
from pymatgen.core import Composition
from sklearn.decomposition import PCA

import pymatviz as pmv
from pymatviz.cluster.composition import ProjectionMethod, project_vectors
from pymatviz.cluster.composition.plot import (
    CHEM_SYS_COLORS,
    CHEM_SYS_SYMBOLS,
    ClusterScatterPlot,
    ClusterScatterPlot3D,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatviz.cluster.composition.plot import ColorScale, ShowChemSys

np_rng = np.random.default_rng(seed=0)
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
@pytest.mark.parametrize("sort", [True, False, -1])
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
    systems = {comp: Composition(comp).chemical_system for comp in df_prop.composition}
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


def test_categorical_property(df_prop: pd.DataFrame) -> None:
    """Non-numeric properties (missing values included) split points into one
    series per category, colored per category also when split by system shape.
    """
    df_prop.loc[5, "category"] = None
    widget = pmv.cluster_compositions(df_prop, prop_name="category", **BASE_KWARGS)
    assert [srs["label"] for srs in widget.series] == ["a", "b", "nan"]
    shapes = pmv.cluster_compositions(
        df_prop, prop_name="category", show_chem_sys="shape", **BASE_KWARGS
    )
    colors = {srs["label"]: srs["color"] for srs in shapes.series}
    assert colors["a · Fe-O"] == colors["a · Al-O"] != colors["b · Fe-O"]
    assert widget.color_scale is None
    assert all("color_values" not in srs for srs in widget.series)


@pytest.mark.parametrize(
    ("projection", "expected_stats"),
    [
        ("tsne", "perplexity = 2\nlearning_rate = 50"),
        ("isomap", "n_neighbors = 3\nmetric = 'minkowski'"),
        ("kernel_pca", "kernel = 'rbf'\ngamma = 0.1"),
    ],
)
def test_projection_stats(
    df_prop: pd.DataFrame, projection: str, expected_stats: str
) -> None:
    """Non-PCA projection parameters show in the 2D annotation badge."""
    projection_kwargs = {
        "tsne": dict(perplexity=2, learning_rate=50),
        "isomap": dict(n_neighbors=3),
        "kernel_pca": dict(gamma=0.1),
    }[projection]
    kwargs: dict[str, Any] = dict(
        embedding_method="one-hot",
        projection=projection,
        projection_kwargs=projection_kwargs,
    )
    for show_stats, expected in ((True, expected_stats), (False, None)):
        widget = pmv.cluster_compositions(
            df_prop, show_projection_stats=show_stats, **kwargs
        )
        assert isinstance(widget, ClusterScatterPlot)
        assert widget.annotation == expected


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
                "df_in": pd.DataFrame({"composition": ["Fe", "Cu"], "prop": [-1, 0]}),
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
    ],
)
def test_cluster_compositions_invalid_inputs(
    df_prop: pd.DataFrame, kwargs: dict[str, Any], err_type: type[Exception], match: str
) -> None:
    """Invalid arguments raise informative errors."""
    call_kwargs: dict[str, Any] = {"df_in": df_prop, **BASE_KWARGS, **kwargs}
    with pytest.raises(err_type, match=match):
        pmv.cluster_compositions(**call_kwargs)


@pytest.mark.parametrize("n_components", [2, 3])
def test_precomputed_coordinates(df_prop: pd.DataFrame, n_components: int) -> None:
    """A column of coordinates is plotted as is, without projector or embeddings."""
    coords = np_rng.random((len(df_prop), n_components))
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


def test_annotate_points(df_prop: pd.DataFrame) -> None:
    """annotate_points labels get each df_in row plus projected coordinates."""
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


def test_non_finite_property_values(df_prop: pd.DataFrame) -> None:
    """NaN/None property values become None in metadata and colors."""
    df_prop.loc[1, "property"] = np.nan
    widget = pmv.cluster_compositions(
        df_prop, prop_name="property", sort=False, **BASE_KWARGS
    )
    assert widget.series[0]["metadata"][1] == {"composition": "FeO", "property": None}
    assert widget.series[0]["color_values"][1] is None


def test_cluster_compositions_kwargs(df_prop: pd.DataFrame) -> None:
    """Extra keyword arguments go to the widget, overriding defaults."""
    widget = pmv.cluster_compositions(
        df_prop,
        prop_name="property",
        color_scale="interpolatePlasma",
        marker_size=12,
        color_bar={"title": "E (eV)"},
        **BASE_KWARGS,
    )
    assert isinstance(widget, ClusterScatterPlot)
    assert widget.color_scale["scheme"] == "interpolatePlasma"
    assert widget.series[0]["point_style"] == {"radius": 6}
    assert widget.color_bar == {"title": "E (eV)"}
