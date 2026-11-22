"""Tests for periodic table heatmaps and tile plots as PeriodicTableWidgets."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import pytest
import traitlets as tl

import pymatviz as pmv
from pymatviz.enums import ElemCountMode
from pymatviz.widgets.periodic_table import PeriodicTableWidget, normalize_tile_plots


def test_ptable_heatmap_values_and_defaults() -> None:
    """Finite values pass through, NaN/inf are dropped, color bar shown by default."""
    widget = pmv.ptable_heatmap({"Fe": 2, "O": 0.0, "Li": np.nan, "Na": np.inf})
    assert isinstance(widget, PeriodicTableWidget)
    assert widget.heatmap_values == {"Fe": 2.0, "O": 0.0}
    assert widget.color_scale == "interpolateViridis"
    assert widget.color_bar == {}
    assert widget.labels == {}


@pytest.mark.parametrize(
    ("count_mode", "expected"),
    [
        (ElemCountMode.composition, {"Fe": 3.0, "O": 4.0, "Li": 1.0}),
        (ElemCountMode.occurrence, {"Fe": 2.0, "O": 2.0, "Li": 1.0}),
        (ElemCountMode.reduced_composition, {"Fe": 3.0, "O": 4.0, "Li": 1.0}),
    ],
)
def test_ptable_heatmap_count_mode(
    count_mode: ElemCountMode, expected: dict[str, float]
) -> None:
    """Compositions are counted per count_mode."""
    widget = pmv.ptable_heatmap(["Fe2O3", "FeO", "Li"], count_mode=count_mode)
    assert widget.heatmap_values == expected


@pytest.mark.parametrize(
    ("heat_mode", "scale", "title"),
    [
        ("value", 1, "Count"),
        ("fraction", 1 / 4, "Count (fraction)"),
        ("percent", 25, "Count (%)"),
    ],
)
def test_ptable_heatmap_heat_mode(heat_mode: Any, scale: float, title: str) -> None:
    """Fraction/percent normalize by the finite sum, Series name titles color bar."""
    srs = pd.Series({"Fe": 1.0, "O": 3.0, "Li": np.nan}, name="Count")
    widget = pmv.ptable_heatmap(srs, heat_mode=heat_mode)
    expected = {"Fe": 1 * scale, "O": 3 * scale}
    np.testing.assert_allclose(
        [widget.heatmap_values[key] for key in expected],
        list(expected.values()),
        rtol=1e-15,
        atol=0,
    )
    assert widget.color_bar == {"title": title}


def test_ptable_heatmap_color_bar_kwargs() -> None:
    """User color bar props override the Series-name title; None hides the bar."""
    srs = pd.Series({"Fe": 1}, name="Count")
    assert pmv.ptable_heatmap(srs, color_bar={"title": "N"}).color_bar == {"title": "N"}
    assert pmv.ptable_heatmap(srs, color_bar=None).color_bar is None


@pytest.mark.parametrize(
    ("color_scale", "expected"),
    [
        ("interpolateMagma", {"scheme": "interpolateMagma", "type": "log"}),
        (
            {"scheme": "interpolatePlasma", "value_range": [1, None]},
            {"scheme": "interpolatePlasma", "value_range": [1, None], "type": "log"},
        ),
    ],
)
def test_ptable_heatmap_log(color_scale: Any, expected: dict[str, Any]) -> None:
    """log=True turns the color scale into a log-type config."""
    widget = pmv.ptable_heatmap(
        {"Fe": 10, "O": 1000}, log=True, color_scale=color_scale
    )
    assert widget.color_scale == expected


def test_ptable_heatmap_exclude_and_fill() -> None:
    """Excluded elements are dropped and labeled, fill_value covers absent elements."""
    widget = pmv.ptable_heatmap(
        {"Fe": 2, "O": 5, "N": 1},
        exclude_elements=["O", "N"],
        fill_value=0,
        labels={"Li": "x", "N": "user"},
    )
    assert "O" not in widget.heatmap_values
    assert widget.heatmap_values["Fe"] == 2
    excl_str = pmv.ptable_heatmap({"Fe": 1, "O": 2}, exclude_elements="O")
    assert "O" not in excl_str.heatmap_values
    assert widget.heatmap_values["Na"] == 0
    assert len(widget.heatmap_values) == 116
    assert widget.labels == {"O": "excl.", "N": "user", "Li": "x"}  # user label wins


@pytest.mark.parametrize(
    ("srs", "title"),
    [
        (pd.Series(["Fe2O3", "FeO"], name="composition"), "Share (%)"),  # not numeric
        (pd.Series({"Fe": 1.0}, name=0), "0 (%)"),  # falsy name kept
        (pd.Series({"Fe": 1.0}), "Share (%)"),  # unnamed
        ({"Fe": 1.0}, "Share (%)"),
    ],
)
def test_ptable_heatmap_series_title(srs: Any, title: str) -> None:
    """A named numeric Series titles the color bar, percent mode always shows (%)."""
    assert pmv.ptable_heatmap(srs, heat_mode="percent").color_bar == {"title": title}
    plain = pmv.ptable_heatmap(srs).color_bar
    assert plain == ({"title": "0"} if title == "0 (%)" else {})


@pytest.mark.parametrize(
    ("kwargs", "err_msg"),
    [
        (dict(log=True, heat_mode="percent"), "Combining log color scale"),
        (dict(heat_mode="bogus"), "heat_mode='bogus' must be"),
        (
            dict(log=True, color_scale={"type": "arcsinh"}),
            "log=True conflicts with color scale type arcsinh",
        ),
        (dict(values={"Fe": 0, "O": 0}, heat_mode="percent"), "nonzero sum"),
        (dict(values={"Fe": -1, "O": 1}, heat_mode="fraction"), "nonzero sum"),
        (dict(values={"Fe": np.nan}), "no finite values"),
        (dict(values={}), "no finite values"),
        (dict(heat_mode="fraction", fill_value=0), "would count toward the fraction"),
    ],
)
def test_ptable_heatmap_raises(kwargs: dict[str, Any], err_msg: str) -> None:
    """Invalid option combinations and data without colorable values raise."""
    values = kwargs.pop("values", {"Fe": 1})
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        pmv.ptable_heatmap(values, **kwargs)


@pytest.mark.parametrize("n_splits", [2, 3, 4])
def test_ptable_heatmap_splits(n_splits: int) -> None:
    """Split values become per-element lists with NaN as None."""
    data = {"Fe": np.arange(n_splits, dtype=float), "O": [np.nan] * n_splits}
    widget = pmv.ptable_heatmap_splits(data, color_bar={"title": "E"})
    assert widget.heatmap_values == {
        "Fe": list(map(float, range(n_splits))),
        "O": [None] * n_splits,
    }
    assert widget.color_bar == {"title": "E"}
    assert widget.color_scale == "interpolateViridis"


@pytest.mark.parametrize(
    ("n_splits", "split_layout"),
    [(2, "diagonal"), (3, "vertical"), (4, "triangular"), (4, None)],
)
def test_ptable_heatmap_splits_layout_and_tile_props(
    n_splits: int, split_layout: str | None
) -> None:
    """split_layout valid for the value count and tile_props pass to the widget."""
    widget = pmv.ptable_heatmap_splits(
        {"Fe": list(range(n_splits))},
        split_layout=split_layout,
        tile_props={"float_fmt": ".2f", "show_number": False},
    )
    assert widget.split_layout == split_layout
    assert widget.tile_props == {"float_fmt": ".2f", "show_number": False}


@pytest.mark.parametrize(
    ("n_splits", "split_layout"), [(2, "vertical"), (3, "quadrant"), (4, "diagonal")]
)
def test_ptable_heatmap_splits_invalid_layout(n_splits: int, split_layout: str) -> None:
    """A split_layout matterviz doesn't support for the value count raises."""
    with pytest.raises(ValueError, match=f"{split_layout=} invalid for {n_splits}"):
        pmv.ptable_heatmap_splits(
            {"Fe": list(range(n_splits))}, split_layout=split_layout
        )


@pytest.mark.parametrize(
    ("tile_props", "match"),
    [
        ({"show_symbol": np.False_}, None),  # numpy values normalized
        ({"role": "img", "tabindex": -1, "style": "opacity: 0.5"}, None),
        ({"aria-hidden": "true", "data-x": 1, "float_fmt": ".1f"}, None),
        *[
            ({key: "x"}, f"tile_props can't set \\['{key}'\\]")
            for key in [
                "active",
                "backdrop",
                "element",
                "href",
                "label",
                "segments",
                "split_layout",
            ]
        ],
        ({"show_nmae": False}, r"tile_props: unknown keys \['show_nmae'\]"),
        ({"onclick": "x"}, r"unknown keys \['onclick'\]"),
    ],
)
def test_tile_props_validation(tile_props: dict[str, Any], match: str | None) -> None:
    """tile_props allows ElementTile/HTML props, rejects table-set and unknown keys."""
    widget = PeriodicTableWidget()
    if match is None:
        widget.tile_props = tile_props
        assert widget.tile_props == tile_props
        assert not any(
            isinstance(val, np.generic) for val in widget.tile_props.values()
        )
        return
    with pytest.raises(tl.TraitError, match=match):
        widget.tile_props = tile_props
    assert widget.tile_props is None


def test_split_layout_rejects_unknown() -> None:
    """split_layout must be a layout matterviz knows."""
    with pytest.raises(tl.TraitError):
        PeriodicTableWidget(split_layout="grid")


def test_ptable_heatmap_splits_dataframe_and_series() -> None:
    """DataFrame rows (index = elements) and Series of lists give the same values."""
    df_splits = pd.DataFrame(
        {"a": [1.0, 2.0], "b": [3.0, 4.0]}, index=pd.Index(["Fe", "O"])
    )
    srs = pd.Series({"Fe": [1.0, 3.0], "O": [2.0, 4.0]})
    expected = {"Fe": [1.0, 3.0], "O": [2.0, 4.0]}
    assert pmv.ptable_heatmap_splits(df_splits).heatmap_values == expected
    assert pmv.ptable_heatmap_splits(srs).heatmap_values == expected


@pytest.mark.parametrize(
    ("data", "err_msg"),
    [
        ({}, "must not be empty"),
        ({"Fe": [1]}, "Number of splits 1 must be 2, 3, or 4"),
        ({"Fe": [1, 2, 3, 4, 5]}, "Number of splits 5 must be 2, 3, or 4"),
        ({"Fe": [1, 2], "O": [1, 2, 3]}, "O: got 3 values, expected 2"),
        ({"Fe": [1, 2], "Xx": [1, 2]}, "keys must be elements, got ['Xx']"),
        ({"Fe": [[1, 2]]}, "ptable_heatmap_splits: Fe must be a 1D numeric array"),
        # repeated keys were silently dropped by dict conversion
        (
            pd.DataFrame([[1, 2], [3, 4]], index=["Fe", "Fe"]),
            "element keys must be unique, got duplicates ['Fe']",
        ),
        (
            pd.Series([[1, 2], [3, 4]], index=["O", "O"]),
            "element keys must be unique, got duplicates ['O']",
        ),
    ],
)
def test_ptable_heatmap_splits_raises(data: Any, err_msg: str) -> None:
    """Empty data, bad split counts and unknown elements raise ValueError."""
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        pmv.ptable_heatmap_splits(data)


@pytest.mark.parametrize(
    "data",
    [
        {"Fe": [1.0, 2.0, np.nan], "O": np.array([3.0])},
        pd.Series({"Fe": [1.0, 2.0, np.nan], "O": [3.0]}),
        pd.DataFrame({"Fe": [1.0, 2.0, np.nan], "O": [3.0, np.nan, np.inf]}),
    ],
)
def test_ptable_hists_data(data: Any) -> None:
    """Dicts, Series and DataFrames bin to shared counts, skipping non-finite."""
    widget = pmv.ptable_hists(data)
    # bins of width 0.1 over [1, 3]: 2.0 opens bin 10, the right edge 3.0 closes bin 19
    fe_counts, o_counts = [0.0] * 20, [0.0] * 20
    fe_counts[0] = fe_counts[10] = o_counts[19] = 1.0
    assert widget.tile_plots == {"Fe": {"counts": fe_counts}, "O": {"counts": o_counts}}
    assert widget.tile_plot_config == {
        "bins": 20,
        "log_y": False,
        "x_range": [1.0, 3.0],
        "color_scale": "interpolateRdBu",
    }
    assert widget.heatmap_values is None


def test_ptable_hists_config_and_labels() -> None:
    """bins, x_range, log_y, color_scale and labels map onto widget props."""
    data = {"Fe": [1, 2, 3], "O": [4.0]}
    widget = pmv.ptable_hists(
        data,
        bins=7,
        x_range=(None, 3.5),
        log_y=True,
        color_scale=None,
        labels=lambda vals: f"n={len(vals)}" if len(vals) > 1 else None,
        color_bar={"title": "E"},
    )
    assert widget.tile_plot_config == {"bins": 7, "log_y": True, "x_range": [1.0, 3.5]}
    assert sum(widget.tile_plots["Fe"]["counts"]) == 3
    assert sum(widget.tile_plots["O"]["counts"]) == 0  # 4.0 lies outside x_range
    assert widget.labels == {"Fe": "n=3"}
    assert widget.color_bar == {"title": "E"}
    assert pmv.ptable_hists(data, labels={"Fe": "a"}).labels == {"Fe": "a"}


@pytest.mark.parametrize("bins", [0, 1001, True, 2.5])
def test_ptable_hists_invalid_bins(bins: Any) -> None:
    """Bin counts that aren't ints in matterviz's [1, 1000] raise."""
    with pytest.raises(tl.TraitError, match="bins must be an int in"):
        pmv.ptable_hists({"Fe": [1, 2]}, bins=bins)


@pytest.mark.parametrize(
    ("data", "err_msg"),
    [
        ({"Fe": [[1, 2], [3, 4]]}, "ptable_hists: Fe must be a 1D numeric array"),
        ({"Fe": 3}, "ptable_hists: Fe must be a 1D numeric array"),
        ({"Xx": [1]}, "keys must be element symbols, got ['Xx']"),
        (
            pd.DataFrame([[1, 2], [3, 4]], columns=["Fe", "Fe"]),
            "element keys must be unique, got duplicates ['Fe']",
        ),
    ],
)
def test_ptable_hists_raises(data: Any, err_msg: str) -> None:
    """2D or scalar values and unknown elements raise instead of being coerced."""
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        pmv.ptable_hists(data)


@pytest.mark.parametrize(
    ("data", "x_range", "expected"),
    [
        ({"Fe": [1, 2, np.inf]}, (None, 5), [1.0, 5]),  # inf ignored
        ({"Fe": [3, 3]}, (None, 10), [2.5, 10]),  # equal values padded by 0.5
        ({"Fe": [np.nan]}, (-1, None), [-1, 1.0]),  # no data: [0, 1]
        ({"Fe": [1, 2]}, (5, None), "x_range=(5, None) resolves to [5, 2.0]"),
        ({"Fe": [1, 2]}, (None, 0), "x_range=(None, 0) resolves to [1.0, 0]"),
        ({"Fe": [1, 2]}, (2, 2), "x_range=(2, 2) resolves to [2, 2]"),
    ],
)
def test_ptable_hists_x_range_resolved(
    data: dict[str, Any], x_range: tuple[Any, Any], expected: list[float] | str
) -> None:
    """None x_range ends are filled from finite data, lo >= hi raises ValueError."""
    if isinstance(expected, str):
        with pytest.raises(ValueError, match=re.escape(expected)):
            pmv.ptable_hists(data, x_range=x_range)
        return
    widget = pmv.ptable_hists(data, x_range=x_range)
    assert widget.tile_plot_config["x_range"] == expected


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(x_range=(5, None)), "x_range=(5, None) resolves to [5, 2.0]"),
        (dict(y_range=(None, -1)), "y_range=(None, -1) resolves to [3.0, -1]"),
    ],
)
def test_ptable_scatter_range_raises(kwargs: dict[str, Any], match: str) -> None:
    """One-sided ranges that end up lo >= hi raise in Python, not the frontend."""
    with pytest.raises(ValueError, match=re.escape(match)):
        pmv.ptable_scatter({"Fe": ([1, 2], [3, 4])}, **kwargs)


@pytest.mark.parametrize("mode", ["markers", "line", "line+markers"])
def test_ptable_scatter_modes_and_series(mode: Any) -> None:
    """(x, y[, color]) tuples become x/y series; mode passes to tile_plot_config."""
    data = {
        "Fe": ([1, 2], [3.0, np.nan]),
        "O": (np.array([1.0, 2.0]), np.array([4.0, 5.0]), [0.1, 0.2]),
    }
    widget = pmv.ptable_scatter(
        data, mode=mode, x_range=(0, None), y_range=(None, 9), color="red"
    )
    assert widget.tile_plots == {
        "Fe": {"x": [1.0, 2.0], "y": [3.0, None]},
        "O": {"x": [1.0, 2.0], "y": [4.0, 5.0], "color_values": [0.1, 0.2]},
    }
    assert widget.tile_plot_config == {
        "mode": mode,
        "color_scale": "interpolateViridis",
        "x_range": [0, 2.0],  # None ends filled from finite data
        "y_range": [3.0, 9],
        "color": "red",
    }


def test_ptable_scatter_named_series_and_labels() -> None:
    """Named series share per-name colors across elements unless colored by values."""
    data = {
        "Fe": {"b": ([1], [2]), "a": ([1], [3])},
        "O": {"a": ([1], [4], [0.5])},
    }
    widget = pmv.ptable_scatter(data, labels=lambda elem: f"{len(elem)} lines")
    fe_b, fe_a = widget.tile_plots["Fe"]
    assert fe_a["color"] == "#66c2a5"  # names are sorted: a first
    assert fe_b["color"] == "#fc8d62"
    assert "color" not in widget.tile_plots["O"][0]  # color_values win
    assert widget.labels == {"Fe": "2 lines", "O": "1 lines"}
    # names of mixed types sort by str instead of raising TypeError
    mixed_names = {"Fe": {1: ([1], [2]), "a": ([1], [3])}}
    mixed = pmv.ptable_scatter(mixed_names)  # ty: ignore[invalid-argument-type]
    assert [srs["color"] for srs in mixed.tile_plots["Fe"]] == ["#66c2a5", "#fc8d62"]


def test_ptable_scatter_without_color_values() -> None:
    """Without color_values no tile color scale is set, so heat values can be added.

    Also accepts (n, len) arrays as rows. With
    color_values, numeric heat values raise since matterviz has one color bar.
    """
    data = {
        "Fe": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "O": {"b": ([1], [2]), "a": ([1], [3])},
    }
    widget = pmv.ptable_scatter(data, heatmap_values={"Fe": 1})
    assert "color_scale" not in widget.tile_plot_config
    assert widget.heatmap_values == {"Fe": 1}
    with pytest.raises(tl.TraitError, match="not both"):
        pmv.ptable_scatter({"Fe": ([1], [2], [3])}, heatmap_values={"Fe": 1})
    hists = pmv.ptable_hists({"Fe": [1]}, color_scale=None, heatmap_values={"Fe": 1})
    assert hists.heatmap_values == {"Fe": 1}
    assert widget.tile_plots["Fe"] == {"x": [1.0, 2.0], "y": [3.0, 4.0]}
    assert [srs["color"] for srs in widget.tile_plots["O"]] == ["#fc8d62", "#66c2a5"]


@pytest.mark.parametrize(
    ("data", "kwargs", "err_type", "err_msg"),
    [
        ({"Fe": ([1], [2])}, dict(mode="lines"), tl.TraitError, "mode must be one of"),
        ({"Fe": ([1],)}, {}, ValueError, "must be an (x, y) or (x, y, color_values)"),
        ({"fe": ([1], [2])}, {}, ValueError, "keys must be element symbols"),
        ({"Fe": ([1], [2], ["red"])}, {}, ValueError, "must be a 1D numeric array:"),
        ({"Fe": ([1, 2], [1])}, {}, ValueError, "array lengths differ"),
    ],
)
def test_ptable_scatter_raises(
    data: dict[str, Any],
    kwargs: dict[str, Any],
    err_type: type[Exception],
    err_msg: str,
) -> None:
    """Bad modes, tuples, keys and non-numeric color values raise."""
    with pytest.raises(err_type, match=re.escape(err_msg)):
        pmv.ptable_scatter(data, **kwargs)


def test_normalize_tile_plots() -> None:
    """Numpy arrays become lists with non-finite as None, structure is kept."""
    plots = {
        "Fe": {"values": np.array([1, np.nan, np.inf]), "color": "red"},
        "O": [{"values": [2]}, {"values": (3.5,)}],
    }
    assert normalize_tile_plots(plots) == {
        "Fe": {"values": [1.0, None, None], "color": "red"},
        "O": [{"values": [2.0]}, {"values": [3.5]}],
    }
    assert normalize_tile_plots(None) is None
    # assignments after construction are normalized too
    widget = PeriodicTableWidget()
    widget.tile_plots = {"Fe": {"x": np.arange(2), "y": [np.nan, 1]}}
    assert widget.tile_plots == {"Fe": {"x": [0.0, 1.0], "y": [None, 1.0]}}


@pytest.mark.parametrize(
    ("plots", "err_type", "err_msg"),
    [
        ({"Fe": [1, 2]}, TypeError, "tile_plots['Fe'][0] must be a dict, got int"),
        ({"Fe": {"x": [1]}}, ValueError, "needs keys {values}, {counts} or {x, y"),
        ({"Fe": {"values": [1], "x": [1]}}, ValueError, "needs keys {values}, {c"),
        ({"Fe": {"counts": [1, -1]}}, ValueError, "counts must be finite and non-neg"),
        ({"Fe": {"counts": [1, np.nan]}}, ValueError, "counts must be finite"),
        ({"Fe": {"values": [1], "size": 2}}, ValueError, "got ['size', 'values']"),
        ({"Fe": {"values": 1}}, ValueError, "must be a 1D numeric array"),
        ({"Fe": {"values": [[1]]}}, ValueError, "must be a 1D numeric array"),
        ({"Xx": {"values": [1]}}, ValueError, "keys must be element symbols"),
        ({"Fe": {"values": [1], "color": 5}}, TypeError, "color must be a CSS color"),
        (
            {"Fe": {"x": [1, 2], "y": [1, 2], "color_values": [1]}},
            ValueError,
            "array lengths differ {'x': 2, 'y': 2, 'color_values': 1}",
        ),
        (
            {"Fe": {"values": [1]}, "O": {"x": [1], "y": [1]}},
            ValueError,
            "mixes ['histogram', 'x/y'] series",
        ),
        (
            {"Fe": {"values": [1]}, "O": {"counts": [1]}},
            ValueError,
            "mixes ['counts', 'histogram'] series",
        ),
    ],
)
def test_normalize_tile_plots_raises(
    plots: dict[str, Any], err_type: type[Exception], err_msg: str
) -> None:
    """Malformed series fail early in Python instead of throwing in the browser."""
    with pytest.raises(err_type, match=re.escape(err_msg)):
        normalize_tile_plots(plots)


@pytest.mark.parametrize(
    "config",
    [
        {"mode": "lines"},
        {"bins": 2.5},
        {"color_scale": "viridis"},
        {"x_range": [1, 2, 3]},
        {"x_range": [2, 1]},
        {"y_range": [1, 1]},
        {"x_range": ["a", None]},
        {"x_range": [False, True]},
        {"color_scale": {"scheme": "viridis"}},
        {"log_y": "yes"},
        {"bnis": 5},
        {"color": 1},
    ],
)
def test_tile_plot_config_validation(config: dict[str, Any]) -> None:
    """tile_plot_config rejects values and keys matterviz would choke on or ignore."""
    with pytest.raises(tl.TraitError):
        PeriodicTableWidget(tile_plot_config=config)


def test_tile_plot_config_post_init_and_color_scale_conflict() -> None:
    """Assignments normalize numpy scalars; one color scale per table is enforced."""
    widget = PeriodicTableWidget()
    widget.tile_plot_config = {"bins": np.int64(5), "x_range": (np.float32(0), None)}
    assert widget.tile_plot_config == {"bins": 5, "x_range": [0.0, None]}
    with pytest.raises(tl.TraitError, match="not both"):
        PeriodicTableWidget(
            heatmap_values={"Fe": [1, None]},
            tile_plot_config={"color_scale": "interpolateViridis"},
        )
    widget.tile_plot_config = {"color_scale": "interpolateViridis"}
    for heat_values in ({"Fe": 1}, {"Fe": np.int64(1)}, np.ones(3), {"Fe": " 1.5"}):
        with pytest.raises(tl.TraitError, match="not both"):
            widget.heatmap_values = heat_values
    widget.heatmap_values = {"Fe": "red", "O": None}  # colors don't need the scale
    # reverse order: heat values first, then a config with a color scale
    widget = PeriodicTableWidget(heatmap_values={"Fe": 1})
    with pytest.raises(tl.TraitError, match="not both"):
        widget.tile_plot_config = {"color_scale": "interpolateViridis"}
    widget.heatmap_values = {"Fe": np.int64(2)}  # normalized to JSON on assignment
    assert type(widget.heatmap_values["Fe"]) is int


def test_default_style() -> None:
    """Width-capped default style keeps tile rows from overlapping in wide cells."""
    widget = PeriodicTableWidget()
    assert widget.style == "width: 100%; max-width: 900px; aspect-ratio: 18/10;"
    assert PeriodicTableWidget(style="height: 300px").style == "height: 300px"
