"""Tests for BoxPlotWidget, SankeyWidget and CoordinationBarPlotWidget."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import traitlets as tl
from kemio.interop import structure_to_pymatgen_dict

from pymatviz.coordination import CnSplitMode
from pymatviz.widgets.box_plot import BoxPlotWidget
from pymatviz.widgets.coordination_bar_plot import CoordinationBarPlotWidget
from pymatviz.widgets.sankey import SankeyWidget
from tests.conftest import cubic_structure


@pytest.mark.parametrize(
    ("widget_cls", "widget_type"),
    [
        (BoxPlotWidget, "box_plot"),
        (SankeyWidget, "sankey"),
        (CoordinationBarPlotWidget, "coordination_bar_plot"),
    ],
)
def test_widget_type_and_defaults(widget_cls: type, widget_type: str) -> None:
    """Widgets construct empty, defer display defaults to the frontend."""
    widget = widget_cls()
    assert widget.widget_type == widget_type
    state = widget.to_dict()
    unset = {"orientation", "show_legend", "padding", "legend"}
    assert all(state[key] is None for key in unset & set(state))


def test_box_plot_series_normalized() -> None:
    """Numpy samples become JSON lists on init and on reassignment."""
    widget = BoxPlotWidget(
        [{"y": np.array([1, 2, 3]), "label": "a"}], kde_clip=[np.float64(0), None]
    )
    assert widget.series == [{"y": [1, 2, 3], "label": "a"}]
    assert widget.kde_clip == [0, None]
    widget.series = [{"y": np.arange(2.0)}]
    assert widget.series == [{"y": [0, 1]}]


@pytest.mark.parametrize(
    ("series", "match"),
    [
        ([{"x": [1]}], "key 'y'"),
        ([{"y": [1, float("nan")]}], "finite"),
        ({"y": [1]}, "list"),
    ],
)
def test_box_plot_rejects_invalid_series(series: Any, match: str) -> None:
    """Invalid series raise on init and leave assigned series unchanged."""
    with pytest.raises(tl.TraitError, match=match):
        BoxPlotWidget(series)
    widget = BoxPlotWidget([{"y": [1, 2]}])
    with pytest.raises(tl.TraitError, match=match):
        widget.series = series
    assert widget.series == [{"y": [1, 2]}]


@pytest.mark.parametrize(
    ("trait", "valid", "invalid"),
    [
        ("kind", "violin+box", "raincloud"),
        ("side", "positive", "left"),
        ("point_side", "negative", "below"),
        ("points", "swarm", "all"),
        ("whisker_mode", "percentile", "iqr"),
        ("bandwidth", "scott", "auto"),
        ("bandwidth", 0.3, "0.3"),
    ],
)
def test_box_plot_enum_traits(trait: str, valid: Any, invalid: Any) -> None:
    """Enum traits accept MatterViz names and reject others."""
    widget = BoxPlotWidget(**{trait: valid})
    assert getattr(widget, trait) == valid
    with pytest.raises(tl.TraitError):
        setattr(widget, trait, invalid)


def test_sankey_data() -> None:
    """Sankey data needs nodes and links lists; numpy values are normalized."""
    data = {
        "nodes": [{"label": "a"}, {"label": "b"}],
        "links": [{"source": 0, "target": 1, "value": np.int64(3)}],
    }
    widget = SankeyWidget(data, link_color_mode="gradient")
    assert widget.data["links"][0]["value"] == 3
    assert type(widget.data["links"][0]["value"]) is int
    with pytest.raises(tl.TraitError, match="'nodes' and 'links'"):
        widget.data = {"nodes": []}
    with pytest.raises(tl.TraitError, match="finite"):
        widget.data = {"nodes": [], "links": [{"value": float("nan")}]}
    with pytest.raises(tl.TraitError):
        widget.link_color_mode = "rainbow"


def test_coordination_structure_inputs() -> None:
    """Single structures reach JS as a bare structure dict, sequences and mappings
    as {label: dict}, the entry forms MatterViz's to_structure_entries reads.
    """
    si_struct = cubic_structure(3, ["Si"], [0, 0, 0])
    as_dict = structure_to_pymatgen_dict(si_struct)
    assert CoordinationBarPlotWidget(si_struct).structures == as_dict
    assert CoordinationBarPlotWidget(as_dict).structures == as_dict
    assert CoordinationBarPlotWidget([si_struct, as_dict]).structures == {
        "Structure 1": as_dict,
        "Structure 2": as_dict,
    }
    widget = CoordinationBarPlotWidget({1: si_struct}, strategy="explicit_only")
    assert widget.structures == {"1": as_dict}
    assert widget.strategy == "explicit_only"
    assert widget.mode == "grouped"
    for split_mode in CnSplitMode:  # pymatviz and MatterViz share split mode names
        widget.split_mode = split_mode.value
    with pytest.raises(tl.TraitError):
        widget.split_mode = "by_structure_and_element"
    widget.structures = si_struct  # assignment converts too, keeping JSON syncable
    assert widget.structures == as_dict
    with pytest.raises(tl.TraitError, match="structure must be"):
        CoordinationBarPlotWidget(["not a structure"])
