"""Tests for hierarchy widgets: SunburstWidget and TreemapWidget."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import traitlets as tl

from pymatviz.widgets.sunburst import SunburstWidget
from pymatviz.widgets.treemap import TreemapWidget


_ROOT_WITH_KIDS = {"label": "root", "children": [{"label": "leaf", "value": 3}]}


@pytest.mark.parametrize("widget_cls", [SunburstWidget, TreemapWidget])
@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"label": "Fe-O", "value": 4}, [{"label": "Fe-O", "value": 4}]),
        ({"label": "Fe-O", "children": []}, [{"label": "Fe-O", "children": []}]),
        (_ROOT_WITH_KIDS, _ROOT_WITH_KIDS),
        ([{"label": "Fe-O", "value": 4}], [{"label": "Fe-O", "value": 4}]),
    ],
)
def test_hierarchy_standalone_leaf_wrapped(
    widget_cls: type, data: Any, expected: Any
) -> None:
    """A standalone leaf node is wrapped in a list so it gets a visible cell."""
    widget = widget_cls(data=data)
    assert widget.data == expected
    widget.data = data  # assignment after construction normalizes too
    assert widget.data == expected


@pytest.mark.parametrize("widget_cls", [TreemapWidget, SunburstWidget])
def test_hierarchy_config_defaults(widget_cls: type) -> None:
    """Hierarchy config defaults to None so each renderer controls its own defaults."""
    base_traits = {"widget_type", "style", "show_controls", "controls_open"}
    state = widget_cls().to_dict()
    non_none = {key for key, val in state.items() if val is not None} - base_traits
    assert not non_none, f"expected None defaults, got non-None for {sorted(non_none)}"


@pytest.mark.parametrize(
    ("widget_cls", "geometry", "enum_key"),
    [
        (
            TreemapWidget,
            {
                "label_fit": "shrink",
                "label_min_font_size": 7,
                "label_max_font_size": 14,
                "parent_label_font_size": 16,
            },
            "label_fit",
        ),
        (
            SunburstWidget,
            {
                "shape": "icicle",
                "inner_radius": 0.2,
                "pad_angle": 0.1,
                "label_rotation": "radial",
            },
            "shape",
        ),
    ],
)
def test_hierarchy_widget_traits(
    widget_cls: type, geometry: dict[str, Any], enum_key: str
) -> None:
    """Both charts normalize hierarchy updates, validate chart options and sync zoom."""
    data = {"label": "root", "children": [{"label": "a", "value": np.float64(2.5)}]}
    config = {
        **geometry,
        "value_mode": "total",
        "label_text": "label+value",
        "max_children": 3,
        "export_formats": ["svg", "png"],
        "zoom_root_id": "root/a",
    }
    widget = widget_cls(data=data, **config)
    leaf_value = widget.data["children"][0]["value"]
    assert leaf_value == 2.5
    assert type(leaf_value) is float  # numpy scalar normalized
    state = widget.to_dict()
    for key, expected in config.items():
        assert state[key] == expected

    # zoom_root_id is two-way synced: assignment after construction must work
    widget.zoom_root_id = None
    assert widget.zoom_root_id is None

    for bad_key in ("value_mode", "label_text", enum_key):
        with pytest.raises(tl.TraitError, match=f"'{bad_key}' trait.*'not-an-option'"):
            widget_cls(data=data, **{bad_key: "not-an-option"})
    with pytest.raises(tl.TraitError, match=r"'max_children' trait.*less than 0"):
        widget_cls(data=data, max_children=-1)
    # not a matterviz ExportFormat
    with pytest.raises(tl.TraitError, match=r"'export_formats' trait.*'pdf'"):
        widget_cls(data=data, export_formats=["pdf"])
    with pytest.raises(TypeError, match="show_counts"):
        widget_cls(data=data, show_counts="value")
    assert widget_cls(data=data, export_formats=[]).export_formats == []

    widget.data = [{"label": "x", "value": np.int64(1)}, {"label": "y", "value": 2}]
    assert type(widget.to_dict()["data"][0]["value"]) is int
    assert widget.describe()["n_leaves"] == 2
