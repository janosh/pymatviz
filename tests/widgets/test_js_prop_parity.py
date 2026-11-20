"""Lint the Python<->JS widget trait contract.

A Python trait the JS bridge (matterviz/extensions/anywidget/anywidget.ts)
doesn't read is silently dead (historic examples: ``missing_color``,
``ScatterPlotWidget.show_legend``); a bridge key without a Python trait is a
prop users cannot set. The bundle build writes its ``WIDGET_MODEL_KEYS`` contract
to ``widget-model-keys.json`` next to ``matterviz.js``; this test diffs it against
each widget's public synced traits in both directions.
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError

import pytest

import pymatviz.widgets
from pymatviz.widgets.matterviz import MatterVizWidget, fetch_widget_asset


# Widgets whose constructors reject empty input
_MINIMAL_KWARGS: dict[str, dict[str, Any]] = {
    "FermiSurfaceWidget": {"fermi_data": {"isosurfaces": []}}
}
CONTRACT_FILE = "widget-model-keys.json"


@pytest.fixture(scope="module")
def js_contract() -> dict[str, list[str]]:
    """The active bundle's {widget_type: model keys} contract: a local build's file
    when MATTERVIZ_ANYWIDGET_DIR is set (required there), else the pinned bundle's.
    Skips only when offline: a reachable host lacking the file fails the test.
    """
    if dev_dir := os.environ.get("MATTERVIZ_ANYWIDGET_DIR"):
        with open(f"{dev_dir}/{CONTRACT_FILE}", encoding="utf-8") as file:
            return json.load(file)
    try:
        return json.loads(fetch_widget_asset(CONTRACT_FILE))
    except FileNotFoundError as exc:
        cause = exc.__cause__
        if isinstance(cause, URLError) and not isinstance(cause, HTTPError):
            pytest.skip(f"offline, can't fetch {CONTRACT_FILE}: {cause}")
        raise


def test_python_traits_match_js_contract(js_contract: dict[str, list[str]]) -> None:
    """Each widget's public synced traits exactly mirror the JS bridge's model keys,
    except base traits (widget_type, style, show_controls) a spec may ignore.
    """
    base_keys = set(MatterVizWidget.public_trait_names())
    widget_classes = {
        cls
        for cls in vars(pymatviz.widgets).values()
        if isinstance(cls, type)
        and issubclass(cls, MatterVizWidget)
        and cls is not MatterVizWidget
    }
    drift: list[str] = []
    for cls in sorted(widget_classes, key=lambda cls: cls.__name__):
        widget = cls(**_MINIMAL_KWARGS.get(cls.__name__, {}))
        if widget.widget_type not in js_contract:
            drift.append(f"{cls.__name__}: {widget.widget_type=} not in JS registry")
            continue
        py_keys = set(widget._synced_keys)
        js_keys = set(js_contract[widget.widget_type])
        if dead := py_keys - js_keys - base_keys:
            drift.append(
                f"{cls.__name__}: dead traits (JS never reads): {sorted(dead)}"
            )
        if missing := js_keys - py_keys:
            drift.append(
                f"{cls.__name__}: JS reads keys Python can't set: {sorted(missing)}"
            )
    assert not drift, (
        "Python<->JS widget trait contract drift (fix in matterviz/extensions/"
        "anywidget/anywidget.ts or the pymatviz widget class):\n" + "\n".join(drift)
    )


def test_public_trait_names_exclude_framework_traits() -> None:
    """public_trait_names() drops anywidget/ipywidgets traits but keeps a subclass's
    redeclaration of one (TrajectoryWidget.layout) and the base widget traits.
    """
    from anywidget import AnyWidget

    from pymatviz.widgets import StructureWidget, TrajectoryWidget

    framework = set(AnyWidget.class_traits(sync=True))
    for cls in (MatterVizWidget, StructureWidget):
        names = set(cls.public_trait_names())
        assert {"widget_type", "style", "show_controls"} <= names
        assert not names & framework
        assert not any(name.startswith("_") for name in names)
    assert "layout" in TrajectoryWidget.public_trait_names()
    widget = StructureWidget()
    assert widget._synced_keys == StructureWidget.public_trait_names()
    assert set(widget.to_dict()) == set(widget._synced_keys)
