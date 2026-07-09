"""Lint the Python<->JS widget trait contract.

A Python trait the JS bridge (matterviz/extensions/anywidget/anywidget.ts)
doesn't drive is silently dead (historic examples: ``missing_color``,
``ScatterPlotWidget.show_legend``); a bridge key without a Python trait is a
prop users cannot set. This test loads the active ESM bundle in headless
Chromium, reads its exported ``WIDGET_MODEL_KEYS`` contract, and diffs it
against each widget's synced traits in both directions. Skips on bundles
predating the export (< 0.4.3).
"""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest

import pymatviz.widgets
from pymatviz.widgets.matterviz import MatterVizWidget


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("playwright") is None, reason="playwright not installed"
)

# Traits every widget syncs but that the JS side handles outside per-widget drive
# specs: widget_type selects the spec in render(); style/show_controls are base-drive
# props that some components legitimately don't consume (e.g. PeriodicTable has no
# control pane, so its spec omits show_controls from the drive deps).
_BASE_TRAITS = frozenset({"widget_type", "style", "show_controls"})

# Widgets whose constructors reject empty input
_MINIMAL_KWARGS: dict[str, dict[str, Any]] = {
    "FermiSurfaceWidget": {"fermi_data": {"isosurfaces": []}}
}

_IMPORT_CONTRACT_JS = """async (esm) => {
  const url = URL.createObjectURL(new Blob([esm], { type: "text/javascript" }));
  try { return (await import(url)).WIDGET_MODEL_KEYS ?? null }
  finally { URL.revokeObjectURL(url) }
}"""


def test_python_traits_match_js_contract() -> None:
    """Each widget's synced traits exactly mirror the JS bridge's model keys."""
    from pymatviz.widgets._headless import _get_browser
    from pymatviz.widgets.matterviz import fetch_widget_asset

    page = _get_browser().new_page()
    try:
        contract = page.evaluate(
            _IMPORT_CONTRACT_JS, fetch_widget_asset("matterviz.js")
        )
    finally:
        page.close()
    if contract is None:
        pytest.skip("bundle predates the WIDGET_MODEL_KEYS export (< 0.4.3)")

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
        if widget.widget_type not in contract:
            drift.append(f"{cls.__name__}: {widget.widget_type=} not in JS registry")
            continue
        py_traits = set(widget.to_dict()) - _BASE_TRAITS
        js_keys = set(contract[widget.widget_type]) - _BASE_TRAITS
        if dead := py_traits - js_keys:
            drift.append(
                f"{cls.__name__}: dead traits (JS never reads): {sorted(dead)}"
            )
        if missing := js_keys - py_traits:
            drift.append(
                f"{cls.__name__}: JS reads keys Python can't set: {sorted(missing)}"
            )
    assert not drift, (
        "Python<->JS widget trait contract drift (fix in matterviz/extensions/"
        "anywidget/anywidget.ts or the pymatviz widget class):\n" + "\n".join(drift)
    )
