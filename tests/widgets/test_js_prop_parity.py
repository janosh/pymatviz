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


_skip_no_playwright = pytest.mark.skipif(
    importlib.util.find_spec("playwright") is None, reason="playwright not installed"
)

# Trait names the matterviz-anywidget bridge no longer reads (removed or renamed in
# matterviz 0.5/0.6, see its changelog). Keyed by widget_type; the JS bridge would
# silently ignore any of these, so no widget may declare them. Pure Python check
# (no browser), so it runs even where the bundle-based parity test below skips.
_REMOVED_TRAITS: dict[str, frozenset[str]] = {
    # 0.6.0: show_gizmo -> gizmo (bool | GizmoOptions)
    "structure": frozenset({"show_gizmo"}),
    "trajectory": frozenset({"show_gizmo"}),
    # 0.6.0: BarPlot.point_tween removed
    "bar_plot": frozenset({"point_tween"}),
    # 0.5.0: `controls` dict -> controls_open/controls_toggle_props/controls_pane_props;
    # point_events is a record of JS callbacks and cannot cross the JSON bridge
    "scatter_plot": frozenset({"controls", "point_events"}),
    "scatter_plot_3d": frozenset({"controls", "on_series_visibility_change"}),
    "rdf_plot": frozenset({"controls"}),
    "xrd": frozenset({"controls"}),
    # 0.5.0: log_scale -> log
    "periodic_table": frozenset({"log_scale"}),
    "heatmap_matrix": frozenset(
        {
            "log_scale",
            "theme",
            "animate_updates",
            "show_gridlines",
            "value_transform",
            "quantile_clip",
            "legend_ticks",
        }
    ),
    # 0.5.0: band_structure -> band_structs, dos -> doses
    "band_structure": frozenset({"band_structure"}),
    "dos": frozenset({"dos"}),
    "bands_and_dos": frozenset({"band_structure", "dos"}),
    # 0.6.0: Composition.{display_mode,color_scheme} settings removed
    "composition": frozenset({"display_mode"}),
}


def _widget_classes() -> list[type[MatterVizWidget]]:
    """Every public MatterVizWidget subclass, sorted by name."""
    return sorted(
        (
            cls
            for cls in vars(pymatviz.widgets).values()
            if isinstance(cls, type)
            and issubclass(cls, MatterVizWidget)
            and cls is not MatterVizWidget
        ),
        key=lambda cls: cls.__name__,
    )


@pytest.mark.parametrize("widget_cls", _widget_classes(), ids=lambda cls: cls.__name__)
def test_no_removed_matterviz_traits(widget_cls: type[MatterVizWidget]) -> None:
    """No widget syncs a trait name matterviz 0.6.0 no longer accepts."""
    widget = widget_cls(**_MINIMAL_KWARGS.get(widget_cls.__name__, {}))
    assert widget.widget_type is not None
    removed = set(widget.to_dict()) & _REMOVED_TRAITS.get(widget.widget_type, set())
    assert not removed, f"{widget_cls.__name__} syncs removed traits {sorted(removed)}"


# Traits every widget syncs but that the JS side handles outside per-widget drive
# specs: widget_type selects the spec in render(); style/show_controls are base-drive
# props that some components legitimately don't consume (e.g. PeriodicTable has no
# control pane, so its spec omits show_controls from the drive deps).
_BASE_TRAITS = frozenset({"widget_type", "style", "show_controls"})

# Widgets whose constructors reject empty input
_MINIMAL_KWARGS: dict[str, dict[str, Any]] = {
    "FermiSurfaceWidget": {"fermi_data": {"isosurfaces": []}}
}

# Traits added to pymatviz ahead of the matterviz PR that reads them, so a bundle
# built from matterviz main before that PR lands may not list them yet. Remove each
# entry once the key shows up in the bundle's WIDGET_MODEL_KEYS (the test then
# enforces it like any other trait).
_PENDING_JS_KEYS: dict[str, frozenset[str]] = {
    # matterviz PR after v0.6.0: atom_type_mapping folded into
    # TrajectoryFileViewer.loading_options (names LAMMPS atom types)
    "trajectory": frozenset({"atom_type_mapping"}),
}

_IMPORT_CONTRACT_JS = """async (esm) => {
  const url = URL.createObjectURL(new Blob([esm], { type: "text/javascript" }));
  try { return (await import(url)).WIDGET_MODEL_KEYS ?? null }
  finally { URL.revokeObjectURL(url) }
}"""


@_skip_no_playwright
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

    drift: list[str] = []
    for cls in _widget_classes():
        widget = cls(**_MINIMAL_KWARGS.get(cls.__name__, {}))
        if widget.widget_type not in contract:
            drift.append(f"{cls.__name__}: {widget.widget_type=} not in JS registry")
            continue
        py_traits = set(widget.to_dict()) - _BASE_TRAITS
        js_keys = set(contract[widget.widget_type]) - _BASE_TRAITS
        pending = _PENDING_JS_KEYS.get(widget.widget_type, frozenset())
        if dead := py_traits - js_keys - pending:
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
