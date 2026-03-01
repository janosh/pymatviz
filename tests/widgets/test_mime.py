"""Tests for MIME auto-display registry and create_widget."""

from __future__ import annotations

from typing import Any

import pytest
from pymatgen.core import Composition, Lattice, Structure

from pymatviz.widgets.band_structure import BandStructureWidget
from pymatviz.widgets.convex_hull import ConvexHullWidget
from pymatviz.widgets.dos import DosWidget
from pymatviz.widgets.xrd import XrdWidget


# === create_widget ===


@pytest.mark.parametrize(
    ("obj_factory", "expected_widget", "expected_type"),
    [
        (lambda: Composition("Fe2O3"), "CompositionWidget", "composition"),
        (
            lambda: __import__(
                "pymatgen.analysis.diffraction.xrd", fromlist=["DiffractionPattern"]
            ).DiffractionPattern(x=[10], y=[100], hkls=[], d_hkls=[]),
            "XrdWidget",
            "xrd",
        ),
    ],
)
def test_create_widget(
    obj_factory: Any, expected_widget: str, expected_type: str
) -> None:
    """create_widget produces correct widget for registered types."""
    from pymatviz.widgets.mime import create_widget

    widget = create_widget(obj_factory())
    assert type(widget).__name__ == expected_widget
    assert widget.widget_type == expected_type


def test_create_widget_phase_diagram_to_convex_hull() -> None:
    """create_widget maps PhaseDiagram -> ConvexHullWidget."""
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram

    from pymatviz.widgets.mime import create_widget

    widget = create_widget(
        PhaseDiagram(
            [
                PDEntry(Composition("Li"), -1.9),
                PDEntry(Composition("O"), -4.2),
                PDEntry(Composition("Li2O"), -14.3),
            ]
        )
    )
    assert type(widget).__name__ == "ConvexHullWidget"


def test_create_widget_trajectory_list_of_dicts() -> None:
    """create_widget trajectory fallback for list of dicts."""
    from pymatviz.widgets.mime import create_widget

    struct = Structure(Lattice.cubic(3), ["Fe"], [[0, 0, 0]])
    widget = create_widget([{"structure": struct, "energy": -1.0}] * 2)
    assert type(widget).__name__ == "TrajectoryWidget"


@pytest.mark.parametrize(
    ("bad_input", "match"),
    [(42, "No widget registered"), ([], "empty sequence"), ("string", "No widget")],
)
def test_create_widget_unknown_type_raises(bad_input: Any, match: str) -> None:
    """create_widget raises ValueError for unregistered/empty types."""
    from pymatviz.widgets.mime import create_widget

    with pytest.raises(ValueError, match=match):
        create_widget(bad_input)


# === _AUTO_DISPLAY registry ===


@pytest.mark.parametrize(
    ("class_name", "expected_widget", "expected_param"),
    [
        ("BandStructure", BandStructureWidget, "band_structure"),
        ("BandStructureSymmLine", BandStructureWidget, "band_structure"),
        ("Dos", DosWidget, "dos"),
        ("CompleteDos", DosWidget, "dos"),
        ("DiffractionPattern", XrdWidget, "patterns"),
        ("PhaseDiagram", ConvexHullWidget, "entries"),
        ("Structure", None, "structure"),
        ("Composition", None, "composition"),
    ],
)
def test_auto_display_registry(
    class_name: str, expected_widget: type | None, expected_param: str
) -> None:
    """_AUTO_DISPLAY maps pymatgen classes to correct widgets."""
    from pymatviz.widgets.mime import _AUTO_DISPLAY

    registered = {
        cls.__name__: (widget, param) for cls, (widget, param) in _AUTO_DISPLAY.items()
    }
    assert class_name in registered, f"{class_name} not registered"
    widget_cls, param_name = registered[class_name]
    assert param_name == expected_param
    if expected_widget is not None:
        assert widget_cls is expected_widget


def test_marimo_display_registered() -> None:
    """Registered classes expose callable _display_ that creates widgets."""
    from pymatviz.widgets.mime import _AUTO_DISPLAY, create_widget

    for cls in _AUTO_DISPLAY:
        display_method = getattr(cls, "_display_", None)
        assert callable(display_method), f"{cls.__name__} missing callable _display_"

    composition_obj = Composition("Fe2O3")
    widget = composition_obj._display_()
    expected_widget = create_widget(composition_obj)
    assert widget.widget_type == expected_widget.widget_type
    assert widget.composition == expected_widget.composition
    assert widget.widget_type == "composition"
