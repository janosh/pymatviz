"""Tests for widget construction, traitlet defaults, to_dict, and validation."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from pymatgen.core import Composition

from pymatviz.widgets.band_structure import BandStructureWidget
from pymatviz.widgets.bands_and_dos import BandsAndDosWidget
from pymatviz.widgets.bar_plot import BarPlotWidget
from pymatviz.widgets.brillouin_zone import BrillouinZoneWidget
from pymatviz.widgets.convex_hull import ConvexHullWidget
from pymatviz.widgets.dos import DosWidget
from pymatviz.widgets.fermi_surface import FermiSurfaceWidget
from pymatviz.widgets.histogram import HistogramWidget
from pymatviz.widgets.matterviz import MatterVizWidget
from pymatviz.widgets.phase_diagram import PhaseDiagramWidget
from pymatviz.widgets.scatter_plot import ScatterPlotWidget
from pymatviz.widgets.xrd import XrdWidget


# === Construction and widget_type ===


@pytest.mark.parametrize(
    ("widget_cls", "kwargs", "expected_type", "state_key", "expected_state"),
    [
        (
            ConvexHullWidget,
            {"entries": [{"composition": {"Li": 1}, "energy": -1.5}]},
            "convex_hull",
            "entries",
            [{"composition": {"Li": 1}, "energy": -1.5}],
        ),
        (
            BandStructureWidget,
            {"band_structure": {"bands": []}},
            "band_structure",
            "band_structure",
            {"bands": []},
        ),
        (DosWidget, {"dos": {"energies": [0]}}, "dos", "dos", {"energies": [0]}),
        (
            BandsAndDosWidget,
            {"band_structure": {"bands": []}, "dos": {"energies": []}},
            "bands_and_dos",
            "dos",
            {"energies": []},
        ),
        (
            FermiSurfaceWidget,
            {"fermi_data": {"isosurfaces": []}},
            "fermi_surface",
            "fermi_data",
            {"isosurfaces": []},
        ),
        (
            FermiSurfaceWidget,
            {"band_data": {"energies": []}},
            "fermi_surface",
            "band_data",
            {"energies": []},
        ),
        (
            BrillouinZoneWidget,
            {"structure": {"lattice": {}, "sites": []}},
            "brillouin_zone",
            "structure",
            {"lattice": {}, "sites": []},
        ),
        (
            PhaseDiagramWidget,
            {"data": {"components": ["A", "B"]}},
            "phase_diagram",
            "data",
            {"components": ["A", "B"]},
        ),
        (
            XrdWidget,
            {"patterns": {"x": [10], "y": [100]}},
            "xrd",
            "patterns",
            {"x": [10], "y": [100]},
        ),
        (
            ScatterPlotWidget,
            {"series": [{"x": [0, 1], "y": [1, 2], "label": "curve"}]},
            "scatter_plot",
            "series",
            [{"x": [0.0, 1.0], "y": [1.0, 2.0], "label": "curve"}],
        ),
        (
            BarPlotWidget,
            {"series": [{"x": [0, 1], "y": [2, 3], "label": "bars"}]},
            "bar_plot",
            "series",
            [{"x": [0.0, 1.0], "y": [2.0, 3.0], "label": "bars"}],
        ),
        (
            HistogramWidget,
            {"series": [{"x": [0, 1], "y": [2, 2.5], "label": "hist"}]},
            "histogram",
            "series",
            [{"x": [0.0, 1.0], "y": [2.0, 2.5], "label": "hist"}],
        ),
    ],
)
def test_widget_construction_and_type(
    widget_cls: type,
    kwargs: dict[str, Any],
    expected_type: str,
    state_key: str,
    expected_state: Any,
) -> None:
    """Widget constructors preserve expected normalized state."""
    widget = widget_cls(**kwargs)
    assert widget.widget_type == expected_type
    assert getattr(widget, state_key) == expected_state


# === to_dict auto-discovery ===


@pytest.mark.parametrize(
    ("widget_cls", "kwargs", "expected_fields"),
    [
        (
            ScatterPlotWidget,
            {"series": [{"x": [0, 1], "y": [1, 2], "label": "s"}]},
            {
                "series",
                "x_axis",
                "y_axis",
                "y2_axis",
                "display",
                "legend",
                "styles",
                "color_scale",
                "size_scale",
                "ref_lines",
                "fill_regions",
                "error_bands",
                "controls",
            },
        ),
        (
            HistogramWidget,
            {"series": [{"x": [0, 1], "y": [1, 2]}], "bins": 50, "mode": "overlay"},
            {
                "series",
                "bins",
                "mode",
                "selected_property",
                "show_legend",
                "x_axis",
                "y_axis",
                "y2_axis",
                "display",
                "legend",
                "bar",
                "ref_lines",
                "controls",
            },
        ),
        (
            BarPlotWidget,
            {"series": [{"x": [0], "y": [1]}], "mode": "grouped"},
            {
                "series",
                "orientation",
                "mode",
                "x_axis",
                "y_axis",
                "y2_axis",
                "display",
                "legend",
                "bar",
                "line",
                "ref_lines",
                "controls",
            },
        ),
        (
            DosWidget,
            {"dos": {"energies": [0, 1]}, "sigma": 0.05},
            {
                "dos",
                "stack",
                "sigma",
                "normalize",
                "orientation",
                "show_legend",
                "spin_mode",
            },
        ),
        (
            ConvexHullWidget,
            {"entries": [{"composition": {"Li": 1}, "energy": -1.5}]},
            {
                "entries",
                "show_stable",
                "show_unstable",
                "show_hull_faces",
                "hull_face_opacity",
                "show_stable_labels",
                "show_unstable_labels",
                "max_hull_dist_show_labels",
                "max_hull_dist_show_phases",
                "temperature",
            },
        ),
    ],
)
def test_to_dict_includes_subclass_fields(
    widget_cls: type,
    kwargs: dict[str, Any],
    expected_fields: set[str],
) -> None:
    """to_dict auto-discovers all synced traitlets including subclass-specific ones."""
    widget = widget_cls(**kwargs)
    state = widget.to_dict()

    base_fields = {"widget_type", "style", "show_controls"}
    assert base_fields <= set(state), f"Missing base fields in {widget_cls.__name__}"
    assert expected_fields <= set(state), (
        f"Missing subclass fields: {expected_fields - set(state)}"
    )
    assert not any(key.startswith("_") for key in state)
    non_synced_internals = {"comm", "keys", "log"}
    assert not non_synced_internals & set(state), (
        f"Non-synced traitlets leaked into to_dict: {non_synced_internals & set(state)}"
    )


def test_to_dict_reflects_constructor_values() -> None:
    """to_dict values match what was passed to the constructor."""
    widget = HistogramWidget(
        series=[{"x": [0, 1], "y": [2, 3], "label": "h"}],
        bins=25,
        mode="overlay",
        x_axis={"label": "Value"},
    )
    state = widget.to_dict()
    assert state["bins"] == 25
    assert state["mode"] == "overlay"
    assert state["series"] == [{"x": [0.0, 1.0], "y": [2.0, 3.0], "label": "h"}]
    assert state["x_axis"] == {"label": "Value"}


def test_to_dict_reflects_runtime_mutations() -> None:
    """to_dict picks up traitlet mutations after construction."""
    widget = ScatterPlotWidget(series=[{"x": [0], "y": [1]}])
    assert widget.to_dict()["x_axis"] is None

    widget.x_axis = {"label": "updated"}
    assert widget.to_dict()["x_axis"] == {"label": "updated"}

    widget.style = "height: 999px;"
    assert widget.to_dict()["style"] == "height: 999px;"


# === Traitlet defaults ===


@pytest.mark.parametrize(
    ("widget_cls", "attr", "expected"),
    [
        (BarPlotWidget, "orientation", "vertical"),
        (BarPlotWidget, "mode", "overlay"),
        (HistogramWidget, "bins", 100),
        (HistogramWidget, "mode", "single"),
        (HistogramWidget, "show_legend", True),
    ],
)
def test_plot_widget_traitlet_defaults(
    widget_cls: type, attr: str, expected: Any
) -> None:
    """Plot widgets expose correct default traitlet values."""
    widget = widget_cls(series=[{"x": [0, 1], "y": [1, 2]}])
    assert getattr(widget, attr) == expected


@pytest.mark.parametrize(
    ("widget_cls", "kwargs", "expected_attrs"),
    [
        (
            DosWidget,
            {"dos": {"energies": [0, 1]}, "sigma": 0.05, "spin_mode": "combined"},
            {
                "sigma": 0.05,
                "spin_mode": "combined",
                "show_controls": True,
                "stack": None,
            },
        ),
        (
            ConvexHullWidget,
            {
                "entries": [{"composition": {"Li": 1}, "energy": -1.5}],
                "show_unstable": False,
                "temperature": 300.0,
            },
            {
                "show_stable": True,
                "show_unstable": False,
                "temperature": 300.0,
                "hull_face_opacity": None,
            },
        ),
    ],
)
def test_widget_custom_and_default_traitlets(
    widget_cls: type,
    kwargs: dict[str, Any],
    expected_attrs: dict[str, Any],
) -> None:
    """Widgets preserve custom constructor values alongside correct defaults."""
    widget = widget_cls(**kwargs)
    for attr, expected in expected_attrs.items():
        assert getattr(widget, attr) == expected, f"{widget_cls.__name__}.{attr}"


# === Input normalization at widget level ===


def test_convex_hull_widget_from_phase_diagram() -> None:
    """ConvexHullWidget accepts a pymatgen PhaseDiagram directly."""
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram

    widget = ConvexHullWidget(
        entries=PhaseDiagram(
            [
                PDEntry(Composition("Fe"), -4.0),
                PDEntry(Composition("O"), -3.0),
                PDEntry(Composition("Fe2O3"), -25.0),
            ]
        )
    )
    assert widget.widget_type == "convex_hull"
    assert len(widget.entries) == 3
    assert all("composition" in entry for entry in widget.entries)


def test_xrd_widget_from_diffraction_pattern() -> None:
    """XrdWidget normalizes DiffractionPattern to plain floats."""
    from pymatgen.analysis.diffraction.xrd import DiffractionPattern

    widget = XrdWidget(
        patterns=DiffractionPattern(
            x=np.array([10.0, 20.0]),
            y=np.array([100.0, 50.0]),
            hkls=[[{"hkl": (1, 0, 0)}], [{"hkl": (1, 1, 0)}]],
            d_hkls=np.array([2.5, 2.0]),
        )
    )
    assert widget.patterns["x"] == [10.0, 20.0]
    assert type(widget.patterns["x"][0]) is float


def test_histogram_widget_delegates_series_validation() -> None:
    """HistogramWidget constructor delegates to normalize_plot_series validation."""
    with pytest.raises(ValueError, match="must include keys 'x' and 'y'"):
        HistogramWidget(series=[{"x": [0, 1]}])


# === FermiSurfaceWidget input validation ===


def test_fermi_surface_widget_dual_input() -> None:
    """FermiSurfaceWidget accepts exactly one data source."""
    assert FermiSurfaceWidget(fermi_data={"iso": []}).band_data is None
    assert FermiSurfaceWidget(band_data={"e": []}).fermi_data is None


@pytest.mark.parametrize(
    ("fermi_data", "band_data"),
    [
        (None, None),
        ({"isosurfaces": []}, {"energies": []}),
    ],
)
def test_fermi_surface_widget_invalid_input_combo(
    fermi_data: dict[str, Any] | None, band_data: dict[str, Any] | None
) -> None:
    """FermiSurfaceWidget rejects ambiguous or missing input sources."""
    with pytest.raises(ValueError, match="exactly one of fermi_data or band_data"):
        FermiSurfaceWidget(fermi_data=fermi_data, band_data=band_data)


# === show() method ===


def test_show_calls_ipython_display_and_returns_none() -> None:
    """show() calls IPython.display.display and returns None."""
    with patch("pymatviz.widgets.matterviz.fetch_widget_asset", return_value="x"):
        widget = MatterVizWidget(widget_type="test")

    with patch("IPython.display.display") as mock_display:
        result = widget.show()

    assert result is None
    mock_display.assert_called_once_with(widget)


@pytest.mark.parametrize(
    "widget_cls",
    [ScatterPlotWidget, BarPlotWidget, HistogramWidget],
)
def test_show_not_shadowed_by_display_traitlet(widget_cls: type) -> None:
    """show() remains callable on widgets that define a 'display' traitlet."""
    widget = widget_cls(
        series=[{"x": [0, 1], "y": [1, 2]}],
        display={"x_grid": True},
    )
    assert callable(widget.show)
    assert isinstance(widget.display, dict)

    with patch("IPython.display.display") as mock_display:
        result = widget.show()

    assert result is None
    mock_display.assert_called_once_with(widget)
