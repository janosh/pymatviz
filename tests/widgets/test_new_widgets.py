"""Tests for new matterviz v0.3.1 widget classes and normalization."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pytest
from pymatgen.core import Composition, Lattice, Structure

from pymatviz.widgets._normalize import (
    _to_dict,
    normalize_convex_hull_entries,
    normalize_structure_for_bz,
    normalize_xrd_pattern,
)
from pymatviz.widgets.band_structure import BandStructureWidget
from pymatviz.widgets.bands_and_dos import BandsAndDosWidget
from pymatviz.widgets.brillouin_zone import BrillouinZoneWidget
from pymatviz.widgets.convex_hull import ConvexHullWidget
from pymatviz.widgets.dos import DosWidget
from pymatviz.widgets.fermi_surface import FermiSurfaceWidget
from pymatviz.widgets.phase_diagram import PhaseDiagramWidget
from pymatviz.widgets.xrd import XrdWidget


class _MockMSONable:
    """Mock object with .as_dict() method."""

    def as_dict(self) -> dict[str, Any]:
        return {"@class": "Mock", "value": 42}


_MOCK_DICT = {"@class": "Mock", "value": 42}


# === _to_dict ===


@pytest.mark.parametrize(
    ("input_val", "label", "expected"),
    [
        (None, "test", None),
        ({"key": "val"}, "test", {"key": "val"}),
        (_MockMSONable(), "test", _MOCK_DICT),
    ],
)
def test_to_dict_passthrough(input_val: Any, label: str, expected: Any) -> None:
    """Test _to_dict handles None, dict, and MSONable objects."""
    assert _to_dict(input_val, label) == expected


@pytest.mark.parametrize("label", ["band structure", "DOS"])
def test_to_dict_raises_for_unsupported_type(label: str) -> None:
    """Test _to_dict raises TypeError with descriptive label."""
    with pytest.raises(TypeError, match=f"Unsupported type for {label}"):
        _to_dict(42, label)


# === normalize_convex_hull_entries ===


def test_normalize_convex_hull_entries_passthrough() -> None:
    """Test None and list passthrough."""
    assert normalize_convex_hull_entries(None) is None
    entries = [{"composition": {"Li": 1}, "energy": -1.5}]
    assert normalize_convex_hull_entries(entries) is entries
    tuple_entries = tuple(entries)
    normalized_entries = normalize_convex_hull_entries(tuple_entries)
    assert isinstance(normalized_entries, list)
    assert normalized_entries == entries


def test_normalize_convex_hull_entries_from_phase_diagram() -> None:
    """Test conversion from pymatgen PhaseDiagram extracts correct fields."""
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram

    phase_diag = PhaseDiagram(
        [
            PDEntry(Composition("Li"), -1.9),
            PDEntry(Composition("O"), -4.2),
            PDEntry(Composition("Li2O"), -14.3),
        ]
    )
    result = normalize_convex_hull_entries(phase_diag)
    assert result is not None

    assert len(result) == 3
    for entry in result:
        assert set(entry) >= {
            "composition",
            "energy",
            "energy_per_atom",
            "e_above_hull",
            "is_stable",
        }
        assert isinstance(entry["composition"], dict)
        assert isinstance(entry["is_stable"], bool)
    assert any(entry["is_stable"] for entry in result)


def test_normalize_convex_hull_entries_unsupported_type() -> None:
    """Test TypeError for unsupported types."""
    with pytest.raises(TypeError, match="Unsupported type for convex hull"):
        normalize_convex_hull_entries("not_valid")


# === normalize_xrd_pattern ===


def test_normalize_xrd_pattern_passthrough() -> None:
    """Test None and dict passthrough."""
    assert normalize_xrd_pattern(None) is None
    test_dict = {"x": [1, 2], "y": [3, 4]}
    assert normalize_xrd_pattern(test_dict) is test_dict


@pytest.mark.parametrize(
    ("xrd_dict", "match"),
    [
        (
            {"x": [1.0, 2.0]},
            "Expected keys: \\['x', 'y'\\]",
        ),
        (
            {"x": [1.0], "y": [2.0, 3.0]},
            "mismatched canonical lengths",
        ),
        (
            {"two_theta": [1.0, 2.0]},
            "Expected keys: \\['two_theta', 'intensities'\\]",
        ),
        (
            {"two_theta": [1.0], "intensities": [2.0, 3.0]},
            "mismatched Ferrox lengths",
        ),
        (
            {"q": [1.0], "intensity": [2.0]},
            "Unsupported XRD dict schema",
        ),
    ],
)
def test_normalize_xrd_pattern_dict_validation_errors(
    xrd_dict: dict[str, list[float]], match: str
) -> None:
    """Test canonical/Ferrox dict validation raises descriptive errors."""
    with pytest.raises(ValueError, match=match):
        normalize_xrd_pattern(xrd_dict)


@pytest.mark.parametrize(
    ("hkls_in", "expected_hkls"),
    [
        (
            [[1, 0, 0], [1, 1, 0]],
            [[{"hkl": [1, 0, 0]}], [{"hkl": [1, 1, 0]}]],
        ),
        (
            [[[1, 0, 0]], [[1, 1, 0]]],
            [[{"hkl": [1, 0, 0]}], [{"hkl": [1, 1, 0]}]],
        ),
        (
            [[{"hkl": [1, 0, 0]}], [{"hkl": [1, 1, 0]}]],
            [[{"hkl": [1, 0, 0]}], [{"hkl": [1, 1, 0]}]],
        ),
    ],
)
def test_normalize_xrd_pattern_ferrox_hkls_shapes(
    hkls_in: list[Any], expected_hkls: list[list[dict[str, list[int]]]]
) -> None:
    """Test Ferrox hkls normalization for flat, nested, and passthrough forms."""
    ferrox_dict = {
        "two_theta": [10.0, 20.0],
        "intensities": [100.0, 50.0],
        "d_spacings": [2.5, 2.0],
        "hkls": hkls_in,
    }
    result = normalize_xrd_pattern(ferrox_dict)
    assert result is not None
    assert result["x"] == [10.0, 20.0]
    assert result["y"] == [100.0, 50.0]
    assert result["d_hkls"] == [2.5, 2.0]
    assert result["hkls"] == expected_hkls


def test_normalize_xrd_pattern_from_diffraction_pattern() -> None:
    """Test DiffractionPattern conversion produces plain floats/ints."""
    from pymatgen.analysis.diffraction.xrd import DiffractionPattern

    result = normalize_xrd_pattern(
        DiffractionPattern(
            x=np.array([10.0, 20.0, 30.0]),
            y=np.array([100.0, 50.0, 25.0]),
            hkls=[[{"hkl": (1, 0, 0)}], [{"hkl": (1, 1, 0)}], [{"hkl": (1, 1, 1)}]],
            d_hkls=np.array([2.5, 2.0, 1.5]),
        )
    )
    assert result is not None

    assert result["x"] == [10.0, 20.0, 30.0]
    assert result["d_hkls"] == [2.5, 2.0, 1.5]
    assert type(result["x"][0]) is float
    assert type(result["d_hkls"][0]) is float
    assert result["hkls"][0][0]["hkl"] == [1, 0, 0]
    assert type(result["hkls"][0][0]["hkl"][0]) is int


def test_normalize_xrd_pattern_unsupported_type() -> None:
    """Test TypeError for unsupported types."""
    with pytest.raises(TypeError, match="Unsupported type for XRD"):
        normalize_xrd_pattern(42)


# === normalize_structure_for_bz ===


def test_normalize_structure_for_bz_passthrough() -> None:
    """Test None and dict passthrough."""
    assert normalize_structure_for_bz(None) is None
    test_dict = {"sites": [], "lattice": {}}
    assert normalize_structure_for_bz(test_dict) is test_dict


@pytest.mark.parametrize(
    ("label", "make_obj"),
    [
        (
            "pymatgen Structure",
            lambda: Structure(Lattice.cubic(3), ["Si"], [[0, 0, 0]]),
        ),
        pytest.param(
            "ASE Atoms",
            lambda: __import__("ase.build", fromlist=["bulk"]).bulk(
                "Si", "diamond", a=5.43
            ),
            marks=pytest.mark.skipif(
                not importlib.util.find_spec("ase"),
                reason="ase not installed",
            ),
        ),
    ],
)
def test_normalize_structure_for_bz_objects(label: str, make_obj: Any) -> None:  # noqa: ARG001
    """Test conversion from pymatgen Structure and ASE Atoms."""
    result = normalize_structure_for_bz(make_obj())
    assert isinstance(result, dict)
    assert "lattice" in result
    assert "sites" in result


def test_normalize_structure_for_bz_unsupported_type() -> None:
    """Test TypeError for unsupported types."""
    with pytest.raises(TypeError, match="Unsupported type for Brillouin zone"):
        normalize_structure_for_bz(42)


# === Widget construction ===


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
    ],
)
def test_widget_construction_and_type(
    widget_cls: type,
    kwargs: dict[str, Any],
    expected_type: str,
    state_key: str,
    expected_state: Any,
) -> None:
    """Test widget constructors preserve expected normalized state."""
    widget = widget_cls(**kwargs)
    assert widget.widget_type == expected_type
    assert getattr(widget, state_key) == expected_state


def test_convex_hull_widget_from_phase_diagram() -> None:
    """Test ConvexHullWidget accepts a pymatgen PhaseDiagram directly."""
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


def test_dos_widget_traitlets() -> None:
    """Test DosWidget traitlet defaults and custom values."""
    widget = DosWidget(dos={"energies": [0, 1]}, sigma=0.05, spin_mode="combined")
    assert widget.sigma == 0.05
    assert widget.spin_mode == "combined"
    assert widget.show_controls is True
    assert widget.stack is None


def test_convex_hull_widget_traitlets() -> None:
    """Test ConvexHullWidget traitlet defaults and custom values."""
    widget = ConvexHullWidget(
        entries=[{"composition": {"Li": 1}, "energy": -1.5}],
        show_unstable=False,
        temperature=300.0,
    )
    assert widget.show_stable is True
    assert widget.show_unstable is False
    assert widget.temperature == 300.0
    assert widget.hull_face_opacity is None


def test_xrd_widget_from_diffraction_pattern() -> None:
    """Test XrdWidget normalizes DiffractionPattern to plain floats."""
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


def test_fermi_surface_widget_dual_input() -> None:
    """Test FermiSurfaceWidget accepts exactly one data source."""
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
    """Test FermiSurfaceWidget rejects ambiguous or missing input sources."""
    with pytest.raises(ValueError, match="exactly one of fermi_data or band_data"):
        FermiSurfaceWidget(fermi_data=fermi_data, band_data=band_data)


# === create_widget + registry ===


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
    """Test create_widget produces correct widget for registered types."""
    from pymatviz.widgets.mime import create_widget

    widget = create_widget(obj_factory())
    assert type(widget).__name__ == expected_widget
    assert widget.widget_type == expected_type


def test_create_widget_phase_diagram_to_convex_hull() -> None:
    """Test create_widget maps PhaseDiagram -> ConvexHullWidget."""
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
    """Test create_widget trajectory fallback for list of dicts."""
    from pymatviz.widgets.mime import create_widget

    struct = Structure(Lattice.cubic(3), ["Fe"], [[0, 0, 0]])
    widget = create_widget([{"structure": struct, "energy": -1.0}] * 2)
    assert type(widget).__name__ == "TrajectoryWidget"


@pytest.mark.parametrize(
    ("bad_input", "match"),
    [(42, "No widget registered"), ([], "empty sequence"), ("string", "No widget")],
)
def test_create_widget_unknown_type_raises(bad_input: Any, match: str) -> None:
    """Test create_widget raises ValueError for unregistered/empty types."""
    from pymatviz.widgets.mime import create_widget

    with pytest.raises(ValueError, match=match):
        create_widget(bad_input)


@pytest.mark.parametrize(
    ("class_name", "expected_widget", "expected_param"),
    [
        ("BandStructure", BandStructureWidget, "band_structure"),
        ("BandStructureSymmLine", BandStructureWidget, "band_structure"),
        ("Dos", DosWidget, "dos"),
        ("CompleteDos", DosWidget, "dos"),
        ("DiffractionPattern", XrdWidget, "patterns"),
        ("PhaseDiagram", ConvexHullWidget, "entries"),
        ("Structure", None, "structure"),  # just check presence
        ("Composition", None, "composition"),
    ],
)
def test_auto_display_registry(
    class_name: str, expected_widget: type | None, expected_param: str
) -> None:
    """Test _AUTO_DISPLAY maps pymatgen classes to correct widgets."""
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
    """Test _display_ is monkey-patched on all registered classes."""
    from pymatviz.widgets.mime import _AUTO_DISPLAY, create_widget

    for cls in _AUTO_DISPLAY:
        assert hasattr(cls, "_display_"), f"{cls.__name__} missing _display_"
        assert cls._display_ is create_widget
