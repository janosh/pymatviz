"""Tests for widget normalization functions in pymatviz.widgets._normalize."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pytest
from pymatgen.core import Composition, Lattice, Structure

from pymatviz.widgets._normalize import (
    _normalize_entry_compositions,
    _parse_formula_to_dict,
    _to_dict,
    normalize_convex_hull_entries,
    normalize_plot_series,
    normalize_structure_for_bz,
    normalize_xrd_pattern,
)


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


# === _parse_formula_to_dict ===


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("Li", {"Li": 1.0}),
        ("Li2O", {"Li": 2.0, "O": 1.0}),
        ("Fe2O3", {"Fe": 2.0, "O": 3.0}),
        ("LiFePO4", {"Li": 1.0, "Fe": 1.0, "P": 1.0, "O": 4.0}),
        ("O", {"O": 1.0}),
    ],
)
def test_parse_formula_to_dict(formula: str, expected: dict[str, float]) -> None:
    """Formula strings are parsed to element-count dicts."""
    assert _parse_formula_to_dict(formula) == expected


# === _normalize_entry_compositions ===


def test_normalize_entry_compositions_parses_strings() -> None:
    """String compositions are parsed to dicts, dict compositions pass through."""
    entries = [
        {"composition": "Li2O", "energy": -14.3},
        {"composition": {"Fe": 1}, "energy": -8.3},
        {"composition": "O", "energy": -4.9},
    ]
    result = _normalize_entry_compositions(entries)
    assert result[0]["composition"] == {"Li": 2.0, "O": 1.0}
    assert result[1]["composition"] == {"Fe": 1}
    assert result[2]["composition"] == {"O": 1.0}
    assert result[1] is entries[1]


def test_normalize_entry_compositions_empty() -> None:
    """Empty list passes through."""
    assert _normalize_entry_compositions([]) == []


# === normalize_convex_hull_entries ===


def test_normalize_convex_hull_entries_passthrough() -> None:
    """Test None and list passthrough."""
    assert normalize_convex_hull_entries(None) is None
    entries = [{"composition": {"Li": 1}, "energy": -1.5}]
    assert normalize_convex_hull_entries(entries) == entries
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


def test_normalize_convex_hull_entries_string_compositions() -> None:
    """String compositions in entry dicts are parsed to element-count dicts."""
    entries = [
        {"composition": "Li2O", "energy": -14.3},
        {"composition": "Fe", "energy": -8.3},
    ]
    result = normalize_convex_hull_entries(entries)
    assert result is not None
    assert result[0]["composition"] == {"Li": 2.0, "O": 1.0}
    assert result[1]["composition"] == {"Fe": 1.0}
    assert result[0]["energy"] == -14.3


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


def test_normalize_xrd_pattern_mixed_schema_prefers_complete_ferrox() -> None:
    """Test mixed-key dict uses Ferrox schema when canonical is incomplete."""
    mixed_dict = {
        "y": [10.0, 20.0],
        "two_theta": [30.0, 40.0],
        "intensities": [1.0, 2.0],
    }
    result = normalize_xrd_pattern(mixed_dict)
    assert result is not None
    assert result["x"] == [30.0, 40.0]
    assert result["y"] == [1.0, 2.0]


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


# === normalize_plot_series ===


def test_normalize_plot_series_from_numpy_arrays() -> None:
    """Series arrays from NumPy must normalize to finite Python floats."""
    normalized = normalize_plot_series(
        [{"x": np.array([0, 1, 2]), "y": np.array([0.1, 0.2, 0.3]), "label": "A"}],
        component_name="ScatterPlot",
    )
    assert normalized == [{"x": [0.0, 1.0, 2.0], "y": [0.1, 0.2, 0.3], "label": "A"}]


@pytest.mark.parametrize(
    ("series_data", "error_cls", "match"),
    [
        ("bad", TypeError, "must be a list/tuple"),
        ([{"x": [0, 1]}], ValueError, "must include keys 'x' and 'y'"),
        ([{"x": [0], "y": [0, 1]}], ValueError, "lengths must match"),
        ([{"x": [0, float("nan")], "y": [1, 2]}], ValueError, "non-finite"),
    ],
)
def test_normalize_plot_series_validation_errors(
    series_data: Any, error_cls: type[Exception], match: str
) -> None:
    """Invalid plot series payloads should fail with helpful messages."""
    with pytest.raises(error_cls, match=match):
        normalize_plot_series(series_data, component_name="ScatterPlot")
