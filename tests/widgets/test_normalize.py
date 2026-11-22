"""Tests for widget normalization functions in pymatviz.widgets._normalize."""

from __future__ import annotations

import copy
import functools
import json
from enum import IntEnum
from types import SimpleNamespace
from typing import Any

import kemio
import numpy as np
import pandas as pd
import pytest

from pymatviz.widgets._normalize import (
    _formula_dict,
    _to_dict,
    normalize_convex_hull_entries,
    normalize_plot_json,
    normalize_plot_series,
    normalize_xrd_pattern,
    structure_to_dict,
    with_defaults,
)


_MOCK_DICT = {"@class": "Mock", "value": 42}
_HKLS = [[{"hkl": [1, 0, 0]}], [{"hkl": [1, 1, 0]}]]


class _MockMSONable:
    """Mock object with .as_dict() method."""

    def as_dict(self) -> dict[str, Any]:
        """Return a fresh copy of the mock dict."""
        return dict(_MOCK_DICT)


# === Passthrough and unsupported types ===


@pytest.mark.parametrize(
    ("normalize", "value"),
    [
        (normalize_xrd_pattern, None),
        (structure_to_dict, None),
        (structure_to_dict, {"sites": [], "lattice": {}}),
    ],
)
def test_normalizers_pass_through(normalize: Any, value: Any) -> None:
    """None and already-normalized dicts come back unchanged."""
    assert normalize(value) == value


@pytest.mark.parametrize(
    ("normalize", "bad_input", "match"),
    [
        (functools.partial(_to_dict, label="band structure"), 42, "band structure"),
        (functools.partial(_to_dict, label="DOS"), 42, "DOS"),
        (normalize_convex_hull_entries, "not_valid", "convex hull"),
        (normalize_xrd_pattern, 42, "XRD"),
    ],
)
def test_normalizers_reject_unsupported_types(
    normalize: Any, bad_input: Any, match: str
) -> None:
    """Normalizers raise TypeError naming the data kind for unsupported inputs."""
    with pytest.raises(TypeError, match=f"Unsupported type for {match}"):
        normalize(bad_input)


# === _to_dict ===


@pytest.mark.parametrize(
    ("input_val", "label", "expected"),
    [
        (None, "test", None),
        ({"key": "val"}, "test", {"key": "val"}),
        (_MockMSONable(), "test", _MOCK_DICT),
        (
            {"DFT": _MockMSONable(), "MACE": _MOCK_DICT},
            "DOS",
            {"DFT": _MOCK_DICT, "MACE": _MOCK_DICT},
        ),
        # dict inputs are JSON-normalized like converted ones (numpy to lists)
        (
            {"energies": np.array([0.5, 1.0]), "efermi": np.float64(0.1)},
            "DOS",
            {"energies": [0.5, 1.0], "efermi": 0.1},
        ),
    ],
)
def test_to_dict_passthrough(input_val: Any, label: str, expected: Any) -> None:
    """_to_dict handles None, dicts and MSONable objects, returning JSON types."""
    result = _to_dict(input_val, label)
    assert result == expected
    assert json.loads(json.dumps(result)) == result


def test_to_dict_drops_unread_fields() -> None:
    """as_dict() output loses projections/pdos/eigendisplacements, dict input not."""
    big = {"bands": [1], "projections": [2], "pdos": [3], "eigendisplacements": [4]}
    obj = type("Bands", (), {"as_dict": lambda _self: big})()
    assert _to_dict(obj, "band structure") == {"bands": [1]}
    assert _to_dict({"DFT": obj}, "band structure") == {"DFT": {"bands": [1]}}
    assert _to_dict(big, "band structure") == big


# === _formula_dict ===


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("Li", {"Li": 1.0}),
        ("Fe2O3", {"Fe": 2.0, "O": 3.0}),
        ("LiFePO4", {"Li": 1.0, "Fe": 1.0, "P": 1.0, "O": 4.0}),
        ("Ca(OH)2", {"Ca": 1.0, "O": 2.0, "H": 2.0}),  # groupings
        ("Fe0.5O", {"Fe": 0.5, "O": 1.0}),
    ],
)
def test_formula_dict(formula: str, expected: dict[str, float]) -> None:
    """Formula strings are parsed to element-amount dicts."""
    assert _formula_dict(formula) == expected


@pytest.mark.parametrize("formula", ["", "   ", "2LiO", "Li2O!", "Li2O)", "Xx2"])
def test_formula_dict_rejects_malformed(formula: str) -> None:
    """Malformed and empty formulas raise instead of silently dropping tokens."""
    with pytest.raises(ValueError, match=r"Formula must not be empty|invalid formula"):
        _formula_dict(formula)


# === normalize_convex_hull_entries ===


@pytest.mark.parametrize("container", [list, tuple])
def test_normalize_convex_hull_entry_dicts(container: type) -> None:
    """None passes through; entry dicts keep dict compositions (same object) and
    parse formula strings, returning a list.
    """
    assert normalize_convex_hull_entries(None) is None
    entries = [
        {"composition": "Li2O", "energy": -14.3},
        {"composition": {"Fe": 1}, "energy": -8.3},
        {"composition": "O", "energy": -4.9},
    ]
    result = normalize_convex_hull_entries(container(entries))
    assert result is not None
    assert result == [
        {"composition": {"Li": 2.0, "O": 1.0}, "energy": -14.3},
        {"composition": {"Fe": 1}, "energy": -8.3},
        {"composition": {"O": 1.0}, "energy": -4.9},
    ]
    assert result[1] is entries[1]


_HULL_ENTRIES = (("Li", -1.9), ("O", -4.2), ("Li2O", -14.3), ("Li2O2", -16.0))
_HULL_ENTRIES += (("Li3O", -16.3),)


@pytest.mark.parametrize("source", ["kemio", "objects", "generator"])
def test_normalize_convex_hull_entries_from_phase_diagram(source: str) -> None:
    """Kemio PhaseDiagrams, entry objects and generators give compositions, total
    energies, energies above hull, stability and entry ids in input order.
    """
    entries = [
        SimpleNamespace(composition=formula, energy=energy, entry_id=f"id-{formula}")
        for formula, energy in _HULL_ENTRIES
    ]
    compositions = [kemio.Composition(formula) for formula, _ in _HULL_ENTRIES]
    obj = {
        "kemio": kemio.PhaseDiagram(
            compositions,
            [
                entry.energy / comp.num_atoms
                for entry, comp in zip(entries, compositions, strict=True)
            ],
        ),
        "objects": entries,
        "generator": (entry for entry in entries),
    }[source]
    result = normalize_convex_hull_entries(obj)
    assert result is not None
    result_comps = [entry["composition"] for entry in result]
    assert result_comps == [comp.as_dict() for comp in compositions]
    np.testing.assert_allclose(  # energy per atom x atoms
        [entry["energy"] for entry in result],
        [energy for _, energy in _HULL_ENTRIES],
        rtol=4e-16,
        atol=0,
    )
    # reference values from pymatgen's PhaseDiagram (parity test in test_interop.py)
    np.testing.assert_allclose(
        [entry["e_above_hull"] for entry in result],
        [0, 0, 0, 0.625, 0],
        rtol=0,
        atol=1e-12,
    )
    assert [entry["is_stable"] for entry in result] == [True] * 3 + [False, True]
    ids = [entry.get("entry_id") for entry in result]
    assert ids == [None if source == "kemio" else f"id-{f}" for f, _ in _HULL_ENTRIES]


# === normalize_xrd_pattern ===


@pytest.mark.parametrize(
    ("pattern", "expected_kind"),
    [
        # measured scans: no Miller indices -> continuous profile
        ({"x": [10.0, 10.02], "y": [5.0, 7.0]}, "profile"),
        ({"x": [10.0, 10.02], "y": [5.0, 7.0], "hkls": []}, "profile"),
        # computed patterns carry hkls and stay untagged (matterviz sticks default)
        ({"x": [10.0, 20.0], "y": [100.0, 50.0], "hkls": _HKLS}, None),
        # explicit kind wins
        ({"x": [10.0, 20.0], "y": [100.0, 50.0], "kind": "sticks"}, "sticks"),
        ({"x": [10.0], "y": [1.0], "hkls": _HKLS[:1], "kind": "profile"}, "profile"),
    ],
)
def test_normalize_xrd_pattern_kind(
    pattern: dict[str, Any], expected_kind: str | None
) -> None:
    """Measured (hkl-less) patterns get kind='profile'; computed ones stay sticks."""
    input_copy = copy.deepcopy(pattern)
    result = normalize_xrd_pattern(pattern)
    assert result is not None
    assert result.get("kind") == expected_kind
    assert pattern == input_copy  # caller's dict not mutated


@pytest.mark.parametrize("bad_kind", ["line", "", None, "Profile"])
def test_normalize_xrd_pattern_rejects_bad_kind(bad_kind: str | None) -> None:
    """Unknown XRD kind values raise with the offending value in the message."""
    with pytest.raises(ValueError, match=f"kind={bad_kind!r} must be one of"):
        normalize_xrd_pattern({"x": [1.0], "y": [2.0], "kind": bad_kind})


@pytest.mark.parametrize(
    ("xrd_dict", "match"),
    [
        ({"x": [1.0, 2.0]}, r"missing \['y'\] in keys \['x'\]"),
        ({"x": [1.0], "y": [2.0, 3.0]}, "mismatched lengths"),
        (  # the old Ferrox keys are no longer read
            {"two_theta": [1.0], "intensities": [2.0]},
            r"missing \['x', 'y'\] in keys \['intensities', 'two_theta'\]",
        ),
    ],
)
def test_normalize_xrd_pattern_dict_validation_errors(
    xrd_dict: dict[str, list[float]], match: str
) -> None:
    """Pattern dicts without x and y or of mismatched lengths raise."""
    with pytest.raises(ValueError, match=match):
        normalize_xrd_pattern(xrd_dict)


def test_normalize_xrd_pattern_from_xrd_pattern() -> None:
    """XrdPatterns give plain floats/ints, sticks with hkls and profiles without."""
    two_theta, intensities = np.array([10.0, 20.0, 30.0]), np.array([100.0, 50, 25])
    result = normalize_xrd_pattern(
        kemio.XrdPattern(
            two_theta,
            intensities,
            d_spacings=np.array([2.5, 2.0, 1.5]),
            hkls=[[([1, 0, 0], 6)], [([1, 1, 0], 12)], [([1, 1, 1], 8)]],
        )
    )
    assert result is not None
    assert "kind" not in result
    measured = normalize_xrd_pattern(kemio.XrdPattern(two_theta, intensities))
    assert measured == {"x": [10.0, 20, 30], "y": [100.0, 50, 25], "kind": "profile"}
    assert result["x"] == [10.0, 20.0, 30.0]
    assert result["d_hkls"] == [2.5, 2.0, 1.5]
    assert type(result["x"][0]) is float
    assert type(result["d_hkls"][0]) is float
    assert result["hkls"][0] == [{"hkl": [1, 0, 0], "multiplicity": 6}]
    assert type(result["hkls"][0][0]["hkl"][0]) is int


# === structure_to_dict ===


@pytest.mark.parametrize("pbc", [(True,) * 3, (False,) * 3])
def test_structure_to_dict_objects(pbc: tuple[bool, bool, bool]) -> None:
    """Kemio crystals convert to dicts with lattice and sites, molecules without
    lattice.
    """
    lattice = np.eye(3) * 3 if any(pbc) else None
    result = structure_to_dict(kemio.Structure(lattice, ["Si"], [[0, 0, 0]], pbc=pbc))
    site = {
        "species": [{"element": "Si", "occu": 1}],
        "xyz": [0.0, 0.0, 0.0],
        "label": "Si",
        "properties": {},
    }
    if any(pbc):
        cell: dict[str, Any] = dict.fromkeys(("a", "b", "c"), 3.0)
        cell |= dict.fromkeys(("alpha", "beta", "gamma"), 90.0)
        cell |= {"volume": 27.0, "pbc": [True] * 3}
        assert result == {
            "@module": "pymatgen.core.structure",
            "@class": "Structure",
            "charge": 0.0,
            "lattice": {"matrix": (np.eye(3) * 3).tolist(), **cell},
            "properties": {},
            "sites": [{**site, "abc": [0.0, 0.0, 0.0]}],
        }
    else:
        assert result == {
            "@module": "pymatgen.core.structure",
            "@class": "Molecule",
            "charge": 0.0,
            "spin_multiplicity": 1,
            "properties": {},
            "sites": [{"name": "Si", **site}],
        }
    # periodic=True (Brillouin zone widgets) rejects lattice-less molecules
    if not any(pbc):
        with pytest.raises(TypeError, match=r"a periodic structure .* is needed"):
            structure_to_dict(result, periodic=True)


@pytest.mark.parametrize(
    ("bad_input", "err_cls", "match"),
    [
        (42, TypeError, "got int"),
        (kemio.Composition("Si"), TypeError, "got Composition"),  # has as_dict()
        (
            [kemio.Structure(np.eye(3) * 3, ["Si"], [[0, 0, 0]])],
            TypeError,
            "got list",
        ),
        ({"lattice": {}}, TypeError, r"'sites' list.*keys=\['lattice'\]"),
        ({"sites": "Si"}, TypeError, "'sites' list"),
    ],
)
def test_structure_to_dict_rejects_invalid(
    bad_input: Any, err_cls: type[Exception], match: str
) -> None:
    """Non-structure objects, sequences and dicts without a sites list raise."""
    with pytest.raises(err_cls, match=match):
        structure_to_dict(bad_input)


# === normalize_plot_series ===


@pytest.mark.parametrize(
    ("x_values", "y_values"),
    [
        (np.array([0, 1, 2]), np.array([0.1, 0.2, 0.3])),
        (pd.Series([0, 1, 2]), pd.Series([0.1, 0.2, 0.3])),
        # 1-element arrays must stay lists (.item() would collapse them to scalars)
        (np.array([0]), pd.Series([0.1])),
    ],
)
def test_normalize_plot_series_from_array_like_inputs(
    x_values: Any, y_values: Any
) -> None:
    """Array-backed series values normalize to Python primitives."""
    normalized = normalize_plot_series(
        [{"x": x_values, "y": y_values, "label": "A"}],
        component_name="ScatterPlot",
    )
    assert normalized is not None
    n_pts = len(x_values)
    assert normalized == [
        {"x": [0.0, 1.0, 2.0][:n_pts], "y": [0.1, 0.2, 0.3][:n_pts], "label": "A"}
    ]
    assert {type(val) for val in normalized[0]["x"]} == {float}  # ints widened


class _MockIntEnum(IntEnum):
    """Mock integer enum for JSON-safe int subclass normalization."""

    VALUE = 42


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (np.float64(4.5), 4.5),
        (_MockIntEnum.VALUE, 42),
        ({"obj": _MockMSONable()}, {"obj": _MOCK_DICT}),  # nested as_dict()
    ],
)
def test_normalize_plot_json_scalar_subclasses(value: Any, expected: Any) -> None:
    """Scalar subclasses and MSONable objects normalize to JSON-safe primitives."""
    normalized = normalize_plot_json(value, "test")
    assert normalized == expected
    assert type(normalized) is type(expected)


@pytest.mark.parametrize(
    ("value", "error_cls", "match"),
    [
        ({1: "a", "1": "b", 2: "c"}, ValueError, r"collide as strings: \[1, '1'\]"),
        ({"a": [1.0, np.inf]}, ValueError, "test.a contains non-finite float"),
        ([np.float32("nan")], ValueError, "non-finite float value: nan"),
        ({"a": {1, 2}}, TypeError, "Unsupported value in test.a"),
    ],
)
def test_normalize_plot_json_errors(
    value: Any, error_cls: type[Exception], match: str
) -> None:
    """Non-finite floats, unsupported types and string-colliding keys raise."""
    with pytest.raises(error_cls, match=match):
        normalize_plot_json(value, "test")


@pytest.mark.parametrize(
    ("series_data", "keys", "error_cls", "match"),
    [
        ("bad", ("x", "y"), TypeError, "must be a list/tuple"),
        ([{"x": [0, 1]}], ("x", "y"), ValueError, "must include keys 'x' and 'y'"),
        ([{"x": [0], "y": [0, 1]}], ("x", "y"), ValueError, "lengths must match"),
        ([{"x": [0, float("nan")], "y": [1, 2]}], ("x", "y"), ValueError, "non-finite"),
        (
            [{"x": ["a", 1], "y": [1, 2]}],
            ("x", "y"),
            TypeError,
            r"x values must be all strings \(categorical\) or all numeric, got mixed",
        ),
        # only x may be categorical
        ([{"x": [0, 1], "y": ["a", "b"]}], ("x", "y"), TypeError, r"y\[0\]='a'"),
        # Histograms require samples in `values`, including when x/y are present.
        ([[1, 2]], ("values",), TypeError, "entries must be dicts"),
        (
            [{"label": "no samples"}],
            ("values",),
            ValueError,
            "must include key 'values'",
        ),
        ([{"y": [1, 2]}], ("values",), ValueError, "must include key 'values'"),
        ([{"x": [0, 1], "y": [0, 1]}], ("values",), ValueError, "key 'values'"),
        ([{"values": 3}], ("values",), TypeError, "values must be list-like"),
        ([{"values": [1, "two"]}], ("values",), TypeError, "must be numeric"),
        (
            [{"x": [0], "y": [1], "z": [2, 3]}],
            tuple("xyz"),
            ValueError,
            "x/y/z lengths",
        ),
        ([{"values": [1, float("inf")]}], ("values",), ValueError, "finite"),
    ],
)
def test_normalize_plot_series_validation_errors(
    series_data: Any, keys: tuple[str, ...], error_cls: type[Exception], match: str
) -> None:
    """Invalid plot/histogram/3D series payloads fail with helpful messages."""
    with pytest.raises(error_cls, match=match):
        normalize_plot_series(series_data, component_name="Plot", keys=keys)


def test_normalize_histogram_series_values() -> None:
    """Histogram sample arrays and metadata are JSON-normalized."""
    hist = functools.partial(
        normalize_plot_series, component_name="Histogram", keys=("values",)
    )
    assert hist(None) is None
    assert hist(
        [
            {"values": np.array([3, 1, 2]), "label": "A", "color": "#ef4444"},
            {"values": pd.Series([0.5]), "visible": False},
            {"values": [], "label": "empty"},
        ]
    ) == [
        {"values": [3.0, 1.0, 2.0], "label": "A", "color": "#ef4444"},
        {"values": [0.5], "visible": False},
        {"values": [], "label": "empty"},
    ]


@pytest.mark.parametrize(
    ("defaults", "kwargs", "expected"),
    [
        # a dict option merges one level deep, so a partial axis keeps the default label
        (
            {"x_axis": {"label": "X", "format": "d"}},
            {"x_axis": {"scale_type": "log"}},
            {"x_axis": {"label": "X", "format": "d", "scale_type": "log"}},
        ),
        # user keys win on conflict, extra user options pass through
        (
            {"x_axis": {"label": "X"}, "gap": "4px"},
            {"x_axis": {"label": "Y"}, "show_legend": False},
            {"x_axis": {"label": "Y"}, "gap": "4px", "show_legend": False},
        ),
        # None or a non-dict replaces a dict default outright
        ({"color_bar": {"title": "T"}}, {"color_bar": None}, {"color_bar": None}),
        ({"x_range": [0, 1]}, {"x_range": [2, 3]}, {"x_range": [2, 3]}),
    ],
)
def test_with_defaults(
    defaults: dict[str, Any], kwargs: dict[str, Any], expected: dict[str, Any]
) -> None:
    """with_defaults merges dict options one level deep, other values replace."""
    defaults_copy, kwargs_copy = copy.deepcopy(defaults), copy.deepcopy(kwargs)
    assert with_defaults(defaults, kwargs) == expected
    assert (defaults, kwargs) == (defaults_copy, kwargs_copy)  # inputs left untouched
