from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import matscikit as mk
import numpy as np
import pytest
from matscikit.io import read_structure

import pymatviz as pmv
from pymatviz.utils.testing import TEST_FILES
from pymatviz.xrd import cu_k_alpha_wavelength


if TYPE_CHECKING:
    from typing import Any

MOCK_HKLS = [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 0, 0), (2, 1, 0)]
MOCK_XRD_PATTERN = mk.XrdPattern(
    [10, 20, 30, 40, 50],
    [100, 50, 75, 25, 60],
    [2.5, 2.0, 1.8, 1.5, 1.3],
    [[(list(hkl), 1)] for hkl in MOCK_HKLS],
)

BI2_ZR2_O7_STRUCT = read_structure(
    f"{TEST_FILES}/xrd/Bi2Zr2O7-Fm3m-experimental-sqs.cif"
)
(BI2_ZR2_O7_XRD,) = mk.xrd_patterns([BI2_ZR2_O7_STRUCT])


def single_peak_pattern(intensity: float) -> mk.XrdPattern:
    """One-peak XrdPattern with the given intensity."""
    return mk.XrdPattern([10], [intensity], [1], [[([1, 0, 0], 1)]])


@pytest.mark.parametrize(
    ("input_data", "expected"),
    [
        (MOCK_XRD_PATTERN, MOCK_XRD_PATTERN),
        (BI2_ZR2_O7_XRD, BI2_ZR2_O7_XRD),
        (BI2_ZR2_O7_STRUCT, BI2_ZR2_O7_XRD),
    ],
)
def test_xrd_pattern_single(input_data: Any, expected: mk.XrdPattern) -> None:
    """A single pattern or structure is passed as one unlabeled pattern."""
    widget = pmv.xrd_plot(input_data, style="height: 300px")
    assert isinstance(widget, pmv.widgets.XrdWidget)
    assert widget.style == "height: 300px"
    for key, values in (("x", expected.two_theta), ("y", expected.intensities)):
        np.testing.assert_array_equal(widget.patterns[key], values)
    assert len(widget.patterns["hkls"]) == len(expected.hkls or [])


def test_xrd_pattern_mapping() -> None:
    """Mappings keep label order, compute structures and pass patterns (objects or
    dicts) through.
    """
    pattern_dict = {"x": [10, 20], "y": [100, 50], "hkls": [[{"hkl": (1, 0, 0)}]] * 2}
    patterns = OrderedDict(
        [
            ("Structure", BI2_ZR2_O7_STRUCT),
            ("Pattern", MOCK_XRD_PATTERN),
            ("Dict", pattern_dict),
        ]
    )
    widget = pmv.xrd_plot(patterns)
    assert list(widget.patterns) == ["Structure", "Pattern", "Dict"]
    assert widget.patterns["Dict"]["y"] == [100, 50]
    np.testing.assert_array_equal(
        widget.patterns["Structure"]["x"], BI2_ZR2_O7_XRD.two_theta
    )
    assert widget.patterns["Pattern"]["d_hkls"] == [2.5, 2.0, 1.8, 1.5, 1.3]
    assert [peak[0]["hkl"] for peak in widget.patterns["Pattern"]["hkls"]] == [
        list(hkl) for hkl in MOCK_HKLS
    ]


@pytest.mark.parametrize(
    ("patterns", "err_type", "match"),
    [
        (mk.XrdPattern([], []), ValueError, "No intensities found"),
        ({}, ValueError, "patterns must not be empty"),
        ("invalid", TypeError, "'XRD Pattern': expected a structure"),
        ({"bad": "invalid"}, TypeError, "'bad': expected a structure"),
        *[
            (single_peak_pattern(val), *err)
            for val, err in [
                (0, (ValueError, "At least one diffraction intensity")),
                (-1, (ValueError, "must be finite and non-negative")),
                (np.nan, (ValueError, "must be finite and non-negative")),
                (np.inf, (ValueError, "must be finite and non-negative")),
            ]
        ],
    ],
)
def test_xrd_pattern_raises(
    patterns: Any, err_type: type[Exception], match: str
) -> None:
    """Reject empty, unsupported and invalid-intensity inputs."""
    with pytest.raises(err_type, match=match):
        pmv.xrd_plot(patterns)


def test_xrd_pattern_zero_and_positive() -> None:
    """An all-zero pattern is fine when another pattern has a positive intensity."""
    widget = pmv.xrd_plot({"zero": single_peak_pattern(0), "pos": MOCK_XRD_PATTERN})
    assert widget.patterns["zero"]["y"] == [0]


def test_xrd_pattern_wavelength() -> None:
    """Default wavelength is Cu K-alpha and peaks shift per Bragg's law."""
    first_peaks = {
        wavelength: pmv.xrd_plot(BI2_ZR2_O7_STRUCT, wavelength=wavelength).patterns[
            "x"
        ][0]
        for wavelength in (cu_k_alpha_wavelength, 0.7093)
    }
    default_peak = pmv.xrd_plot(BI2_ZR2_O7_STRUCT).patterns["x"][0]
    assert default_peak == first_peaks[cu_k_alpha_wavelength]
    # same lattice plane d = λ / (2 sin θ) for both wavelengths
    d_spacings = [
        wavelength / (2 * np.sin(np.radians(two_theta / 2)))
        for wavelength, two_theta in first_peaks.items()
    ]
    # observed rel. difference ~3e-16 (1.5 eps)
    assert d_spacings[0] == pytest.approx(d_spacings[1], rel=1e-12)


@pytest.mark.parametrize(
    "pattern",
    [
        {"x": [10.0, 20.0], "y": [50.0, 100.0]},
        {"two_theta": [10.0, 20.0], "intensities": [50.0, 100.0]},
    ],
    ids=["canonical", "ferrox"],
)
def test_xrd_plot_pattern_dict(pattern: dict[str, list[float]]) -> None:
    """A bare pattern dict (canonical or Ferrox keys) is one unlabeled pattern."""
    widget = pmv.xrd_plot(pattern)
    assert (widget.patterns["x"], widget.patterns["y"]) == ([10, 20], [50, 100])


def test_xrd_pattern_labels() -> None:
    """Labels named like pattern keys stay labels, labels colliding as strings raise
    instead of dropping a pattern.
    """
    widget = pmv.xrd_plot({"x": MOCK_XRD_PATTERN, "y": BI2_ZR2_O7_STRUCT})
    assert list(widget.patterns) == ["x", "y"]
    assert widget.patterns["x"]["y"] == MOCK_XRD_PATTERN.intensities.tolist()

    with pytest.raises(ValueError, match="Labels must be unique as strings"):
        pmv.xrd_plot({1: MOCK_XRD_PATTERN, "1": MOCK_XRD_PATTERN})
