from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import kemio
import numpy as np
import pandas as pd
import pytest
from kemio.io import read_structure

import pymatviz as pmv
from pymatviz.utils.testing import TEST_FILES
from pymatviz.xrd import cu_k_alpha_wavelength


if TYPE_CHECKING:
    from typing import Any

MOCK_HKLS = [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 0, 0), (2, 1, 0)]
MOCK_XRD_PATTERN = kemio.XrdPattern(
    [10, 20, 30, 40, 50],
    [100, 50, 75, 25, 60],
    [2.5, 2.0, 1.8, 1.5, 1.3],
    [[(list(hkl), 1)] for hkl in MOCK_HKLS],
)

BI2_ZR2_O7_STRUCT = read_structure(
    f"{TEST_FILES}/xrd/Bi2Zr2O7-Fm3m-experimental-sqs.cif"
)
(BI2_ZR2_O7_XRD,) = kemio.xrd_patterns([BI2_ZR2_O7_STRUCT])


def single_peak_pattern(intensity: float) -> kemio.XrdPattern:
    """One-peak XrdPattern with the given intensity."""
    return kemio.XrdPattern([10], [intensity], [1], [[([1, 0, 0], 1)]])


@pytest.mark.parametrize(
    ("input_data", "expected"),
    [
        (MOCK_XRD_PATTERN, MOCK_XRD_PATTERN),
        (BI2_ZR2_O7_XRD, BI2_ZR2_O7_XRD),
        (BI2_ZR2_O7_STRUCT, BI2_ZR2_O7_XRD),
    ],
)
def test_xrd_pattern_single(input_data: Any, expected: kemio.XrdPattern) -> None:
    """A single pattern or structure is passed as one unlabeled pattern."""
    widget = pmv.xrd_plot(input_data, style="height: 300px")
    assert isinstance(widget, pmv.widgets.XrdWidget)
    assert widget.style == "height: 300px"
    for key, values in (("x", expected.two_theta), ("y", expected.intensities)):
        np.testing.assert_array_equal(widget.patterns[key], values)
    assert len(widget.patterns["hkls"]) == len(expected.hkls or [])


def test_xrd_pattern_mapping() -> None:
    """Mappings keep label order, compute structures and pass patterns (objects or
    dicts) through. An all-zero pattern is fine when another has positive intensity.
    """
    pattern_dict = {"x": [10, 20], "y": [100, 50], "hkls": [[{"hkl": (1, 0, 0)}]] * 2}
    patterns = OrderedDict(
        [
            ("Structure", BI2_ZR2_O7_STRUCT),
            ("Pattern", MOCK_XRD_PATTERN),
            ("Dict", pattern_dict),
            ("Zero", single_peak_pattern(0)),
        ]
    )
    widget = pmv.xrd_plot(patterns)
    assert list(widget.patterns) == ["Structure", "Pattern", "Dict", "Zero"]
    assert widget.patterns["Dict"]["y"] == [100, 50]
    assert widget.patterns["Zero"]["y"] == [0]
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
        (kemio.XrdPattern([], []), ValueError, "No intensities found"),
        ({}, ValueError, "patterns must not be empty"),
        ("invalid", TypeError, "'XRD Pattern': expected a structure"),
        ({"bad": "invalid"}, TypeError, "'bad': expected a structure"),
        *[  # pattern dicts (kemio's XrdPattern rejects these values itself)
            ({"x": [10], "y": [val]}, *err)
            for val, err in [
                (0, (ValueError, "At least one diffraction intensity")),
                (-1, (ValueError, "must be finite and non-negative")),
                (np.nan, (ValueError, "must be finite and non-negative")),
                (np.inf, (ValueError, "must be finite and non-negative")),
            ]
        ],
        (  # labels colliding as strings raise instead of dropping a pattern
            {1: MOCK_XRD_PATTERN, "1": MOCK_XRD_PATTERN},
            ValueError,
            "Labels must be unique as strings",
        ),
    ],
)
def test_xrd_pattern_raises(
    patterns: Any, err_type: type[Exception], match: str
) -> None:
    """Reject empty, unsupported, invalid-intensity and colliding-label inputs."""
    with pytest.raises(err_type, match=match):
        pmv.xrd_plot(patterns)


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


def test_xrd_plot_pattern_dict() -> None:
    """A bare x/y pattern dict is one unlabeled pattern, also when holding pandas
    Series, which serialize as arrays.
    """
    pattern = {"x": [10.0, 20.0], "y": [50.0, 100.0]}
    widget = pmv.xrd_plot(pattern)
    assert (widget.patterns["x"], widget.patterns["y"]) == ([10, 20], [50, 100])
    series_widget = pmv.xrd_plot({key: pd.Series(val) for key, val in pattern.items()})
    np.testing.assert_array_equal(series_widget.patterns["y"], [50, 100])
    assert '"patterns"' in series_widget.to_json()


def test_xrd_pattern_labels() -> None:
    """Labels named like pattern keys stay labels."""
    widget = pmv.xrd_plot({"x": MOCK_XRD_PATTERN, "y": BI2_ZR2_O7_STRUCT})
    assert list(widget.patterns) == ["x", "y"]
    assert widget.patterns["x"]["y"] == MOCK_XRD_PATTERN.intensities.tolist()
