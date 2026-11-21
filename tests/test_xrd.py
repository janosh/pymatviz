from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np
import pytest
from pymatgen.analysis.diffraction.xrd import DiffractionPattern, XRDCalculator
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

import pymatviz as pmv
from pymatviz.utils.testing import TEST_FILES
from pymatviz.xrd import cu_k_alpha_wavelength


if TYPE_CHECKING:
    from typing import Any

MOCK_DIFFRACTION_PATTERN = DiffractionPattern(
    x=[10, 20, 30, 40, 50],
    y=[100, 50, 75, 25, 60],
    hkls=[
        [{"hkl": (1, 0, 0)}],
        [{"hkl": (1, 1, 0)}],
        [{"hkl": (1, 1, 1)}],
        [{"hkl": (2, 0, 0)}],
        [{"hkl": (2, 1, 0)}],
    ],
    d_hkls=[2.5, 2.0, 1.8, 1.5, 1.3],
)

BI2_ZR2_O7_STRUCT = Structure.from_file(
    f"{TEST_FILES}/xrd/Bi2Zr2O7-Fm3m-experimental-sqs.cif"
)
BI2_ZR2_O7_XRD = XRDCalculator().get_pattern(BI2_ZR2_O7_STRUCT)


@pytest.mark.parametrize(
    ("input_data", "expected"),
    [
        (MOCK_DIFFRACTION_PATTERN, MOCK_DIFFRACTION_PATTERN),
        (BI2_ZR2_O7_XRD, BI2_ZR2_O7_XRD),
        (BI2_ZR2_O7_STRUCT, BI2_ZR2_O7_XRD),
        (AseAtomsAdaptor.get_atoms(BI2_ZR2_O7_STRUCT), BI2_ZR2_O7_XRD),
    ],
)
def test_xrd_pattern_single(input_data: Any, expected: DiffractionPattern) -> None:
    """A single pattern or structure is passed as one unlabeled pattern."""
    widget = pmv.xrd_plot(input_data, style="height: 300px")
    assert isinstance(widget, pmv.widgets.XrdWidget)
    assert widget.style == "height: 300px"
    # ASE Atoms round-trip perturbs coordinates by ~1.5e-14 relative (~70 eps)
    for key in ("x", "y"):
        np.testing.assert_allclose(
            widget.patterns[key], getattr(expected, key), rtol=1e-12, atol=0
        )
    assert len(widget.patterns["hkls"]) == len(expected.hkls)


def test_xrd_pattern_mapping() -> None:
    """Mappings keep label order, compute structures and pass patterns through."""
    patterns = OrderedDict(
        [("Structure", BI2_ZR2_O7_STRUCT), ("Pattern", MOCK_DIFFRACTION_PATTERN)]
    )
    widget = pmv.xrd_plot(patterns)
    assert list(widget.patterns) == ["Structure", "Pattern"]
    np.testing.assert_array_equal(widget.patterns["Structure"]["x"], BI2_ZR2_O7_XRD.x)
    assert widget.patterns["Pattern"]["d_hkls"] == MOCK_DIFFRACTION_PATTERN.d_hkls


@pytest.mark.parametrize(
    ("patterns", "err_type", "match"),
    [
        (DiffractionPattern([], [], [], []), ValueError, "No intensities found"),
        ({}, ValueError, "patterns must not be empty"),
        ("invalid", TypeError, "'XRD Pattern': expected a pymatgen Structure"),
        ({"bad": "invalid"}, TypeError, "'bad': expected a pymatgen Structure"),
        *[
            (DiffractionPattern([10], [val], [[{"hkl": (1, 0, 0)}]], [1]), *err)
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
    zero_pattern = DiffractionPattern([10], [0], [[{"hkl": (1, 0, 0)}]], [1])
    widget = pmv.xrd_plot({"zero": zero_pattern, "pos": MOCK_DIFFRACTION_PATTERN})
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


def test_xrd_pattern_labels() -> None:
    """Labels named like pattern keys stay labels, IStructures are computed, labels
    colliding as strings raise instead of dropping a pattern.
    """
    from pymatgen.core import IStructure

    widget = pmv.xrd_plot({"x": MOCK_DIFFRACTION_PATTERN, "y": BI2_ZR2_O7_STRUCT})
    assert list(widget.patterns) == ["x", "y"]
    assert widget.patterns["x"]["y"] == MOCK_DIFFRACTION_PATTERN.y.tolist()

    widget = pmv.xrd_plot(IStructure.from_sites(BI2_ZR2_O7_STRUCT))
    np.testing.assert_array_equal(widget.patterns["x"], BI2_ZR2_O7_XRD.x)

    with pytest.raises(ValueError, match="Labels must be unique as strings"):
        pmv.xrd_plot({1: MOCK_DIFFRACTION_PATTERN, "1": MOCK_DIFFRACTION_PATTERN})
