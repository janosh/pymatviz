from __future__ import annotations

import copy
import re
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
import pytest
from pymatgen.phonon.dos import PhononDos

import pymatviz as pmv
from pymatviz.typing import SET_INTERSECTION, SET_STRICT, SET_UNION


if TYPE_CHECKING:
    from typing import Literal

    from phonopy import Phonopy

    from pymatviz.typing import SetMode

    from .conftest import BandsDoses


@pytest.mark.parametrize(
    ("units", "stack", "sigma", "normalize", "last_peak_anno"),
    [
        ("eV", False, 0.01, "max", "{key}={last_peak:.1f}"),
        ("meV", False, 0.05, "sum", "{key}={last_peak:.4} ({units})"),
        ("cm-1", True, 0.1, "integral", None),
        ("THz", True, 0.1, None, ""),
    ],
)
def test_phonon_dos(
    phonon_bands_doses_mp_2758: BandsDoses,
    units: Literal["eV", "meV", "cm-1", "THz"],
    stack: bool,
    sigma: float,
    normalize: Literal["max", "sum", "integral"] | None,
    last_peak_anno: str | None,
) -> None:
    fig = pmv.phonon_dos(
        phonon_bands_doses_mp_2758["doses"],  # test dict
        stack=stack,
        sigma=sigma,
        normalize=normalize,
        units=units,
        last_peak_anno=last_peak_anno,
    )

    assert isinstance(fig, go.Figure)
    assert fig.layout.xaxis.title.text == f"Frequency ({units})"
    assert fig.layout.yaxis.title.text == "Density of States"
    assert fig.layout.font.size == 16

    fig = pmv.phonon_dos(
        phonon_bands_doses_mp_2758["doses"]["DFT"],  # test single
        stack=stack,
        sigma=sigma,
        normalize=normalize,
        units=units,
        last_peak_anno=last_peak_anno,
    )
    assert isinstance(fig, go.Figure)


def test_phonon_dos_raises(phonon_bands_doses_mp_2758: BandsDoses) -> None:
    """Test phonon_dos raises appropriate errors."""
    with pytest.raises(
        TypeError, match=f"Only {PhononDos.__name__} or dict supported, got str"
    ):
        pmv.phonon_dos("invalid input")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Empty DOS dict"):
        pmv.phonon_dos({})

    expected_msg = (
        "Invalid unit='invalid', must be one of ['THz', 'eV', 'meV', 'Ha', 'cm-1']"
    )
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        pmv.phonon_dos(phonon_bands_doses_mp_2758["doses"], units="invalid")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("units", "stack", "sigma", "normalize", "last_peak_anno"),
    [
        ("eV", False, 0.01, "max", "{key}={last_peak:.1f}"),
        ("meV", False, 0.05, "sum", "{key}={last_peak:.4} ({units})"),
        ("cm-1", True, 0.1, "integral", None),
        ("THz", True, 0.1, None, ""),
    ],
)
def test_phonon_bands_and_dos(
    phonon_bands_doses_mp_2758: BandsDoses,
    phonon_doses: dict[str, PhononDos],  # different keys
    units: Literal["eV", "meV", "cm-1", "THz"],
    stack: bool,
    sigma: float,
    normalize: Literal["max", "sum", "integral"] | None,
    last_peak_anno: str | None,
) -> None:
    bands, doses = (
        phonon_bands_doses_mp_2758["bands"],
        phonon_bands_doses_mp_2758["doses"],
    )
    dos_kwargs = dict(
        stack=stack,
        sigma=sigma,
        normalize=normalize,
        units=units,
        last_peak_anno=last_peak_anno,
    )
    # test dicts
    fig = pmv.phonon_bands_and_dos(bands, doses, dos_kwargs=dos_kwargs)

    assert isinstance(fig, go.Figure)
    assert fig.layout.xaxis.title.text == "Wave Vector"
    assert fig.layout.yaxis.title.text == f"Frequency ({units})"
    assert fig.layout.xaxis2.title.text == "DOS"
    assert fig.layout.font.size == 16
    # check legend labels
    assert {trace.name for trace in fig.data} == {"DFT", "MACE"}

    fig = pmv.phonon_bands_and_dos(bands["DFT"], doses["DFT"])
    assert isinstance(fig, go.Figure)

    band_keys, dos_keys = set(bands), set(phonon_doses)
    with pytest.raises(
        ValueError, match=f"{band_keys=} and {dos_keys=} must be identical"
    ):
        pmv.phonon_bands_and_dos(bands, phonon_doses)


@pytest.mark.parametrize(
    ("path_mode", "expected_segments"),
    [
        (SET_STRICT, {}),
        (
            SET_INTERSECTION,
            {("Γ", "L"), ("L", "X"), ("X", "Γ"), ("Γ", "W"), ("W", "U")},
        ),
        (
            SET_UNION,
            {("X", "U"), ("W", "X"), ("Γ", "W"), ("Γ", "L"), ("L", "X"), ("X", "Γ")},
        ),
    ],
)
def test_phonon_bands_path_modes(
    phonon_bands_doses_mp_2758: BandsDoses,
    path_mode: SetMode,
    expected_segments: set[tuple[str, str]],
) -> None:
    """Test different path_mode options for phonon band structure plotting."""
    bands = phonon_bands_doses_mp_2758["bands"]

    # Modify one band structure to have a different path
    modified_bands = bands.copy()
    modified_bands["MACE"] = copy.deepcopy(modified_bands["MACE"])
    # Remove last branch to create a mismatch
    modified_bands["MACE"].branches = modified_bands["MACE"].branches[:-1]

    if path_mode == SET_STRICT:
        with pytest.raises(
            ValueError, match="Band structures have different q-point paths"
        ):
            pmv.phonon_bands(modified_bands, path_mode=path_mode)
        return
    if path_mode == SET_INTERSECTION:
        bands = modified_bands

    fig = pmv.phonon_bands(modified_bands, path_mode=path_mode)

    # Extract plotted segments from x-axis labels
    plotted_segments = set()
    labels = fig.layout.xaxis.ticktext
    for idx in range(len(labels) - 1):
        if not (labels[idx] and labels[idx + 1]):  # Skip empty labels
            continue
        # Convert from pretty format back to raw
        start = labels[idx]
        end = labels[idx + 1]
        if "|" in end:
            end = end.split("|")[0]  # Take first part of combined labels
        plotted_segments.add((start, end))

    assert plotted_segments == expected_segments


def test_phonon_bands_path_mode_raises(phonon_bands_doses_mp_2758: BandsDoses) -> None:
    """Test error cases for path_mode parameter."""
    with pytest.raises(ValueError, match="Invalid path_mode='invalid'"):
        pmv.phonon_bands(
            phonon_bands_doses_mp_2758["bands"]["DFT"],
            path_mode="invalid",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("path_mode", [SET_STRICT, SET_INTERSECTION, SET_UNION])
def test_phonon_bands_and_dos_path_modes(
    phonon_bands_doses_mp_2758: BandsDoses,
    path_mode: SetMode,
) -> None:
    """Test path_mode is correctly passed through to phonon_bands."""
    bands = phonon_bands_doses_mp_2758["bands"]
    doses = phonon_bands_doses_mp_2758["doses"]

    if path_mode == SET_STRICT:
        # Modify one band structure to have a different path
        modified_bands = bands.copy()
        modified_bands["MACE"] = copy.deepcopy(modified_bands["MACE"])
        modified_bands["MACE"].branches = modified_bands["MACE"].branches[:-1]

        with pytest.raises(
            ValueError, match="Band structures have different q-point paths"
        ):
            pmv.phonon_bands_and_dos(modified_bands, doses, path_mode=path_mode)
        return

    fig = pmv.phonon_bands_and_dos(bands, doses, path_mode=path_mode)
    assert isinstance(fig, go.Figure)

    # Check that band structure subplot exists
    assert fig.layout.xaxis.title.text == "Wave Vector"
    # Check that DOS subplot exists
    assert fig.layout.xaxis2.title.text == "DOS"

    # Verify the number of traces matches expectations
    n_traces = len(fig.data)
    assert n_traces > 0  # Should have at least some traces

    # Check that the legend contains each model name (only once)
    legend_entries = {trace.name for trace in fig.data}
    assert legend_entries == {"DFT", "MACE"}


def test_phonon_bands_and_dos_path_mode_raises(
    phonon_bands_doses_mp_2758: BandsDoses,
) -> None:
    """Test error handling for invalid path_mode in phonon_bands_and_dos."""
    with pytest.raises(ValueError, match="Invalid path_mode='invalid'"):
        pmv.phonon_bands_and_dos(
            phonon_bands_doses_mp_2758["bands"],
            phonon_bands_doses_mp_2758["doses"],
            path_mode="invalid",  # type: ignore[arg-type]
        )


@pytest.mark.importorskip("phonopy")
def test_phonon_dos_with_phonopy(phonopy_nacl: Phonopy) -> None:
    """Test that phonon_dos works with phonopy TotalDos objects."""
    phonopy_nacl.run_mesh([5, 5, 5])
    phonopy_nacl.run_total_dos(freq_pitch=1, use_tetrahedron_method=False)

    # Test single TotalDos
    fig = pmv.phonon_dos(
        phonopy_nacl.total_dos, stack=False, sigma=0.1, normalize="max", units="THz"  # type: ignore[arg-type]
    )
    assert isinstance(fig, go.Figure)
    assert fig.layout.xaxis.title.text == "Frequency (THz)"
    assert fig.layout.yaxis.title.text == "Density of States"

    # Test dictionary of TotalDos objects
    dos_dict = {"DFT": phonopy_nacl.total_dos, "ML": phonopy_nacl.total_dos}
    fig = pmv.phonon_dos(dos_dict, stack=True, sigma=0.1, normalize="max", units="THz")  # type: ignore[arg-type]
    assert isinstance(fig, go.Figure)
    assert {trace.name for trace in fig.data} == {"DFT", "ML"}


@pytest.mark.importorskip("phonopy")
def test_phonon_bands_with_phonopy(phonopy_nacl: Phonopy) -> None:
    """Test plotting phonon bands from phonopy band structure."""
    bands = {  # Define q-points path
        "L -> Γ": np.linspace([1 / 3, 1 / 3, 0], [0, 0, 0], 51),
        "Γ -> X": np.linspace([0, 0, 0], [1 / 2, 0, 0], 51),
        "X -> K": np.linspace([1 / 2, 0, 0], [1 / 2, 1 / 2, 0], 51),
        "K -> Γ": np.linspace([1 / 2, 1 / 2, 0], [0, 0, 0], 51),
        "Γ -> W": np.linspace([0, 0, 0], [1 / 3, 1 / 3, 1 / 2], 51),
    }

    # Run band structure calculation
    phonopy_nacl.run_band_structure(paths=bands.values())

    # Test single band structure
    fig = pmv.phonon_bands(phonopy_nacl.band_structure)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 6

    # Test multiple band structures in dict
    bands_dict = {
        "NaCl (phonopy)": phonopy_nacl.band_structure,
        "DFT": phonopy_nacl.band_structure,
    }
    fig = pmv.phonon_bands(bands_dict)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 6 * len(bands_dict)
