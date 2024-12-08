from __future__ import annotations

import copy
import json
import re
from glob import glob
from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go
import pytest
from monty.io import zopen
from monty.json import MontyDecoder, MSONable
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine as PhononBands
from pymatgen.phonon.dos import PhononDos

import pymatviz as pmv
from pymatviz.typing import SET_INTERSECTION, SET_STRICT, SET_UNION
from pymatviz.utils.testing import TEST_FILES


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    from phonopy import Phonopy

    from pymatviz.typing import SetMode


BandsDoses = dict[str, dict[str, PhononBands | PhononDos]]
bs_key, dos_key = "phonon_bandstructure", "phonon_dos"
# enable loading PhononDBDocParsed with @module set to uninstalled ffonons.dbs.phonondb
# by changing to identical dataclass in pymatviz.phonons module
MSONable.REDIRECT["ffonons.dbs.phonondb"] = {
    "PhononDBDocParsed": {"@class": "PhononDBDoc", "@module": "pymatviz.phonons"}
}
MSONable.REDIRECT["atomate2.common.schemas.phonons"] = {
    "PhononBSDOSDoc": {"@class": "PhononDBDoc", "@module": "pymatviz.phonons"}
}


@pytest.fixture
def phonon_bands_doses_mp_2758() -> BandsDoses:
    with zopen(f"{TEST_FILES}/phonons/mp-2758-Sr4Se4-pbe.json.lzma") as file:
        dft_doc = json.loads(file.read(), cls=MontyDecoder)

    with zopen(f"{TEST_FILES}/phonons/mp-2758-Sr4Se4-mace-y7uhwpje.json.lzma") as file:
        ml_doc = json.loads(file.read(), cls=MontyDecoder)

    bands = {"DFT": getattr(dft_doc, bs_key), "MACE": getattr(ml_doc, bs_key)}
    doses = {"DFT": getattr(dft_doc, dos_key), "MACE": getattr(ml_doc, dos_key)}
    return {"bands": bands, "doses": doses}


@pytest.fixture
def phonon_bands_doses_mp_2667() -> BandsDoses:
    # with zopen(f"{TEST_FILES}/phonons/mp-2691-Cd4Se4-pbe.json.lzma") as file:
    with zopen(f"{TEST_FILES}/phonons/mp-2667-Cs1Au1-pbe.json.lzma") as file:
        return json.loads(file.read(), cls=MontyDecoder)


@pytest.fixture
def phonon_doses() -> dict[str, PhononDos]:
    paths = glob(f"{TEST_FILES}/phonons/mp-*-pbe.json.lzma")
    assert len(paths) >= 2
    return {
        path.split("/")[-1].split("-pbe")[0]: getattr(
            json.loads(zopen(path).read(), cls=MontyDecoder), dos_key
        )
        for path in paths
    }


@pytest.mark.parametrize(
    ("branches", "path_mode", "line_kwargs", "expected_x_labels"),
    [
        # test original single dict behavior
        (
            ["GAMMA-X", "X-U"],
            "union",
            dict(width=2),
            tuple("ΓLXΓWXU"),
        ),
        # test empty tuple branches with intersection mode
        ((), "intersection", None, tuple("ΓLXΓWXU")),
        # test separate acoustic/optical styling
        (
            ["GAMMA-X"],
            "union",
            {
                "acoustic": dict(width=2.5, dash="solid", name="Acoustic modes"),
                "optical": dict(width=1, dash="dash", name="Optical modes"),
            },
            tuple("ΓLXΓWXU"),
        ),
        # test callable line_kwargs
        (
            (),
            "union",
            lambda _freqs, idx: dict(dash="solid" if idx < 3 else "dash"),
            tuple("ΓLXΓWXU"),
        ),
    ],
)
def test_phonon_bands(
    phonon_bands_doses_mp_2758: BandsDoses,
    branches: tuple[str, str],
    path_mode: SetMode,
    line_kwargs: dict[str, Any] | Callable[[np.ndarray, int], dict[str, Any]] | None,
    expected_x_labels: tuple[str, ...],
) -> None:
    # test single band structure
    fig = pmv.phonon_bands(
        phonon_bands_doses_mp_2758["bands"]["DFT"],
        path_mode=path_mode,
        branches=branches,
        line_kwargs=line_kwargs,
    )
    assert isinstance(fig, go.Figure)
    assert fig.layout.xaxis.title.text == "Wave Vector"
    assert fig.layout.yaxis.title.text == "Frequency (THz)"
    assert fig.layout.font.size == 16

    actual_x_labels = fig.layout.xaxis.ticktext
    assert (
        actual_x_labels == expected_x_labels
    ), f"{actual_x_labels=}, {expected_x_labels=}"
    assert fig.layout.xaxis.range is None
    assert fig.layout.yaxis.range == pytest.approx((0, 5.36385427095))

    # test line styling
    if isinstance(line_kwargs, dict) and "acoustic" in line_kwargs:
        # check that acoustic and optical modes have different styles
        trace_names = {trace.name for trace in fig.data}
        assert trace_names == {"", "Acoustic modes", "Optical modes"}

        acoustic_traces = [t for t in fig.data if t.name == "Acoustic modes"]
        optical_traces = [t for t in fig.data if t.name == "Optical modes"]

        assert all(t.line.width == 2.5 for t in acoustic_traces)
        assert all(t.line.dash == "solid" for t in acoustic_traces)
        assert all(t.line.width == 1 for t in optical_traces)
        assert all(t.line.dash == "dash" for t in optical_traces)
    elif callable(line_kwargs):
        # check that line width increases with band index
        traces_by_width = sorted(fig.data, key=lambda t: t.line.width)
        assert traces_by_width[0].line.width < traces_by_width[-1].line.width

        # check acoustic/optical line style separation
        acoustic_traces = [t for t in fig.data if t.line.dash == "solid"]
        optical_traces = [t for t in fig.data if t.line.dash == "dash"]
        assert len(acoustic_traces) == 18  # 6 segments for the first 3 bands
        assert len(optical_traces) > 0  # should have some optical bands

    # test dict of band structures
    fig = pmv.phonon_bands(
        phonon_bands_doses_mp_2758["bands"],
        path_mode=path_mode,
        branches=branches,
        line_kwargs=line_kwargs,
    )
    assert isinstance(fig, go.Figure)
    assert {trace.name for trace in fig.data} == {"DFT", "MACE"}
    assert fig.layout.xaxis.ticktext == expected_x_labels


def test_phonon_bands_raises(
    phonon_bands_doses_mp_2758: BandsDoses, capsys: pytest.CaptureFixture
) -> None:
    with pytest.raises(
        TypeError, match=f"Only {PhononBands.__name__} or dict supported, got str"
    ):
        pmv.phonon_bands("invalid input")

    # issues warning when requesting some available and some unavailable branches
    pmv.phonon_bands(
        phonon_bands_doses_mp_2758["bands"]["DFT"], branches=("X-U", "foo-bar")
    )
    stdout, stderr = capsys.readouterr()
    assert stdout == ""
    assert "Warning missing_branches={'foo-bar'}, available branches:" in stderr

    with pytest.raises(ValueError, match="Invalid path_mode='invalid'"):
        pmv.phonon_bands(
            phonon_bands_doses_mp_2758["bands"]["DFT"],
            path_mode="invalid",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="Empty band structure dict"):
        pmv.phonon_bands({})


@pytest.mark.parametrize(
    ("sym_point", "expected"),
    [("Γ", "Γ"), ("Γ|DELTA", "Γ|Δ"), ("GAMMA", "Γ"), ("S_0|SIGMA", "S<sub>0</sub>|Σ")],
)
def test_pretty_sym_point(sym_point: str, expected: str) -> None:
    assert pmv.phonons.pretty_sym_point(sym_point) == expected


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
    with pytest.raises(
        TypeError, match=f"Only {PhononDos.__name__} or dict supported, got str"
    ):
        pmv.phonon_dos("invalid input")

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
        phonopy_nacl.total_dos, stack=False, sigma=0.1, normalize="max", units="THz"
    )
    assert isinstance(fig, go.Figure)
    assert fig.layout.xaxis.title.text == "Frequency (THz)"
    assert fig.layout.yaxis.title.text == "Density of States"

    # Test dictionary of TotalDos objects
    dos_dict = {"DFT": phonopy_nacl.total_dos, "ML": phonopy_nacl.total_dos}
    fig = pmv.phonon_dos(dos_dict, stack=True, sigma=0.1, normalize="max", units="THz")
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
