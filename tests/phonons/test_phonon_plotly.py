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
    from pymatgen.phonon.dos import CompletePhononDos

    from pymatviz.typing import SetMode
    from tests.conftest import BandsDoses


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
    with pytest.raises(TypeError, match="supported, got str"):
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
        phonopy_nacl.total_dos,  # type: ignore[arg-type]
        stack=False,
        sigma=0.1,
        normalize="max",
        units="THz",
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
    phonopy_nacl.run_band_structure(paths=list(bands.values()))

    # Test single band structure
    fig = pmv.phonon_bands(phonopy_nacl.band_structure)  # type: ignore[arg-type]
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 6

    # Test multiple band structures in dict
    bands_dict = {
        "NaCl (phonopy)": phonopy_nacl.band_structure,
        "DFT": phonopy_nacl.band_structure,
    }
    fig = pmv.phonon_bands(bands_dict)  # type: ignore[arg-type]
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 6 * len(bands_dict)


# === Element/site-projected phonon DOS tests ===


@pytest.mark.parametrize(
    ("project", "show_total", "expected_names"),
    [
        ("element", True, {"Na", "Cl", "Total"}),
        ("element", False, {"Na", "Cl"}),
        ("site", True, {"Na0", "Cl1", "Total"}),
        ("site", False, {"Na0", "Cl1"}),
    ],
)
def test_phonon_dos_projection(
    complete_phonon_dos: CompletePhononDos,
    project: Literal["element", "site"],
    show_total: bool,
    expected_names: set[str],
) -> None:
    """Test element/site projection with show_total on/off."""
    fig = pmv.phonon_dos(complete_phonon_dos, project=project, show_total=show_total)
    assert {trace.name for trace in fig.data} == expected_names

    if project == "element":
        expected_dos_by_name = {
            str(element): dos
            for element, dos in complete_phonon_dos.get_element_dos().items()
        }
    else:
        expected_dos_by_name = {
            f"{site.specie}{site_idx}": complete_phonon_dos.get_site_dos(site)
            for site_idx, site in enumerate(complete_phonon_dos.structure)
        }
    for trace in fig.data:
        if trace.name == "Total":
            continue
        expected_dos = expected_dos_by_name[trace.name]
        assert np.allclose(trace.x, expected_dos.frequencies)
        assert np.allclose(trace.y, expected_dos.densities)

    if show_total:
        total_trace = next(tr for tr in fig.data if tr.name == "Total")
        assert np.allclose(total_trace.x, complete_phonon_dos.frequencies)
        assert np.allclose(total_trace.y, complete_phonon_dos.densities)
        assert total_trace.line.dash == "dash"
        assert total_trace.line.color == "gray"


@pytest.mark.parametrize(
    ("project", "expected_names"),
    [
        (
            "element",
            {"DFT - Na", "DFT - Cl", "ML - Na", "ML - Cl", "DFT - Total", "ML - Total"},
        ),
        (
            "site",
            {
                "DFT - Na0",
                "DFT - Cl1",
                "ML - Na0",
                "ML - Cl1",
                "DFT - Total",
                "ML - Total",
            },
        ),
    ],
)
def test_phonon_dos_projection_dict(
    complete_phonon_dos: CompletePhononDos,
    project: Literal["element", "site"],
    expected_names: set[str],
) -> None:
    """Test projection with dict of multiple CompletePhononDos."""
    dos_dict = {"DFT": complete_phonon_dos, "ML": complete_phonon_dos}
    fig = pmv.phonon_dos(dos_dict, project=project)
    assert {trace.name for trace in fig.data} == expected_names


@pytest.mark.parametrize(
    "model_labels",
    [
        ("DFT", "ML"),
        ("DFT - baseline", "DFT - variant"),
    ],
)
def test_phonon_dos_projection_dict_stack_resets_by_model(
    complete_phonon_dos: CompletePhononDos,
    model_labels: tuple[str, str],
) -> None:
    """Test stacked projected DOS accumulates separately for each model label."""
    dos_dict = {label: copy.deepcopy(complete_phonon_dos) for label in model_labels}
    fig_unstacked = pmv.phonon_dos(
        dos_dict, project="element", show_total=False, stack=False
    )
    fig_stacked = pmv.phonon_dos(
        dos_dict, project="element", show_total=False, stack=True
    )
    unstacked_by_name = {
        trace.name: np.asarray(trace.y) for trace in fig_unstacked.data
    }
    stacked_by_name = {trace.name: np.asarray(trace.y) for trace in fig_stacked.data}
    for model_label in model_labels:
        trace_name = next(
            trace.name
            for trace in fig_stacked.data
            if trace.name.startswith(f"{model_label} - ")
        )
        assert np.allclose(stacked_by_name[trace_name], unstacked_by_name[trace_name])


def test_phonon_dos_projection_stack_keeps_total_unstacked(
    complete_phonon_dos: CompletePhononDos,
) -> None:
    """Test total overlay trace values are unchanged by stack=True."""
    fig_unstacked = pmv.phonon_dos(complete_phonon_dos, project="element", stack=False)
    fig_stacked = pmv.phonon_dos(complete_phonon_dos, project="element", stack=True)
    unstacked_total = next(
        trace for trace in fig_unstacked.data if trace.name == "Total"
    )
    stacked_total = next(trace for trace in fig_stacked.data if trace.name == "Total")
    assert np.allclose(stacked_total.y, unstacked_total.y)


def test_phonon_dos_projection_raises(
    complete_phonon_dos: CompletePhononDos,
    phonon_bands_doses_mp_2758: BandsDoses,
) -> None:
    """Test error cases for project parameter."""
    plain_dos = phonon_bands_doses_mp_2758["doses"]["DFT"]
    # single plain PhononDos
    with pytest.raises(TypeError, match="project='element' requires CompletePhononDos"):
        pmv.phonon_dos(plain_dos, project="element")
    # dict containing plain PhononDos
    mixed = {"good": complete_phonon_dos, "bad": plain_dos}
    with pytest.raises(TypeError, match="project='element' requires CompletePhononDos"):
        pmv.phonon_dos(mixed, project="element")
    # invalid project value
    with pytest.raises(ValueError, match="Invalid project='invalid'"):
        pmv.phonon_dos(complete_phonon_dos, project="invalid")  # type: ignore[arg-type]


def test_phonon_dos_projection_unit_conversion(
    complete_phonon_dos: CompletePhononDos,
) -> None:
    """Test that unit conversion is applied to projected traces."""
    fig_thz = pmv.phonon_dos(complete_phonon_dos, project="element", show_total=False)
    fig_ev = pmv.phonon_dos(
        complete_phonon_dos, project="element", show_total=False, units="eV"
    )
    assert max(fig_ev.data[0].x) < max(fig_thz.data[0].x)
    assert fig_ev.layout.xaxis.title.text == "Frequency (eV)"


def test_phonon_dos_projection_normalization_values(
    complete_phonon_dos: CompletePhononDos,
) -> None:
    """Test that max normalization produces peak=1.0 for each trace."""
    fig = pmv.phonon_dos(
        complete_phonon_dos, project="element", normalize="max", show_total=False
    )
    for trace in fig.data:
        assert max(trace.y) == pytest.approx(1.0)


def test_phonon_dos_projection_sigma_smearing(
    complete_phonon_dos: CompletePhononDos,
) -> None:
    """Test that sigma smearing lowers peak height in projected traces."""
    fig_raw = pmv.phonon_dos(
        complete_phonon_dos, project="element", sigma=0, show_total=False
    )
    fig_smooth = pmv.phonon_dos(
        complete_phonon_dos, project="element", sigma=0.5, show_total=False
    )
    assert max(fig_smooth.data[0].y) < max(fig_raw.data[0].y)


@pytest.mark.parametrize("normalize_mode", ["max", "sum", "integral"])
def test_phonon_dos_normalize_zero_density_raises(
    normalize_mode: Literal["max", "sum", "integral"],
) -> None:
    """Test normalization raises on all-zero DOS."""
    frequencies = np.linspace(0, 10, 100)
    zero_dos = PhononDos(frequencies, np.zeros_like(frequencies))
    with pytest.raises(ValueError, match=f"mode='{normalize_mode}'"):
        pmv.phonon_dos(zero_dos, normalize=normalize_mode)


def test_phonon_dos_integral_normalize_requires_two_frequency_points() -> None:
    """Test integral normalization raises for a single frequency point."""
    single_point_dos = PhononDos(np.array([1.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="need >=2 frequency points"):
        pmv.phonon_dos(single_point_dos, normalize="integral")


def test_phonon_dos_show_total_ignored_without_project(
    phonon_bands_doses_mp_2758: BandsDoses,
) -> None:
    """Test that show_total has no effect when project is None."""
    dos = phonon_bands_doses_mp_2758["doses"]["DFT"]
    fig_a = pmv.phonon_dos(dos)
    fig_b = pmv.phonon_dos(dos, show_total=False)
    assert [t.name for t in fig_a.data] == [t.name for t in fig_b.data]


def test_phonon_dos_complete_dos_without_projection(
    complete_phonon_dos: CompletePhononDos,
) -> None:
    """Test CompletePhononDos defaults to total DOS when project is None."""
    fig = pmv.phonon_dos(complete_phonon_dos)
    assert len(fig.data) == 1
    trace = fig.data[0]
    assert np.allclose(trace.x, complete_phonon_dos.frequencies)
    assert np.allclose(trace.y, complete_phonon_dos.densities)
