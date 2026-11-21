from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pytest
from pymatgen.phonon.dos import PhononDos

import pymatviz as pmv


if TYPE_CHECKING:
    from phonopy import Phonopy
    from pymatgen.phonon.dos import CompletePhononDos

    from tests.conftest import BandsDoses

GRID_012 = PhononDos([0, 1, 2], [1, 2, 1])

# skipif, not pytest.importorskip: the phonopy_nacl fixture imports phonopy at setup
needs_phonopy = pytest.mark.skipif(
    importlib.util.find_spec("phonopy") is None, reason="phonopy not installed"
)


@pytest.mark.parametrize("labeled", [False, True])
def test_phonon_bands(phonon_bands_doses_mp_2758: BandsDoses, labeled: bool) -> None:
    """Single band structures pass unlabeled, dicts keep their labels and bands."""
    bands = phonon_bands_doses_mp_2758["bands"]
    widget = pmv.phonon_bands(bands if labeled else bands["DFT"], fermi_level=None)
    assert isinstance(widget, pmv.BandStructureWidget)
    docs = widget.band_structs if labeled else {"DFT": widget.band_structs}
    assert set(docs) == ({"DFT", "MACE"} if labeled else {"DFT"})
    for label, doc in docs.items():
        np.testing.assert_array_equal(doc["bands"], bands[label].bands)

    widget = pmv.phonon_bands(bands["DFT"], reference_frequency=1.5, style="height:9em")
    assert (widget.reference_frequency, widget.style) == (1.5, "height:9em")


@pytest.mark.parametrize(
    ("band_structs", "err_type", "match"),
    [
        ("invalid", TypeError, "Only PhononBandStructureSymmLine, phonopy"),
        ({}, ValueError, "Empty band structure mapping"),
    ],
)
def test_phonon_bands_raises(
    band_structs: Any, err_type: type[Exception], match: str
) -> None:
    """Reject unsupported and empty band structure inputs."""
    with pytest.raises(err_type, match=match):
        pmv.phonon_bands(band_structs)


@pytest.mark.parametrize(
    ("stack", "sigma", "normalize"),
    [(False, 0, None), (True, 0.1, "max"), (False, 0.2, "integral"), (True, 0, "sum")],
)
def test_phonon_dos(
    phonon_doses: dict[str, PhononDos],
    stack: bool,
    sigma: float,
    normalize: Literal["max", "sum", "integral"] | None,
) -> None:
    """Raw DOS data goes to the widget, smearing and normalization as traits."""
    if stack:  # MatterViz stacks by index, so stacked DOS must share a grid
        freqs = next(iter(phonon_doses.values())).frequencies
        phonon_doses = {
            label: PhononDos(freqs, dos.densities)
            for label, dos in phonon_doses.items()
        }
    widget = pmv.phonon_dos(phonon_doses, stack=stack, sigma=sigma, normalize=normalize)
    assert isinstance(widget, pmv.DosWidget)
    assert (widget.stack, widget.sigma, widget.normalize) == (stack, sigma, normalize)
    assert list(widget.doses) == list(phonon_doses)
    for label, dos in phonon_doses.items():
        np.testing.assert_array_equal(widget.doses[label]["densities"], dos.densities)

    single = pmv.phonon_dos(next(iter(phonon_doses.values())))
    assert "densities" in single.doses  # unlabeled DOS is passed as is


@pytest.mark.parametrize(
    ("kwargs", "err_type", "match"),
    [
        (dict(doses={}), ValueError, "Empty DOS dict"),
        (dict(doses="invalid"), TypeError, "Only PhononDos, CompletePhononDos"),
        (dict(normalize="invalid"), ValueError, "Invalid normalize='invalid'"),
        (dict(project="invalid"), ValueError, "Invalid project='invalid'"),
        (dict(project="element"), TypeError, "project='element' requires Complete"),
        (
            dict(
                doses={"a": GRID_012, "b": PhononDos([0, 2, 4], [1, 2, 1])}, stack=True
            ),
            ValueError,
            r"one frequency grid, but \['b'\] differ from 'a'",
        ),
    ],
)
def test_phonon_dos_raises(
    phonon_doses: dict[str, PhononDos],
    kwargs: dict[str, Any],
    err_type: type[Exception],
    match: str,
) -> None:
    """Reject invalid DOS inputs and options."""
    options = dict(kwargs)  # copy: don't mutate the shared parametrize dict
    doses = options.pop("doses", phonon_doses)
    with pytest.raises(err_type, match=match):
        pmv.phonon_dos(doses, **options)


@pytest.mark.parametrize(
    ("project", "show_total", "labeled", "expected_names"),
    [
        ("element", True, False, {"Na", "Cl", "Total"}),
        ("site", False, False, {"Na0", "Cl1"}),
        (
            "element",
            True,
            True,
            {f"{m} - {e}" for m in "AB" for e in ("Na", "Cl", "Total")},
        ),
        ("site", False, True, {f"{m} - {s}" for m in "AB" for s in ("Na0", "Cl1")}),
        (None, True, False, None),  # total DOS only, passed unlabeled
    ],
)
def test_phonon_dos_projection(
    complete_phonon_dos: CompletePhononDos,
    project: Literal["element", "site"] | None,
    show_total: bool,
    labeled: bool,
    expected_names: set[str] | None,
) -> None:
    """Element/site projection plots each partial DOS (and optionally the total)."""
    doses = {"A": complete_phonon_dos, "B": complete_phonon_dos}
    widget = pmv.phonon_dos(
        doses if labeled else complete_phonon_dos,
        project=project,
        show_total=show_total,
    )
    if expected_names is None:
        np.testing.assert_array_equal(
            widget.doses["densities"], complete_phonon_dos.densities
        )
        assert "pdos" not in widget.doses  # CompletePhononDos reduced to its total
        return
    assert set(widget.doses) == expected_names
    expected = (
        {str(elem): dos for elem, dos in complete_phonon_dos.get_element_dos().items()}
        if project == "element"
        else {
            f"{site.specie}{idx}": complete_phonon_dos.get_site_dos(site)
            for idx, site in enumerate(complete_phonon_dos.structure)
        }
    ) | {"Total": complete_phonon_dos}
    for name, doc in widget.doses.items():
        np.testing.assert_array_equal(
            doc["densities"], expected[name.split(" - ")[-1]].densities
        )


def test_phonon_bands_and_dos(phonon_bands_doses_mp_2758: BandsDoses) -> None:
    """Bands and DOS share labels, single inputs pass unlabeled."""
    bands = phonon_bands_doses_mp_2758["bands"]
    doses = phonon_bands_doses_mp_2758["doses"]
    widget = pmv.phonon_bands_and_dos(bands, doses, sigma=0.1, stack=False)
    assert isinstance(widget, pmv.BandsAndDosWidget)
    assert set(widget.band_structs) == set(widget.doses) == {"DFT", "MACE"}
    assert (widget.sigma, widget.stack) == (0.1, False)
    # DFT and MACE DOS use different frequency grids, which MatterViz can't stack
    with pytest.raises(ValueError, match=r"one frequency grid, but \['MACE'\]"):
        pmv.phonon_bands_and_dos(bands, doses, stack=True)
    shared_grid = dict.fromkeys(doses, doses["DFT"])
    assert pmv.phonon_bands_and_dos(bands, shared_grid, stack=True).stack is True

    widget = pmv.phonon_bands_and_dos(bands["DFT"], doses["DFT"])
    assert "bands" in widget.band_structs
    assert "densities" in widget.doses

    with pytest.raises(ValueError, match="must be identical"):
        pmv.phonon_bands_and_dos(bands, {"DFT": doses["DFT"]})


@needs_phonopy
def test_phonopy_inputs(phonopy_nacl: Phonopy) -> None:
    """Phonopy band structures and TotalDos are converted to pymatgen objects."""
    phonopy_nacl.run_mesh([5, 5, 5])
    phonopy_nacl.run_total_dos(freq_pitch=1, use_tetrahedron_method=False)
    total_dos = phonopy_nacl.total_dos
    phonopy_nacl.run_band_structure(
        paths=[np.linspace([0, 0, 0], [1 / 2, 0, 0], 11)], labels=["Γ", "X"]
    )
    band_structure = phonopy_nacl.band_structure
    assert total_dos is not None
    assert band_structure is not None

    widget = pmv.phonon_dos({"DFT": total_dos, "ML": total_dos})
    np.testing.assert_array_equal(
        widget.doses["DFT"]["frequencies"], total_dos.frequency_points
    )
    np.testing.assert_array_equal(widget.doses["ML"]["densities"], total_dos.dos)

    widget = pmv.phonon_bands({"NaCl": band_structure})
    assert np.array(widget.band_structs["NaCl"]["bands"]).shape == (6, 11)
    pmv.phonon_bands_and_dos(band_structure, total_dos)


@pytest.mark.parametrize(
    ("labeled", "show_total"), [(False, True), (True, False), (True, True)]
)
def test_phonon_dos_stack_projection_raises(
    complete_phonon_dos: CompletePhononDos, labeled: bool, show_total: bool
) -> None:
    """MatterViz stacks all curves onto one sum, so stacking partials with their
    total or with another DOS's partials would double count.
    """
    doses = {"A": complete_phonon_dos, "B": complete_phonon_dos}
    with pytest.raises(ValueError, match="needs show_total=False and a single DOS"):
        pmv.phonon_dos(
            doses if labeled else complete_phonon_dos,
            project="element",
            stack=True,
            show_total=show_total,
        )
    widget = pmv.phonon_dos(
        complete_phonon_dos, project="element", stack=True, show_total=False
    )
    assert set(widget.doses) == {"Na", "Cl"}


def test_phonon_bands_drop_eigendisplacements(
    phonon_bands_doses_mp_2758: BandsDoses,
) -> None:
    """Eigendisplacements, unused by the band plot, are not sent to the widget."""
    bands = phonon_bands_doses_mp_2758["bands"]["DFT"]
    bands.eigendisplacements = np.ones((*bands.bands.shape, bands.nb_bands // 3, 3))
    assert "eigendisplacements" in bands.as_dict()

    class Unserializable:
        """Eigendisplacements that fail when as_dict() splits them into lists."""

        @property
        def real(self) -> None:
            """Raise like a (slow) list conversion would happen."""
            raise AssertionError("eigendisplacements were serialized")

    bands.eigendisplacements = sentinel = Unserializable()  # ty: ignore[invalid-assignment]
    widget = pmv.phonon_bands(bands)
    assert "eigendisplacements" not in widget.band_structs
    assert bands.eigendisplacements is sentinel  # input not mutated
    combined = pmv.phonon_bands_and_dos(
        bands, phonon_bands_doses_mp_2758["doses"]["DFT"]
    )
    assert "eigendisplacements" not in combined.band_structs
