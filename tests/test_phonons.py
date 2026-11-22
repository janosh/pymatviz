"""Tests for pymatviz.phonons: PhononDBDoc and phonon_dos."""

from __future__ import annotations

import json
import lzma
from dataclasses import fields, replace
from typing import TYPE_CHECKING, Any, Literal

import kemio
import numpy as np
import pytest

import pymatviz as pmv
from pymatviz.data import TEST_FILES, load_nacl_phonons
from pymatviz.phonons import PhononDBDoc


if TYPE_CHECKING:
    from tests.conftest import BandsDoses


def test_phonon_db_doc_deserialization() -> None:
    """Decode database fields to kemio objects (equal to the raw JSON) without
    changing dataclass construction or replacement.
    """
    with lzma.open(f"{TEST_FILES}/phonons/mp-2758-Sr4Se4-pbe.json.xz", "rt") as source:
        raw = json.load(source)
    original_init = PhononDBDoc.__init__
    doc = PhononDBDoc.from_dict(raw)
    assert isinstance(doc, PhononDBDoc)
    assert isinstance(doc.structure, kemio.Structure)
    assert isinstance(doc.phonon_bandstructure, kemio.PhononBands)
    assert isinstance(doc.phonon_dos, kemio.PhononDos)
    np.testing.assert_array_equal(
        doc.structure.positions, [site["xyz"] for site in raw["structure"]["sites"]]
    )
    np.testing.assert_array_equal(
        doc.phonon_bandstructure.frequencies.T, raw["phonon_bandstructure"]["bands"]
    )
    np.testing.assert_array_equal(
        doc.phonon_dos.total_dos, raw["phonon_dos"]["densities"]
    )
    assert doc.supercell is not None
    assert np.shape(doc.supercell) == (3, 3)
    assert doc.free_energies == raw["free_energies"]
    assert not hasattr(doc, "builder_meta")
    assert replace(doc, formula="SrSe").formula == "SrSe"
    assert doc.formula is None
    assert PhononDBDoc.__init__ is original_init
    kwargs = {field.name: getattr(doc, field.name) for field in fields(doc)}
    assert PhononDBDoc(**kwargs).structure is doc.structure
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        PhononDBDoc(**kwargs, unknown=True)  # ty: ignore[unknown-argument]
    with pytest.raises(TypeError, match="required positional arguments"):
        PhononDBDoc.from_dict({})


GRID_012 = kemio.PhononDos([0, 1, 2], [1, 2, 1])


@pytest.mark.parametrize("labeled", [False, True])
def test_band_structure_widget_phonons(
    phonon_bands_doses_mp_2758: BandsDoses, labeled: bool
) -> None:
    """Single band structures pass unlabeled, dicts keep their labels and bands,
    eigendisplacements (unused by the plot) are not sent.
    """
    bands = phonon_bands_doses_mp_2758["bands"]
    widget = pmv.band_structure(
        band_structs=bands if labeled else bands["DFT"], fermi_level=None
    )
    docs = widget.band_structs if labeled else {"DFT": widget.band_structs}
    assert set(docs) == ({"DFT", "MACE"} if labeled else {"DFT"})
    for label, doc in docs.items():
        np.testing.assert_array_equal(doc["bands"], bands[label].frequencies.T)

    widget = pmv.band_structure(
        band_structs=bands["DFT"], reference_frequency=1.5, style="height:9em"
    )
    assert (widget.reference_frequency, widget.style) == (1.5, "height:9em")
    assert "eigendisplacements" in bands["DFT"].as_dict()
    assert "eigendisplacements" not in widget.band_structs

    with pytest.raises(TypeError, match="Unsupported type for band structure"):
        pmv.band_structure(band_structs="invalid")


@pytest.mark.parametrize(
    ("stack", "sigma", "normalize"),
    [(False, 0, None), (True, 0.1, "max"), (False, 0.2, "integral"), (True, 0, "sum")],
)
def test_phonon_dos(
    phonon_doses: dict[str, kemio.PhononDos],
    stack: bool,
    sigma: float,
    normalize: Literal["max", "sum", "integral"] | None,
) -> None:
    """Raw DOS data goes to the widget, smearing and normalization as traits."""
    if stack:  # MatterViz stacks by index, so stacked DOS must share a grid
        freqs = next(iter(phonon_doses.values())).frequency_points
        phonon_doses = {
            label: kemio.PhononDos(freqs, dos.total_dos)
            for label, dos in phonon_doses.items()
        }
    widget = pmv.phonon_dos(phonon_doses, stack=stack, sigma=sigma, normalize=normalize)
    assert isinstance(widget, pmv.widgets.DosWidget)
    assert (widget.stack, widget.sigma, widget.normalize) == (stack, sigma, normalize)
    assert list(widget.doses) == list(phonon_doses)
    for label, dos in phonon_doses.items():
        np.testing.assert_array_equal(widget.doses[label]["densities"], dos.total_dos)

    single = pmv.phonon_dos(next(iter(phonon_doses.values())))
    assert "densities" in single.doses  # unlabeled DOS is passed as is


@pytest.mark.parametrize(
    ("kwargs", "err_type", "match"),
    [
        (dict(doses={}), ValueError, "Empty DOS dict"),
        (
            dict(doses="invalid"),
            TypeError,
            r"to_phonon_dos takes a kemio, phonopy or pymatgen object.*key ''\)",
        ),
        (dict(normalize="invalid"), ValueError, "Invalid normalize='invalid'"),
        (dict(project="invalid"), ValueError, "Invalid project='invalid'"),
        (dict(project="element"), TypeError, "project='element' needs a DOS with site"),
        (
            dict(
                doses={"a": GRID_012, "b": kemio.PhononDos([0, 2, 4], [1, 2, 1])},
                stack=True,
            ),
            ValueError,
            r"one grid, but \['b'\] differ from 'a'",
        ),
    ],
)
def test_phonon_dos_raises(
    phonon_doses: dict[str, kemio.PhononDos],
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
        ("site", False, False, {"Na1", "Cl2"}),
        ("element", True, True, {"Na", "Cl", "Total"}),  # prefixed with "A - " etc.
        ("site", False, True, {"Na1", "Cl2"}),
        (None, True, False, None),  # total DOS only, passed unlabeled
    ],
)
def test_phonon_dos_projection(
    complete_phonon_dos: kemio.PhononDos,
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
            widget.doses["densities"], complete_phonon_dos.total_dos
        )
        assert "pdos" not in widget.doses  # projected DOS reduced to its total
        return
    if labeled:
        expected_names = {f"{lbl} - {name}" for lbl in "AB" for name in expected_names}
    assert set(widget.doses) == expected_names
    site_densities = complete_phonon_dos.projected_dos
    assert site_densities is not None
    if project == "element":
        projected = complete_phonon_dos.element_dos()
    else:
        projected = {"Na1": site_densities[0], "Cl2": site_densities[1]}
    expected = projected | {"Total": complete_phonon_dos.total_dos}
    for name, doc in widget.doses.items():
        np.testing.assert_array_equal(doc["densities"], expected[name.split(" - ")[-1]])


def test_bands_and_dos_widget_phonons(phonon_bands_doses_mp_2758: BandsDoses) -> None:
    """Bands and DOS share labels, single inputs pass unlabeled (without
    eigendisplacements), stacked DOS need one grid.
    """
    bands = phonon_bands_doses_mp_2758["bands"]
    doses = phonon_bands_doses_mp_2758["doses"]
    widget = pmv.bands_and_dos(band_structs=bands, doses=doses, sigma=0.1, stack=False)
    assert set(widget.band_structs) == set(widget.doses) == {"DFT", "MACE"}
    assert (widget.sigma, widget.stack) == (0.1, False)
    # DFT and MACE DOS use different frequency grids, which MatterViz can't stack
    with pytest.raises(ValueError, match=r"one grid, but \['MACE'\]"):
        pmv.bands_and_dos(band_structs=bands, doses=doses, stack=True)
    with pytest.raises(ValueError, match=r"one grid, but \['MACE'\]"):
        pmv.dos(doses=doses, stack=True)
    list_and_tuple = {  # equal grids in different containers stack
        label: {"frequencies": grid, "densities": [1, 2]}
        for label, grid in (("list", [0, 1]), ("tuple", (0, 1)))
    }
    assert pmv.dos(doses=list_and_tuple, stack=True).stack is True
    shared_grid = dict.fromkeys(doses, doses["DFT"])
    widget = pmv.bands_and_dos(band_structs=bands, doses=shared_grid, stack=True)
    assert widget.stack is True

    widget = pmv.bands_and_dos(band_structs=bands["DFT"], doses=doses["DFT"])
    assert "bands" in widget.band_structs
    assert "eigendisplacements" not in widget.band_structs
    assert "densities" in widget.doses

    for mismatched in ({"DFT": doses["DFT"]}, doses["DFT"]):
        with pytest.raises(ValueError, match="must be identical"):
            pmv.bands_and_dos(band_structs=bands, doses=mismatched)


def test_nacl_phonons_inputs() -> None:
    """Band structures, mesh DOS and projected DOS of kemio Phonons."""
    phonons = load_nacl_phonons()
    mesh = phonons.mesh([5, 5, 5], total_dos=True)
    total_dos = kemio.PhononDos(mesh.frequency_points, mesh.total_dos)
    band_structure = phonons.bands(
        [[[0, 0, 0], [1 / 2, 0, 0]]], labels=["Γ", "X"], n_points=11
    )

    widget = pmv.phonon_dos({"DFT": total_dos, "ML": total_dos})
    np.testing.assert_array_equal(
        widget.doses["DFT"]["frequencies"], mesh.frequency_points
    )
    np.testing.assert_array_equal(widget.doses["ML"]["densities"], mesh.total_dos)

    widget = pmv.band_structure(band_structs={"NaCl": band_structure})
    assert np.array(widget.band_structs["NaCl"]["bands"]).shape == (6, 11)
    widget = pmv.bands_and_dos(band_structs=band_structure, doses=total_dos)
    np.testing.assert_array_equal(widget.doses["densities"], mesh.total_dos)
    np.testing.assert_array_equal(
        widget.band_structs["bands"], band_structure.frequencies.T
    )
    assert widget.band_structs["labels_dict"] == {"Γ": [0, 0, 0], "X": [0.5, 0, 0]}

    projected_dos = phonons.projected_dos([5, 5, 5])
    assert projected_dos.projected_dos is not None
    widget = pmv.phonon_dos(projected_dos, project="element")
    np.testing.assert_allclose(
        np.add(widget.doses["Na"]["densities"], widget.doses["Cl"]["densities"]),
        projected_dos.projected_dos.sum(axis=0),
        rtol=1e-12,
        atol=0,
    )


@pytest.mark.parametrize(
    ("labeled", "show_total"), [(False, True), (True, False), (True, True)]
)
def test_phonon_dos_stack_projection_raises(
    complete_phonon_dos: kemio.PhononDos, labeled: bool, show_total: bool
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
