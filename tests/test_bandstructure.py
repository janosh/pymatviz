from __future__ import annotations

import json

import plotly.graph_objects as go
import pytest
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.util.testing import TEST_FILES_DIR

from pymatviz import plot_band_structure


@pytest.fixture()
def phonon_band_structures() -> dict[str, PhononBandStructureSymmLine]:
    with open(f"{TEST_FILES_DIR}/NaCl_phonon_bandstructure.json") as file:
        NaCl_dct = json.loads(file.read())
    NaCl_bands = PhononBandStructureSymmLine.from_dict(NaCl_dct)

    with open(f"{TEST_FILES_DIR}/SrTiO3_phonon_bandstructure.json") as file:
        SrTiO3_dct = json.loads(file.read())
    SrTiO3_bands = PhononBandStructureSymmLine.from_dict(SrTiO3_dct)

    return {"NaCl": NaCl_bands, "SrTiO3": SrTiO3_bands}


def test_plot_band_structure(
    phonon_band_structures: dict[str, PhononBandStructureSymmLine],
) -> None:
    # test single band structure
    fig = plot_band_structure(phonon_band_structures["NaCl"])
    assert isinstance(fig, go.Figure)
    assert fig.layout.xaxis.title.text == "Wave Vector"
    assert fig.layout.yaxis.title.text == "Frequency (THz)"
    assert fig.layout.font.size == 16

    x_labels = ("Gamma", "X", "X", "Y", "Y", "Gamma", "Gamma", "Z")
    assert fig.layout.xaxis.ticktext == x_labels
    assert fig.layout.xaxis.range is None
    assert fig.layout.yaxis.range is None

    # test dict
    fig = plot_band_structure({"NaCl": phonon_band_structures["NaCl"]})
    assert isinstance(fig, go.Figure)

    # test multiple band structures
    fig = plot_band_structure(
        {key: phonon_band_structures[key] for key in ("SrTiO3", "SrTiO3")}
    )
    assert isinstance(fig, go.Figure)

    with pytest.raises(
        ValueError, match="No common branches found among the band structures"
    ):
        plot_band_structure(phonon_band_structures)
