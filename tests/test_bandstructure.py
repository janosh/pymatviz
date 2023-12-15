from __future__ import annotations

import json

import plotly.graph_objects as go
import pytest
from monty.io import zopen
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine

from pymatviz import plot_band_structure
from pymatviz.utils import TEST_FILES


@pytest.fixture()
def phonon_band_structures() -> dict[str, PhononBandStructureSymmLine]:
    with zopen(f"{TEST_FILES}/mp-2758-ph-bands-pbe-vs-mace.json.lzma") as file:
        dct = json.loads(file.read())

    return {k: PhononBandStructureSymmLine.from_dict(v) for k, v in dct.items()}


def test_plot_band_structure(
    phonon_band_structures: dict[str, PhononBandStructureSymmLine],
) -> None:
    # test single band structure
    fig = plot_band_structure(phonon_band_structures["PBE"])
    assert isinstance(fig, go.Figure)
    assert fig.layout.xaxis.title.text == "Wave Vector"
    assert fig.layout.yaxis.title.text == "Frequency (THz)"
    assert fig.layout.font.size == 16

    x_labels = ("Γ", "X", "X", "U|K", "Γ", "Γ", "L", "L", "W", "W", "X")
    assert fig.layout.xaxis.ticktext == x_labels
    assert fig.layout.xaxis.range is None
    assert fig.layout.yaxis.range == pytest.approx((0, 5.36385427095))

    # test dict
    fig = plot_band_structure({"PBE": phonon_band_structures["PBE"]})
    assert isinstance(fig, go.Figure)

    # test multiple band structures
    fig = plot_band_structure(
        {key: phonon_band_structures[key] for key in ("MACE", "MACE")}
    )
    assert isinstance(fig, go.Figure)

    # with pytest.raises(
    #     ValueError, match="No common branches found among the band structures"
    # ):
    #     plot_band_structure(phonon_band_structures)
