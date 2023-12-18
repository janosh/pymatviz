from __future__ import annotations

import json
from glob import glob
from typing import Literal

import plotly.graph_objects as go
import pytest
from monty.io import zopen
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine as PhononBands
from pymatgen.phonon.dos import PhononDos

from pymatviz import plot_phonon_bands, plot_phonon_dos
from pymatviz.phonons import pretty_sym_point
from pymatviz.utils import TEST_FILES


@pytest.fixture()
def phonon_band_structures() -> dict[str, PhononBands]:
    with zopen(f"{TEST_FILES}/mp-2758-ph-bands-pbe-vs-mace.json.lzma") as file:
        dct = json.loads(file.read())

    return {key: PhononBands.from_dict(val) for key, val in dct.items()}


@pytest.fixture()
def phonon_doses() -> dict[str, PhononDos]:
    paths = glob(f"{TEST_FILES}/mp-*-pbe.json.lzma")
    assert len(paths) >= 2
    return {
        path.split("/")[-1].split("-pbe")[0]: PhononDos.from_dict(
            json.loads(zopen(path).read())["phonon_dos"]
        )
        for path in paths
    }


def test_plot_band_structure(phonon_band_structures: dict[str, PhononBands]) -> None:
    # test single band structure
    fig = plot_phonon_bands(phonon_band_structures["PBE"])
    assert isinstance(fig, go.Figure)
    assert fig.layout.xaxis.title.text == "Wave Vector"
    assert fig.layout.yaxis.title.text == "Frequency (THz)"
    assert fig.layout.font.size == 16

    x_labels = ("Γ", "X", "X", "U|K", "Γ", "Γ", "L", "L", "W", "W", "X")
    assert fig.layout.xaxis.ticktext == x_labels
    assert fig.layout.xaxis.range is None
    assert fig.layout.yaxis.range == pytest.approx((0, 5.36385427095))

    # test dict
    fig = plot_phonon_bands({"PBE": phonon_band_structures["PBE"]})
    assert isinstance(fig, go.Figure)

    # test multiple band structures
    fig = plot_phonon_bands(
        {key: phonon_band_structures[key] for key in ("MACE", "MACE")}
    )
    assert isinstance(fig, go.Figure)

    # with pytest.raises(
    #     ValueError, match="No common branches found among the band structures"
    # ):
    #     plot_band_structure(phonon_band_structures)


@pytest.mark.parametrize(
    "sym_point, expected",
    [("Γ", "Γ"), ("Γ|DELTA", "Γ|Δ"), ("GAMMA", "Γ"), ("S_0|SIGMA", "S<sub>0</sub>|Σ")],
)
def test_prety_sym_point(sym_point: str, expected: str) -> None:
    assert pretty_sym_point(sym_point) == expected


@pytest.mark.parametrize(
    "units, stack, sigma, normalize",
    [
        ("eV", False, 0.01, "max"),
        ("meV", False, 0.05, "sum"),
        ("cm-1", True, 0.1, "integral"),
        ("THz", True, 0.1, None),
    ],
)
def test_plot_phonon_dos(
    phonon_doses: dict[str, PhononDos],
    units: Literal["eV", "meV", "cm-1", "THz"],
    stack: bool,
    sigma: float,
    normalize: Literal["max", "sum", "integral"] | None,
) -> None:
    fig = plot_phonon_dos(
        phonon_doses, stack=stack, sigma=sigma, normalize=normalize, units=units
    )  # test dict

    assert isinstance(fig, go.Figure)
    assert fig.layout.xaxis.title.text == f"Frequency ({units})"
    assert fig.layout.yaxis.title.text == "Density of States"
    assert fig.layout.font.size == 16

    fig = plot_phonon_dos(
        phonon_doses["mp-2691-Cd4Se4"],
        stack=stack,
        sigma=sigma,
        normalize=normalize,
        units=units,
    )  # test single
    assert isinstance(fig, go.Figure)
