from __future__ import annotations

import json
from glob import glob
from typing import Literal

import plotly.graph_objects as go
import pytest
from monty.io import zopen
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine as PhononBands
from pymatgen.phonon.dos import PhononDos

from pymatviz import plot_phonon_bands, plot_phonon_bands_and_dos, plot_phonon_dos
from pymatviz.phonons import pretty_sym_point
from pymatviz.utils import TEST_FILES


BandsDoses = dict[str, dict[str, PhononBands | PhononDos]]


@pytest.fixture()
def phonon_bands_doses() -> BandsDoses:
    with zopen(f"{TEST_FILES}/mp-2758-Sr4Se4-pbe.json.lzma") as file:
        dft_dct = json.loads(file.read())
    with zopen(f"{TEST_FILES}/mp-2758-Sr4Se4-mace-y7uhwpje.json.lzma") as file:
        ml_dct = json.loads(file.read())

    bands = {
        "DFT": PhononBands.from_dict(dft_dct["phonon_bandstructure"]),
        "MACE": PhononBands.from_dict(ml_dct["phonon_bandstructure"]),
    }
    doses = {
        "DFT": PhononDos.from_dict(dft_dct["phonon_dos"]),
        "MACE": PhononDos.from_dict(ml_dct["phonon_dos"]),
    }
    return {"bands": bands, "doses": doses}


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


def test_plot_phonon_bands(phonon_bands_doses: BandsDoses) -> None:
    # test single band structure
    fig = plot_phonon_bands(phonon_bands_doses["bands"]["DFT"])
    assert isinstance(fig, go.Figure)
    assert fig.layout.xaxis.title.text == "Wave Vector"
    assert fig.layout.yaxis.title.text == "Frequency (THz)"
    assert fig.layout.font.size == 16

    x_labels = ("Γ", "X", "X", "U|K", "Γ", "Γ", "L", "L", "W", "W", "X")
    assert fig.layout.xaxis.ticktext == x_labels
    assert fig.layout.xaxis.range is None
    assert fig.layout.yaxis.range == pytest.approx((0, 5.36385427095))

    # test dict of band structures
    fig = plot_phonon_bands(phonon_bands_doses["bands"])
    assert isinstance(fig, go.Figure)

    # with pytest.raises(
    #     ValueError, match="No common branches found among the band structures"
    # ):
    #     plot_band_structure(phonon_bands_doses)

    with pytest.raises(
        TypeError, match=f"Only {PhononBands.__name__} objects supported, got str"
    ):
        plot_phonon_bands("invalid input")


@pytest.mark.parametrize(
    "sym_point, expected",
    [("Γ", "Γ"), ("Γ|DELTA", "Γ|Δ"), ("GAMMA", "Γ"), ("S_0|SIGMA", "S<sub>0</sub>|Σ")],
)
def test_prety_sym_point(sym_point: str, expected: str) -> None:
    assert pretty_sym_point(sym_point) == expected


@pytest.mark.parametrize(
    "units, stack, sigma, normalize, last_peak_anno",
    [
        ("eV", False, 0.01, "max", "{key}={last_peak:.1f}"),
        ("meV", False, 0.05, "sum", "{key}={last_peak:.4} ({units})"),
        ("cm-1", True, 0.1, "integral", None),
        ("THz", True, 0.1, None, ""),
    ],
)
def test_plot_phonon_dos(
    phonon_bands_doses: BandsDoses,
    units: Literal["eV", "meV", "cm-1", "THz"],
    stack: bool,
    sigma: float,
    normalize: Literal["max", "sum", "integral"] | None,
    last_peak_anno: str | None,
) -> None:
    fig = plot_phonon_dos(
        phonon_bands_doses["doses"],  # test dict
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

    fig = plot_phonon_dos(
        phonon_bands_doses["doses"]["DFT"],  # test single
        stack=stack,
        sigma=sigma,
        normalize=normalize,
        units=units,
        last_peak_anno=last_peak_anno,
    )
    assert isinstance(fig, go.Figure)


@pytest.mark.parametrize(
    "units, stack, sigma, normalize, last_peak_anno",
    [
        ("eV", False, 0.01, "max", "{key}={last_peak:.1f}"),
        ("meV", False, 0.05, "sum", "{key}={last_peak:.4} ({units})"),
        ("cm-1", True, 0.1, "integral", None),
        ("THz", True, 0.1, None, ""),
    ],
)
def test_plot_phonon_bands_and_dos(
    phonon_bands_doses: BandsDoses,
    phonon_doses: dict[str, PhononDos],  # different keys
    units: Literal["eV", "meV", "cm-1", "THz"],
    stack: bool,
    sigma: float,
    normalize: Literal["max", "sum", "integral"] | None,
    last_peak_anno: str | None,
) -> None:
    bands, doses = phonon_bands_doses["bands"], phonon_bands_doses["doses"]
    dos_kwargs = dict(
        stack=stack,
        sigma=sigma,
        normalize=normalize,
        units=units,
        last_peak_anno=last_peak_anno,
    )
    # test dicts
    fig = plot_phonon_bands_and_dos(bands, doses, dos_kwargs=dos_kwargs)

    assert isinstance(fig, go.Figure)
    assert fig.layout.xaxis.title.text == "Wave Vector"
    assert fig.layout.yaxis.title.text == f"Frequency ({units})"
    assert fig.layout.xaxis2.title.text == "Density of States"
    assert fig.layout.font.size == 16
    # check legend labels
    assert {trace.name for trace in fig.data} == {"DFT", "MACE"}

    fig = plot_phonon_bands_and_dos(bands["DFT"], doses["DFT"])
    assert isinstance(fig, go.Figure)

    band_keys, dos_keys = set(bands), set(phonon_doses)
    with pytest.raises(ValueError) as exc:
        plot_phonon_bands_and_dos(bands, phonon_doses)

    assert str(exc.value) == f"{band_keys=} and {dos_keys=} must be identical"
