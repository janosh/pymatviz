from __future__ import annotations

import json
import re
from glob import glob
from typing import Literal, Union

import plotly.graph_objects as go
import pytest
from monty.io import zopen
from monty.json import MontyDecoder, MSONable
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine as PhononBands
from pymatgen.phonon.dos import PhononDos

from pymatviz import plot_phonon_bands, plot_phonon_bands_and_dos, plot_phonon_dos
from pymatviz.phonons import BranchMode, pretty_sym_point
from pymatviz.utils import TEST_FILES


BandsDoses = dict[str, dict[str, Union[PhononBands, PhononDos]]]
bs_key, dos_key = "phonon_bandstructure", "phonon_dos"
# enable loading PhononDBDocParsed with @module set to uninstalled ffonons.dbs.phonondb
# by changing to identical dataclass in pymatviz.phonons module
MSONable.REDIRECT["ffonons.dbs.phonondb"] = {
    "PhononDBDocParsed": {"@class": "PhononDBDoc", "@module": "pymatviz.phonons"}
}
MSONable.REDIRECT["atomate2.common.schemas.phonons"] = {
    "PhononBSDOSDoc": {"@class": "PhononDBDoc", "@module": "pymatviz.phonons"}
}


@pytest.fixture()
def phonon_bands_doses_mp_2758() -> BandsDoses:
    with zopen(f"{TEST_FILES}/phonons/mp-2758-Sr4Se4-pbe.json.lzma") as file:
        dft_doc = json.loads(file.read(), cls=MontyDecoder)

    with zopen(f"{TEST_FILES}/phonons/mp-2758-Sr4Se4-mace-y7uhwpje.json.lzma") as file:
        ml_doc = json.loads(file.read(), cls=MontyDecoder)

    bands = {"DFT": getattr(dft_doc, bs_key), "MACE": getattr(ml_doc, bs_key)}
    doses = {"DFT": getattr(dft_doc, dos_key), "MACE": getattr(ml_doc, dos_key)}
    return {"bands": bands, "doses": doses}


@pytest.fixture()
def phonon_bands_doses_mp_2667() -> BandsDoses:
    # with zopen(f"{TEST_FILES}/phonons/mp-2691-Cd4Se4-pbe.json.lzma") as file:
    with zopen(f"{TEST_FILES}/phonons/mp-2667-Cs1Au1-pbe.json.lzma") as file:
        return json.loads(file.read(), cls=MontyDecoder)


@pytest.fixture()
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
    "branches, branch_mode", [(["GAMMA-X", "X-U"], "union"), ((), "intersection")]
)
def test_plot_phonon_bands(
    phonon_bands_doses_mp_2758: BandsDoses,
    branches: tuple[str, str],
    branch_mode: BranchMode,
) -> None:
    # test single band structure
    fig = plot_phonon_bands(
        phonon_bands_doses_mp_2758["bands"]["DFT"],
        branch_mode=branch_mode,
        branches=branches,
    )
    assert isinstance(fig, go.Figure)
    assert fig.layout.xaxis.title.text == "Wave Vector"
    assert fig.layout.yaxis.title.text == "Frequency (THz)"
    assert fig.layout.font.size == 16

    if branches == ():
        x_labels = ["Γ", "X", "X", "U|K", "Γ", "Γ", "L", "L", "W", "W", "X"]
    else:
        x_labels = ["Γ", "X", "X", "U|K"]
    assert list(fig.layout.xaxis.ticktext) == x_labels
    assert fig.layout.xaxis.range is None
    assert fig.layout.yaxis.range == pytest.approx((0, 5.36385427095))

    # test dict of band structures
    fig = plot_phonon_bands(
        phonon_bands_doses_mp_2758["bands"], branch_mode=branch_mode, branches=branches
    )
    assert isinstance(fig, go.Figure)


def test_plot_phonon_bands_raises(
    phonon_bands_doses_mp_2758: BandsDoses, capsys: pytest.CaptureFixture
) -> None:
    with pytest.raises(
        TypeError, match=f"Only {PhononBands.__name__} or dict supported, got str"
    ):
        plot_phonon_bands("invalid input")

    with pytest.raises(ValueError) as exc:  # noqa: PT011
        plot_phonon_bands(
            phonon_bands_doses_mp_2758["bands"]["DFT"], branches=("foo-bar",)
        )

    assert (
        "No common branches with branch_mode='union'.\n"
        "- : GAMMA-X, X-U, K-GAMMA, GAMMA-L, L-W, W-X\n"
        "- Only branches ('foo-bar',) were requested." in str(exc.value)
    )

    # issues warning when requesting some available and some unavailable branches
    plot_phonon_bands(
        phonon_bands_doses_mp_2758["bands"]["DFT"], branches=("X-U", "foo-bar")
    )
    stdout, stderr = capsys.readouterr()
    assert stdout == ""
    assert "Warning: missing_branches={'foo-bar'}, available branches:" in stderr

    with pytest.raises(ValueError, match="Invalid branch_mode='invalid'"):
        plot_phonon_bands(
            phonon_bands_doses_mp_2758["bands"]["DFT"],
            branch_mode="invalid",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="Empty band structure dict"):
        plot_phonon_bands({})


@pytest.mark.parametrize(
    "sym_point, expected",
    [("Γ", "Γ"), ("Γ|DELTA", "Γ|Δ"), ("GAMMA", "Γ"), ("S_0|SIGMA", "S<sub>0</sub>|Σ")],
)
def test_pretty_sym_point(sym_point: str, expected: str) -> None:
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
    phonon_bands_doses_mp_2758: BandsDoses,
    units: Literal["eV", "meV", "cm-1", "THz"],
    stack: bool,
    sigma: float,
    normalize: Literal["max", "sum", "integral"] | None,
    last_peak_anno: str | None,
) -> None:
    fig = plot_phonon_dos(
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

    fig = plot_phonon_dos(
        phonon_bands_doses_mp_2758["doses"]["DFT"],  # test single
        stack=stack,
        sigma=sigma,
        normalize=normalize,
        units=units,
        last_peak_anno=last_peak_anno,
    )
    assert isinstance(fig, go.Figure)


def test_plot_phonon_dos_raises(phonon_bands_doses_mp_2758: BandsDoses) -> None:
    with pytest.raises(
        TypeError, match=f"Only {PhononDos.__name__} or dict supported, got str"
    ):
        plot_phonon_dos("invalid input")

    with pytest.raises(ValueError, match="Empty DOS dict"):
        plot_phonon_dos({})

    expected_msg = (
        "Invalid unit='invalid', must be one of ['THz', 'eV', 'meV', 'Ha', 'cm-1']"
    )
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        plot_phonon_dos(phonon_bands_doses_mp_2758["doses"], units="invalid")  # type: ignore[arg-type]


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
    fig = plot_phonon_bands_and_dos(bands, doses, dos_kwargs=dos_kwargs)

    assert isinstance(fig, go.Figure)
    assert fig.layout.xaxis.title.text == "Wave Vector"
    assert fig.layout.yaxis.title.text == f"Frequency ({units})"
    assert fig.layout.xaxis2.title.text == "DOS"
    assert fig.layout.font.size == 16
    # check legend labels
    assert {trace.name for trace in fig.data} == {"DFT", "MACE"}

    fig = plot_phonon_bands_and_dos(bands["DFT"], doses["DFT"])
    assert isinstance(fig, go.Figure)

    band_keys, dos_keys = set(bands), set(phonon_doses)
    with pytest.raises(
        ValueError, match=f"{band_keys=} and {dos_keys=} must be identical"
    ):
        plot_phonon_bands_and_dos(bands, phonon_doses)
