from __future__ import annotations

import json
import re
from glob import glob
from typing import TYPE_CHECKING, Any

import plotly.graph_objects as go
import pytest
from monty.io import zopen
from monty.json import MontyDecoder, MSONable
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine as PhononBands
from pymatgen.phonon.dos import PhononDos

import pymatviz as pmv
from pymatviz.utils import TEST_FILES


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    import numpy as np

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
    ("branches", "branch_mode", "line_kwargs"),
    [
        # test original single dict behavior
        (["GAMMA-X", "X-U"], "union", dict(width=2)),
        # test empty tuple branches with intersection mode
        ((), "intersection", None),
        # test separate acoustic/optical styling
        (
            ["GAMMA-X"],
            "union",
            {
                "acoustic": dict(width=2.5, dash="solid", name="Acoustic modes"),
                "optical": dict(width=1, dash="dash", name="Optical modes"),
            },
        ),
        # test callable line_kwargs
        ((), "union", lambda _freqs, idx: dict(dash="solid" if idx < 3 else "dash")),
    ],
)
def test_phonon_bands(
    phonon_bands_doses_mp_2758: BandsDoses,
    branches: tuple[str, str],
    branch_mode: pmv.phonons.BranchMode,
    line_kwargs: dict[str, Any] | Callable[[np.ndarray, int], dict[str, Any]] | None,
) -> None:
    # test single band structure
    fig = pmv.phonon_bands(
        phonon_bands_doses_mp_2758["bands"]["DFT"],
        branch_mode=branch_mode,
        branches=branches,
        line_kwargs=line_kwargs,
    )
    assert isinstance(fig, go.Figure)
    assert fig.layout.xaxis.title.text == "Wave Vector"
    assert fig.layout.yaxis.title.text == "Frequency (THz)"
    assert fig.layout.font.size == 16

    x_labels: tuple[str, ...]
    if branches == ():
        x_labels = ("Γ", "X", "X", "U|K", "Γ", "Γ", "L", "L", "W", "W", "X")
    else:
        x_labels = ("Γ", "U|K") if len(branches) == 1 else ("Γ", "X", "X", "U|K")
    assert fig.layout.xaxis.ticktext == x_labels
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
        branch_mode=branch_mode,
        branches=branches,
        line_kwargs=line_kwargs,
    )
    assert isinstance(fig, go.Figure)
    assert {trace.name for trace in fig.data} == {"DFT", "MACE"}
    assert fig.layout.xaxis.ticktext == x_labels


def test_phonon_bands_raises(
    phonon_bands_doses_mp_2758: BandsDoses, capsys: pytest.CaptureFixture
) -> None:
    with pytest.raises(
        TypeError, match=f"Only {PhononBands.__name__} or dict supported, got str"
    ):
        pmv.phonon_bands("invalid input")

    with pytest.raises(
        ValueError,
        match=re.escape(
            "No common branches with branch_mode='union'.\n"
            "- : GAMMA-X, X-U, K-GAMMA, GAMMA-L, L-W, W-X\n"
            "- Only branches ('foo-bar',) were requested."
        ),
    ):
        pmv.phonon_bands(
            phonon_bands_doses_mp_2758["bands"]["DFT"], branches=("foo-bar",)
        )

    # issues warning when requesting some available and some unavailable branches
    pmv.phonon_bands(
        phonon_bands_doses_mp_2758["bands"]["DFT"], branches=("X-U", "foo-bar")
    )
    stdout, stderr = capsys.readouterr()
    assert stdout == ""
    assert "Warning: missing_branches={'foo-bar'}, available branches:" in stderr

    with pytest.raises(ValueError, match="Invalid branch_mode='invalid'"):
        pmv.phonon_bands(
            phonon_bands_doses_mp_2758["bands"]["DFT"],
            branch_mode="invalid",  # type: ignore[arg-type]
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
