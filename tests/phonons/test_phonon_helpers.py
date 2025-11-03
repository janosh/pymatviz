from __future__ import annotations

from typing import TYPE_CHECKING, Any

import plotly.graph_objects as go
import pytest
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine as PhononBands

import pymatviz as pmv
from pymatviz.phonons.helpers import pretty_sym_point


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy as np

    from pymatviz.typing import SetMode
    from tests.conftest import BandsDoses


@pytest.mark.parametrize(
    ("branches", "path_mode", "line_kwargs", "expected_x_labels"),
    [
        (  # test original single dict behavior
            ["GAMMA-X", "X-U"],
            "union",
            dict(width=2),
            tuple("ΓLXΓWXU"),
        ),
        # test empty tuple branches with intersection mode
        ((), "intersection", None, tuple("ΓLXΓWXU")),
        (  # test separate acoustic/optical styling
            ["GAMMA-X"],
            "union",
            {
                "acoustic": dict(width=2.5, dash="solid", name="Acoustic modes"),
                "optical": dict(width=1, dash="dash", name="Optical modes"),
            },
            tuple("ΓLXΓWXU"),
        ),
        (  # test callable line_kwargs
            (),
            "union",
            lambda _freqs, idx: dict(dash="solid" if idx < 3 else "dash"),
            tuple("ΓLXΓWXU"),
        ),
        (  # test shaded_ys as True
            (),
            "union",
            None,
            tuple("ΓLXΓWXU"),
        ),
        pytest.param(  # test invalid line_kwargs structure
            (),
            "union",
            {"acoustic": dict(width=2)},  # missing optical key
            tuple("ΓLXΓWXU"),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ],
)
def test_phonon_bands(
    phonon_bands_doses_mp_2758: BandsDoses,
    branches: Sequence[str],
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
    assert actual_x_labels == expected_x_labels, (
        f"{actual_x_labels=}, {expected_x_labels=}"
    )
    assert fig.layout.xaxis.range is None
    assert fig.layout.yaxis.range == pytest.approx((0, 5.36385427095))

    # test line styling
    if isinstance(line_kwargs, dict) and "acoustic" in line_kwargs:
        # check that acoustic and optical modes have different styles
        trace_names = {trace.name for trace in fig.data}
        assert trace_names == {"", "Acoustic modes", "Optical modes"}

        acoustic_traces = [
            trace for trace in fig.data if trace.name == "Acoustic modes"
        ]
        optical_traces = [trace for trace in fig.data if trace.name == "Optical modes"]

        assert all(t.line.width == 2.5 for t in acoustic_traces)
        assert all(t.line.dash == "solid" for t in acoustic_traces)
        assert all(t.line.width == 1 for t in optical_traces)
        assert all(t.line.dash == "dash" for t in optical_traces)
    elif callable(line_kwargs):
        # check that line width increases with band index
        traces_by_width = sorted(fig.data, key=lambda t: t.line.width)
        assert traces_by_width[0].line.width < traces_by_width[-1].line.width

        # check acoustic/optical line style separation
        acoustic_traces = [trace for trace in fig.data if trace.line.dash == "solid"]
        optical_traces = [trace for trace in fig.data if trace.line.dash == "dash"]
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
    # verify default shading is present
    assert any(
        shape.fillcolor == "gray" and shape.opacity == 0.07
        for shape in fig.layout.shapes or []
    ), f"{fig.layout.shapes=}"

    # test shaded_ys as dict with custom values
    fig = pmv.phonon_bands(
        phonon_bands_doses_mp_2758["bands"]["DFT"],
        path_mode=path_mode,
        branches=branches,
        line_kwargs=line_kwargs,
        shaded_ys={(0, "y_max"): dict(fillcolor="red", opacity=0.1)},
    )
    assert isinstance(fig, go.Figure)
    assert any(
        shape.fillcolor == "red" and shape.opacity == 0.1
        for shape in fig.layout.shapes or []
    ), f"{fig.layout.shapes=}"


def test_phonon_bands_raises(
    phonon_bands_doses_mp_2758: BandsDoses, capsys: pytest.CaptureFixture
) -> None:
    with pytest.raises(
        TypeError,
        match=f"Only {PhononBands.__name__}, phonopy BandStructure or dict supported, "
        "got str",
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

    # test invalid shaded_ys values
    with pytest.raises(TypeError, match="expect shaded_ys as dict, got str"):
        pmv.phonon_bands(
            phonon_bands_doses_mp_2758["bands"]["DFT"],
            shaded_ys="invalid",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="Invalid y_val='invalid', must be one of"):
        pmv.phonon_bands(
            phonon_bands_doses_mp_2758["bands"]["DFT"],
            shaded_ys={(0, "invalid"): dict(fillcolor="red")},  # type: ignore[dict-item]
        )


@pytest.mark.parametrize(
    ("sym_point", "expected"),
    [("Γ", "Γ"), ("Γ|DELTA", "Γ|Δ"), ("GAMMA", "Γ"), ("S_0|SIGMA", "S<sub>0</sub>|Σ")],
)
def test_pretty_sym_point(sym_point: str, expected: str) -> None:
    assert pretty_sym_point(sym_point) == expected
