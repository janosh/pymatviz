from __future__ import annotations

import pytest
from matplotlib.offsetbox import AnchoredText
from plotly.graph_objs._figure import Figure

from pymatviz.utils import add_identity_line, add_mae_r2_box, get_crystal_sys

from .conftest import y_pred, y_true


def test_add_mae_r2_box():

    text_box = add_mae_r2_box(y_pred, y_true)

    assert isinstance(text_box, AnchoredText)

    txt = "$\\mathrm{MAE} = 0.113$\n$R^2 = 0.765$"
    assert text_box.txt.get_text() == txt

    prefix, suffix = "Metrics:\n", "\nthe end"
    text_box = add_mae_r2_box(y_pred, y_true, prefix=prefix, suffix=suffix)
    assert text_box.txt.get_text() == prefix + txt + suffix


@pytest.mark.parametrize(
    "input, expected",
    [
        (1, "triclinic"),
        (15, "monoclinic"),
        (16, "orthorhombic"),
        (75, "tetragonal"),
        (143, "trigonal"),
        (168, "hexagonal"),
        (230, "cubic"),
    ],
)
def test_get_crystal_sys(input, expected):
    assert expected == get_crystal_sys(input)


@pytest.mark.parametrize("spg", [-1, 0, 231])
def test_get_crystal_sys_invalid(spg):
    with pytest.raises(ValueError, match=f"Invalid space group {spg}"):
        get_crystal_sys(spg)


@pytest.mark.parametrize("trace_idx", [0, 1])
@pytest.mark.parametrize("line_kwds", [None, {"color": "blue"}])
def test_add_identity_line(plotly_scatter, trace_idx, line_kwds):
    fig = add_identity_line(plotly_scatter, trace_idx, line_kwds)
    assert isinstance(fig, Figure)
