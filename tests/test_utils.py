from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest
from matplotlib.offsetbox import AnchoredText

from pymatviz.utils import (
    add_identity_line,
    add_mae_r2_box,
    df_to_arrays,
    get_crystal_sys,
)
from tests.conftest import y_pred, y_true


def test_add_mae_r2_box():

    text_box = add_mae_r2_box(y_pred, y_true)

    assert isinstance(text_box, AnchoredText)

    txt = "$\\mathrm{MAE} = 0.113$\n$R^2 = 0.765$"
    assert text_box.txt.get_text() == txt

    prefix, suffix = "Metrics:\n", "\nthe end"
    text_box = add_mae_r2_box(y_pred, y_true, prefix=prefix, suffix=suffix)
    assert text_box.txt.get_text() == prefix + txt + suffix


@pytest.mark.parametrize(
    "spg_num, crystal_sys",
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
def test_get_crystal_sys(spg_num, crystal_sys):
    assert crystal_sys == get_crystal_sys(spg_num)


@pytest.mark.parametrize("spg", [-1, 0, 231])
def test_get_crystal_sys_invalid(spg):
    with pytest.raises(ValueError, match=f"Invalid space group {spg}"):
        get_crystal_sys(spg)


@pytest.mark.parametrize("trace_idx", [0, 1])
@pytest.mark.parametrize("line_kwds", [None, {"color": "blue"}])
def test_add_identity_line(plotly_scatter, trace_idx, line_kwds):
    fig = add_identity_line(plotly_scatter, trace_idx, line_kwds)
    assert isinstance(fig, go.Figure)

    line = [shape for shape in fig.layout["shapes"] if shape["type"] == "line"][0]
    assert line["x0"] == line["y0"]  # fails if we don't handle nan since nan != nan
    assert line["x1"] == line["y1"]
    assert line["layer"] == "below"
    assert line["line"]["color"] == line_kwds["color"] if line_kwds else "gray"


def test_df_to_arrays():
    df = pd.DataFrame([y_true, y_pred]).T
    x1, y1 = df_to_arrays(y_true, y_pred, df=None)
    x_col, y_col = df.columns[:2]
    x2, y2 = df_to_arrays(x=x_col, y=y_col, df=df)
    assert all(x1 == x2) and all(y1 == y2)
    assert all(x1 == y_true) and all(y1 == y_pred)

    with pytest.raises(TypeError, match="df should be pandas DataFrame or None"):
        df_to_arrays(y_true, y_pred, df="foo")

    with pytest.raises(ValueError, match="x and y must be column names of df.columns="):
        df_to_arrays("not-real-col-name", df.columns[0], df=df)
