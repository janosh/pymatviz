from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import plotly.graph_objects as go
import pytest
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText

from pymatviz.utils import (
    CrystalSystem,
    add_identity_line,
    annotate_metrics,
    bin_df_cols,
    df_to_arrays,
    get_crystal_sys,
    save_fig,
)
from tests.conftest import y_pred, y_true


@pytest.mark.parametrize(
    "metrics, prec",
    [
        [["RMSE"], 1],
        [("MAPE", "MSE"), 2],
        [{"MAE", "R2", "RMSE"}, 3],
        [{"MAE": 1, "R2": 2, "RMSE": 3}, 0],
    ],
)
def test_annotate_metrics(metrics: dict[str, float] | Sequence[str], prec: int) -> None:
    text_box = annotate_metrics(y_pred, y_true, metrics=metrics, prec=prec)

    assert isinstance(text_box, AnchoredText)

    expected = dict(MAE=0.113, R2=0.765, RMSE=0.144, MAPE=0.5900, MSE=0.0206)

    txt = ""
    if isinstance(metrics, dict):
        for key, val in metrics.items():
            txt += f"{key} = {val:.{prec}f}\n"
    else:
        for key in metrics:
            txt += f"{key} = {expected[key]:.{prec}f}\n"

    assert text_box.txt.get_text() == txt

    prefix, suffix = "Metrics:\n", "\nthe end"
    text_box = annotate_metrics(
        y_pred, y_true, metrics=metrics, prec=prec, prefix=prefix, suffix=suffix
    )
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
def test_get_crystal_sys(spg_num: int, crystal_sys: CrystalSystem) -> None:
    assert crystal_sys == get_crystal_sys(spg_num)


@pytest.mark.parametrize("spg", [-1, 0, 231])
def test_get_crystal_sys_invalid(spg: int) -> None:
    with pytest.raises(ValueError, match=f"Invalid space group {spg}"):
        get_crystal_sys(spg)


@pytest.mark.parametrize("trace_idx", [0, 1])
@pytest.mark.parametrize("line_kwds", [None, {"color": "blue"}])
def test_add_identity_line(
    plotly_scatter: go.Figure, trace_idx: int, line_kwds: dict[str, str] | None
) -> None:
    fig = add_identity_line(plotly_scatter, line_kwds=line_kwds, trace_idx=trace_idx)
    assert isinstance(fig, go.Figure)

    line = [shape for shape in fig.layout["shapes"] if shape["type"] == "line"][0]
    assert line["x0"] == line["y0"]  # fails if we don't handle nan since nan != nan
    assert line["x1"] == line["y1"]
    assert line["layer"] == "below"
    assert line["line"]["color"] == line_kwds["color"] if line_kwds else "gray"


def test_df_to_arrays() -> None:
    df = pd.DataFrame([y_true, y_pred]).T
    x1, y1 = df_to_arrays(None, y_true, y_pred)
    x_col, y_col = df.columns[:2]
    x2, y2 = df_to_arrays(df, x_col, y_col)
    # TODO find a mypy-compat way to check for exact equality
    assert x1 == pytest.approx(x2)
    assert y1 == pytest.approx(y2)
    assert x1 == pytest.approx(y_true)
    assert y1 == pytest.approx(y_pred)

    with pytest.raises(TypeError, match="df should be pandas DataFrame or None"):
        df_to_arrays("foo", y_true, y_pred)

    bad_col_name = "not-real-col-name"
    with pytest.raises(KeyError) as exc_info:
        df_to_arrays(df, bad_col_name, df.columns[0])

    assert "not-real-col-name" in str(exc_info.value)


@pytest.mark.parametrize("fig", [go.Figure(), plt.figure()])
@pytest.mark.parametrize("ext", ["html", "svelte", "png", "svg", "pdf"])
@pytest.mark.parametrize(
    "plotly_config", [None, {"showTips": True}, {"scrollZoom": True}]
)
def test_save_fig(
    fig: go.Figure | plt.Figure | plt.Axes,
    ext: str,
    tmp_path: Path,
    plotly_config: dict[str, Any] | None,
) -> None:
    if isinstance(fig, plt.Figure) and ext in ("svelte", "html"):
        pytest.skip("svelte not supported for matplotlib figures")

    path = f"{tmp_path}/fig.{ext}"
    save_fig(fig, path, plotly_config=plotly_config)

    if ext in ("svelte", "html"):
        with open(path) as file:
            html = file.read()
        if plotly_config and plotly_config.get("showTips"):
            assert '"showTips": true' in html
        else:
            assert '"showTips": false' in html
        assert '"modeBarButtonsToRemove": ' in html
        assert '"displaylogo": false' in html
        if plotly_config and plotly_config.get("scrollZoom"):
            assert '"scrollZoom": true' in html

        if ext == "svelte":
            assert html.startswith("<div {...$$props}>")
        else:
            assert html.startswith("<div>")


@pytest.mark.parametrize(
    "bin_by_cols, group_by_cols, n_bins, expected_n_bins",
    [
        (["col1"], [], 2, [2]),
        (["col1", "col2"], [], 2, [2, 2]),
        (["col1", "col2"], [], [2, 3], [2, 3]),
        (["col1"], ["col2"], 2, [2]),
    ],
)
@pytest.mark.parametrize("verbose", [True, False])
def test_bin_df_cols(
    bin_by_cols: list[str],
    group_by_cols: list[str],
    n_bins: int | list[int],
    expected_n_bins: list[int],
    verbose: bool,
) -> None:
    data = {"col1": [1, 2, 3, 4], "col2": [2, 3, 4, 5], "col3": [3, 4, 5, 6]}
    df = pd.DataFrame(data)

    df_binned = bin_df_cols(df, bin_by_cols, group_by_cols, n_bins, verbose=verbose)

    df_grouped = (
        df.reset_index()
        .groupby([*[f"{c}_bins" for c in bin_by_cols], *group_by_cols])
        .first()
        .dropna()
    )

    for col, bins in zip(bin_by_cols, expected_n_bins):
        binned_col = f"{col}_bins"
        assert binned_col in df_grouped.index.names

        unique_bins = df_grouped.index.get_level_values(binned_col).nunique()
        assert unique_bins <= bins

    assert not df_binned.empty


def test_bin_df_cols_raises_value_error() -> None:
    df = pd.DataFrame({"col1": [1, 2, 3, 4], "col2": [2, 3, 4, 5]})
    bin_by_cols = ["col1", "col2"]

    with pytest.raises(ValueError):
        bin_df_cols(df, bin_by_cols, n_bins=[2])
