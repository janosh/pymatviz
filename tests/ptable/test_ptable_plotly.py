from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import plotly.graph_objects as go
import pytest
from plotly.exceptions import PlotlyError

from pymatviz import ptable_heatmap_plotly
from pymatviz.utils import df_ptable


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any


def test_ptable_heatmap_plotly(glass_formulas: list[str]) -> None:
    fig = ptable_heatmap_plotly(glass_formulas)
    assert isinstance(fig, go.Figure)
    assert (
        len(fig.layout.annotations) == 18 * 10
    ), "not all periodic table tiles have annotations"
    assert (
        sum(anno.text != "" for anno in fig.layout.annotations) == 118
    ), "no annotations should be empty"

    # test hover_props and show_values=False
    ptable_heatmap_plotly(
        glass_formulas,
        hover_props=("atomic_mass", "atomic_number", "density"),
        show_values=False,
    )
    ptable_heatmap_plotly(
        glass_formulas,
        hover_data="density = " + df_ptable.density.astype(str) + " g/cm^3",
    )
    # test label_map as dict
    fig = ptable_heatmap_plotly(df_ptable.density, fmt=".1f", label_map={"0": "zero"})
    # test label_map as callable
    ptable_heatmap_plotly(
        df_ptable.density,
        fmt=".1f",
        label_map=lambda x: "meaning of life" if x == 42 else x,
    )

    ptable_heatmap_plotly(glass_formulas, heat_mode="percent")

    # test log color scale with -1, 0, 1 and random negative value
    for val in (-9.72, -1, 0, 1):
        ptable_heatmap_plotly([f"H{val}", "O2"], log=True)
        df_ptable["tmp"] = val
        fig = ptable_heatmap_plotly(df_ptable["tmp"], log=True)
        assert isinstance(fig, go.Figure)
        heatmap = next(
            trace
            for trace in fig.data
            if isinstance(trace, go.Heatmap) and "colorbar" in trace
        )
        assert heatmap.colorbar.title.text == "tmp"
        c_scale = heatmap.colorscale
        assert isinstance(c_scale, tuple)
        assert isinstance(c_scale[0], tuple)
        assert isinstance(c_scale[0][0], float)
        assert isinstance(c_scale[0][1], str)
        assert val <= max(c[0] for c in c_scale)

    with pytest.raises(TypeError, match="should be string, list of strings or list"):
        # test that bad colorscale raises ValueError
        ptable_heatmap_plotly(glass_formulas, colorscale=lambda: "bad scale")  # type: ignore[arg-type]

    # test that unknown builtin colorscale raises ValueError
    with pytest.raises(PlotlyError, match="Colorscale foobar is not a built-in scale"):
        ptable_heatmap_plotly(glass_formulas, colorscale="foobar")

    with pytest.raises(ValueError, match="Combining log color scale and"):
        ptable_heatmap_plotly(glass_formulas, log=True, heat_mode="percent")


@pytest.mark.parametrize("exclude_elements", [(), [], ["O", "P"]])
@pytest.mark.parametrize(
    "heat_mode, log", [(None, True), ("fraction", False), ("percent", False)]
)
@pytest.mark.parametrize("show_scale", [False, True])
@pytest.mark.parametrize("font_size", [None, 14])
@pytest.mark.parametrize("font_colors", [["red"], ("black", "white")])
def test_ptable_heatmap_plotly_kwarg_combos(
    glass_formulas: list[str],
    exclude_elements: Sequence[str],
    heat_mode: Literal["value", "fraction", "percent"] | None,
    show_scale: bool,
    font_size: int,
    font_colors: tuple[str] | tuple[str, str],
    log: bool,
) -> None:
    fig = ptable_heatmap_plotly(
        glass_formulas,
        exclude_elements=exclude_elements,
        heat_mode=heat_mode,
        show_scale=show_scale,
        font_size=font_size,
        font_colors=font_colors,
        log=log,
    )
    assert isinstance(fig, go.Figure)


@pytest.mark.parametrize(
    "colorscale", ["YlGn", ["blue", "red"], [(0, "blue"), (1, "red")]]
)
def test_ptable_heatmap_plotly_colorscale(
    glass_formulas: list[str], colorscale: str | list[tuple[float, str]] | list[str]
) -> None:
    fig = ptable_heatmap_plotly(glass_formulas, colorscale=colorscale)
    assert isinstance(fig, go.Figure)


@pytest.mark.parametrize(
    "color_bar", [{}, dict(orientation="v", len=0.8), dict(orientation="h", len=0.3)]
)
def test_ptable_heatmap_plotly_color_bar(
    glass_formulas: list[str], color_bar: dict[str, Any]
) -> None:
    fig = ptable_heatmap_plotly(glass_formulas, color_bar=color_bar)
    # check color bar has expected length
    assert fig.data[0].colorbar.len == color_bar.get("len", 0.4)
    # check color bar has expected title side
    assert (
        fig.data[0].colorbar.title.side == "right"
        if color_bar.get("orientation") == "v"
        else "top"
    )


@pytest.mark.parametrize(
    "cscale_range", [(None, None), (None, 10), (2, None), (2, 87123)]
)
def test_ptable_heatmap_plotly_cscale_range(
    cscale_range: tuple[float | None, float | None],
) -> None:
    fig = ptable_heatmap_plotly(df_ptable.density, cscale_range=cscale_range)
    trace = fig.data[0]
    assert "colorbar" in trace
    # check for correct color bar range
    if cscale_range == (None, None):
        # if both None, range is dynamic based on plotted data
        assert trace["zmin"] == pytest.approx(df_ptable.density.min())
        assert trace["zmax"] == pytest.approx(df_ptable.density.max())
    else:
        assert cscale_range == (trace["zmin"], trace["zmax"])


def test_ptable_heatmap_plotly_cscale_range_raises() -> None:
    cscale_range = (0, 10, 20)
    with pytest.raises(
        ValueError, match=re.escape(f"{cscale_range=} should have length 2")
    ):
        ptable_heatmap_plotly(df_ptable.density, cscale_range=cscale_range)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "label_map",
    [None, False, {"1.0": "one", "2.0": "two", "3.0": "three", np.nan: "N/A"}],
)
def test_ptable_heatmap_plotly_label_map(
    label_map: dict[str, str] | Literal[False] | None,
) -> None:
    elem_vals = dict(Al=1.0, Cr=2.0, Fe=3.0, Ni=np.nan)
    fig = ptable_heatmap_plotly(elem_vals, label_map=label_map)
    assert isinstance(fig, go.Figure)

    # if label_map is not False, ensure mapped labels appear in figure annotations
    if label_map is not False:
        if label_map is None:
            # use default map
            label_map = dict.fromkeys([np.nan, None, "nan"], " ")  # type: ignore[list-item]
        # check for non-empty intersection between label_map values and annotations
        # we use `val in anno.text` cause the labels are wrapped in non-matching
        # HTML <span> tags
        assert sum(
            any(val in anno.text for val in label_map.values())
            for anno in fig.layout.annotations
        )
