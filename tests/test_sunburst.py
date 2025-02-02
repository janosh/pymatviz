from __future__ import annotations

from typing import TYPE_CHECKING, get_args

import pandas as pd
import plotly.graph_objects as go
import pytest

import pymatviz as pmv
from pymatviz.sunburst import ShowCounts, spacegroup_sunburst


if TYPE_CHECKING:
    from pymatgen.core import Structure


@pytest.mark.parametrize("show_counts", get_args(ShowCounts))
def test_spacegroup_sunburst(show_counts: ShowCounts) -> None:
    # spg numbers
    fig = spacegroup_sunburst(range(1, 231), show_counts=show_counts)

    assert isinstance(fig, go.Figure)
    assert set(fig.data[0].parents) == {"", *get_args(pmv.typing.CrystalSystem)}
    assert fig.data[0].branchvalues == "total"

    if show_counts == "value":
        assert "texttemplate" in fig.data[0]
        assert "N=" in fig.data[0].texttemplate

    elif show_counts == "value+percent":
        assert "texttemplate" in fig.data[0]
        assert "N=" in fig.data[0].texttemplate
        assert "percentEntry" in fig.data[0].texttemplate

    elif show_counts == "percent":
        assert "textinfo" in fig.data[0]
        assert "percent" in fig.data[0].textinfo

    elif show_counts is False:
        assert fig.data[0].textinfo is None


def test_spacegroup_sunburst_invalid_show_counts() -> None:
    """Test that invalid show_counts values raise ValueError."""
    show_counts = "invalid"
    with pytest.raises(ValueError, match=f"Invalid {show_counts=}"):
        spacegroup_sunburst([1], show_counts=show_counts)  # type: ignore[arg-type]


def test_spacegroup_sunburst_single_item() -> None:
    """Test with single-item input."""
    fig = spacegroup_sunburst([1], show_counts="value")
    assert isinstance(fig, go.Figure)
    assert len(fig.data[0].ids) == 2  # one for crystal system, one for spg number

    fig = spacegroup_sunburst(["P1"], show_counts="value+percent")
    assert isinstance(fig, go.Figure)
    assert len(fig.data[0].ids) == 2


def test_spacegroup_sunburst_other_types(
    spg_symbols: list[str], structures: list[Structure]
) -> None:
    """Test with other types of input."""
    # test with pandas series
    series = pd.Series([*[1] * 3, *[2] * 10, *[3] * 5])
    fig = spacegroup_sunburst(series, show_counts="value")
    assert isinstance(fig, go.Figure)
    values = [*map(int, fig.data[0].values)]
    assert values == [10, 5, 3, 13, 5], f"actual {values=}"

    # test with strings of space group symbols
    fig = spacegroup_sunburst(spg_symbols)
    assert isinstance(fig, go.Figure)

    # test with pymatgen structures
    fig = spacegroup_sunburst(structures)
    assert isinstance(fig, go.Figure)
