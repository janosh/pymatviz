from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
import pytest

import pymatviz as pmv


if TYPE_CHECKING:
    from typing import Literal


@pytest.mark.parametrize("labels_with_counts", [True, False, "percent"])
def test_sankey_from_2_df_cols(
    df_float: pd.DataFrame, labels_with_counts: bool | Literal["percent"]
) -> None:
    """Test basic functionality with random data."""
    fig = pmv.sankey_from_2_df_cols(
        df_float, ["A", "B"], labels_with_counts=labels_with_counts
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.layout.annotations) == 2
    assert [anno.text for anno in fig.layout.annotations] == ["<b>A</b>", "<b>B</b>"]


def test_sankey_annotate_columns_dict(df_float: pd.DataFrame) -> None:
    """Test annotate_columns with custom dictionary settings."""
    anno_kwargs = {"font_size": 20, "xshift": (xshift := 25)}

    fig = pmv.sankey_from_2_df_cols(df_float, ["A", "B"], annotate_columns=anno_kwargs)

    assert anno_kwargs == {"font_size": 20, "xshift": xshift}
    assert isinstance(fig, go.Figure)
    # Check if annotations are present in layout
    assert len(fig.layout.annotations) == 2
    # Verify annotation properties
    for anno in fig.layout.annotations:
        assert anno.font.size == 20
        assert abs(anno.xshift) == xshift
        assert anno.text in ["<b>A</b>", "<b>B</b>"]


def test_sankey_annotate_columns_false(df_float: pd.DataFrame) -> None:
    """annotate_columns=False suppresses default column labels."""
    fig = pmv.sankey_from_2_df_cols(
        df_float,
        ["A", "B"],
        annotate_columns=False,
    )
    assert len(fig.layout.annotations) == 0


def test_sankey_with_invalid_columns() -> None:
    """Test handling of invalid column names."""
    df_in = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(KeyError):
        pmv.sankey_from_2_df_cols(df_in, ["nonexistent", "b"])
