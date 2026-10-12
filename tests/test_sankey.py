from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

import pymatviz as pmv


if TYPE_CHECKING:
    from typing import Any, Literal


@pytest.mark.parametrize(
    ("labels_with_counts", "annotate_columns", "expected_xshift", "expected_font_size"),
    [
        (True, True, 35, 20),
        ("percent", {"font_size": 14, "xshift": 25}, 25, 14),
        (False, False, None, None),
    ],
)
def test_sankey_from_2_df_cols(
    df_float: pd.DataFrame,
    labels_with_counts: bool | Literal["percent"],
    annotate_columns: bool | dict[str, Any],
    expected_xshift: int | None,
    expected_font_size: int | None,
) -> None:
    """Column annotations flank the diagram and can be customized or disabled."""
    anno_kwargs_orig = (
        dict(annotate_columns) if isinstance(annotate_columns, dict) else None
    )
    fig = pmv.sankey_from_2_df_cols(
        df_float,
        ["A", "B"],
        labels_with_counts=labels_with_counts,
        annotate_columns=annotate_columns,
    )
    flow_data = pmv.sankey_flow_data(
        df_float, ["A", "B"], labels_with_counts=labels_with_counts
    )
    assert list(fig.data[0].node.label) == flow_data["labels"]
    if isinstance(annotate_columns, dict):  # caller's dict is not mutated
        assert annotate_columns == anno_kwargs_orig
    if expected_xshift is None:
        assert len(fig.layout.annotations) == 0
        return
    assert [anno.text for anno in fig.layout.annotations] == ["<b>A</b>", "<b>B</b>"]
    assert [anno.xshift for anno in fig.layout.annotations] == [
        -expected_xshift,
        expected_xshift,
    ]
    assert {anno.font.size for anno in fig.layout.annotations} == {expected_font_size}


def test_sankey_with_invalid_columns() -> None:
    """Test handling of invalid column names."""
    df_in = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(KeyError):
        pmv.sankey_from_2_df_cols(df_in, ["nonexistent", "b"])
