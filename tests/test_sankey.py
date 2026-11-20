from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

import pymatviz as pmv
from pymatviz.widgets.sankey import SankeyWidget


if TYPE_CHECKING:
    from typing import Literal


@pytest.mark.parametrize("labels_with_counts", [True, "percent", False])
def test_sankey_from_2_df_cols(
    df_float: pd.DataFrame, labels_with_counts: bool | Literal["percent"]
) -> None:
    """Widget data is sankey_flow_data and links cover every row."""
    widget = pmv.sankey_from_2_df_cols(
        df_float, ["A", "B"], labels_with_counts=labels_with_counts, node_align="left"
    )
    assert isinstance(widget, SankeyWidget)
    assert widget.node_align == "left"
    flow = pmv.sankey_flow_data(
        df_float, ["A", "B"], labels_with_counts=labels_with_counts
    )
    assert widget.data == flow
    assert sum(lnk["value"] for lnk in widget.data["links"]) == len(df_float)


def test_sankey_shared_values_get_two_nodes() -> None:
    """A value in both columns becomes one node per side."""
    df_in = pd.DataFrame({"a": ["x", "x", "y"], "b": ["x", "y", "y"]})
    widget = pmv.sankey_from_2_df_cols(df_in, ["a", "b"], labels_with_counts=False)
    nodes, links = widget.data["nodes"], widget.data["links"]
    assert [node["label"] for node in nodes] == ["x", "y", "x", "y"]
    assert [node["metadata"]["column"] for node in nodes] == ["a", "a", "b", "b"]
    assert sorted((lnk["source"], lnk["target"], lnk["value"]) for lnk in links) == [
        (0, 2, 1),
        (0, 3, 1),
        (1, 3, 1),
    ]


@pytest.mark.parametrize(
    ("cols", "err"),
    [(["nonexistent", "b"], KeyError), (["a"], ValueError), (["a", "c"], ValueError)],
)
def test_sankey_with_invalid_columns(cols: list[str], err: type[Exception]) -> None:
    """Missing columns and wrong column counts raise."""
    df_in = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [None, None]})
    with pytest.raises(err):
        pmv.sankey_from_2_df_cols(df_in, cols)
