from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

import pymatviz as pmv
from pymatviz.sankey import sankey_flow_data
from pymatviz.widgets.sankey import SankeyWidget


if TYPE_CHECKING:
    from typing import Any, Literal


@pytest.mark.parametrize("labels_with_counts", [True, "percent", False])
def test_sankey(
    df_float: pd.DataFrame, labels_with_counts: bool | Literal["percent"]
) -> None:
    """Widget data is sankey_flow_data and links cover every row."""
    widget = pmv.sankey(
        df_float, "A", "B", labels_with_counts=labels_with_counts, node_align="left"
    )
    assert isinstance(widget, SankeyWidget)
    assert widget.node_align == "left"
    flow = sankey_flow_data(df_float, "A", "B", labels_with_counts=labels_with_counts)
    assert widget.data == flow
    assert sum(lnk["value"] for lnk in widget.data["links"]) == len(df_float)


def test_sankey_shared_values_get_two_nodes() -> None:
    """A value in both columns becomes one node per side."""
    df_in = pd.DataFrame({"a": ["x", "x", "y"], "b": ["x", "y", "y"]})
    widget = pmv.sankey(df_in, "a", "b", labels_with_counts=False)
    nodes, links = widget.data["nodes"], widget.data["links"]
    assert [node["label"] for node in nodes] == ["x", "y", "x", "y"]
    assert [node["metadata"]["column"] for node in nodes] == ["a", "a", "b", "b"]
    assert sorted((lnk["source"], lnk["target"], lnk["value"]) for lnk in links) == [
        (0, 2, 1),
        (0, 3, 1),
        (1, 3, 1),
    ]


@pytest.mark.parametrize(
    ("source", "target", "err"),
    [("nonexistent", "b", KeyError), ("a", "c", ValueError)],
)
def test_sankey_with_invalid_columns(
    source: str, target: str, err: type[Exception]
) -> None:
    """Missing columns and columns without shared rows raise."""
    df_in = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [None, None]})
    with pytest.raises(err):
        pmv.sankey(df_in, source, target)


@pytest.mark.parametrize(
    ("labels_with_counts", "expected_labels"),
    [
        (True, ["A: 3", "B: 1", "C: 3", "A: 1"]),
        ("percent", ["A: 75.0%", "B: 25.0%", "C: 75.0%", "A: 25.0%"]),
        (False, ["A", "B", "C", "A"]),
    ],
)
def test_sankey_flow_data(
    labels_with_counts: bool | Literal["percent"], expected_labels: list[str]
) -> None:
    """Values in both columns get a node per side, so no link is a self-loop."""
    df_flows = pd.DataFrame({"src": ["A", "A", "A", "B"], "tgt": ["C", "C", "A", "C"]})
    flow_data = sankey_flow_data(
        df_flows, "src", "tgt", labels_with_counts=labels_with_counts
    )
    columns = ["src", "src", "tgt", "tgt"]
    assert flow_data == {
        "nodes": [
            {"id": idx, "label": label, "metadata": {"column": col}}
            for idx, (label, col) in enumerate(
                zip(expected_labels, columns, strict=True)
            )
        ],
        "links": [
            {"source": 0, "target": 2, "value": 2},
            {"source": 0, "target": 3, "value": 1},
            {"source": 1, "target": 2, "value": 1},
        ],
    }


@pytest.mark.parametrize(
    ("labels_with_counts", "expected_labels"),
    [
        (True, ["a: 2", "b: 1", "x: 2", "y: 1"]),
        ("percent", ["a: 66.7%", "b: 33.3%", "x: 66.7%", "y: 33.3%"]),
    ],
)
def test_sankey_flow_data_labels_match_flows_with_nan(
    labels_with_counts: bool | Literal["percent"], expected_labels: list[str]
) -> None:
    """Rows missing either value count toward neither node labels nor links."""
    df_flows = pd.DataFrame(
        {"src": ["a", "a", "a", "b", None], "tgt": ["x", "x", None, "y", "y"]}
    )
    flow_data = sankey_flow_data(
        df_flows, "src", "tgt", labels_with_counts=labels_with_counts
    )
    assert [node["label"] for node in flow_data["nodes"]] == expected_labels
    assert flow_data["links"] == [
        {"source": 0, "target": 2, "value": 2},
        {"source": 1, "target": 3, "value": 1},
    ]


@pytest.mark.parametrize(
    ("source", "target", "expected_labels"),
    [
        ([2**53 + 1], [1.5], [f"{2**53 + 1}: 1", "1.5: 1"]),  # would round as float
        ([221, 221], [1.5, 2.5], ["221: 2", "1.5: 1", "2.5: 1"]),
        ([("A",)], ["B"], ["('A',): 1", "B: 1"]),
    ],
)
@pytest.mark.parametrize("cols", [["src", "tgt"], ["count", "tgt"]])
def test_sankey_flow_data_keeps_column_values(
    source: list[Any], target: list[Any], expected_labels: list[str], cols: list[str]
) -> None:
    """Node values keep each column's own type instead of a shared dtype, also for
    a column named like value_counts' "count" result.
    """
    df_flows = pd.DataFrame(dict(zip(cols, (source, target), strict=True)))
    nodes = sankey_flow_data(df_flows, *cols)["nodes"]
    assert [node["label"] for node in nodes] == expected_labels
