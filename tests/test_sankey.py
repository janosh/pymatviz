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


@pytest.mark.parametrize(
    ("source", "target", "kwargs", "err", "match"),
    [
        ("nonexistent", "b", {}, KeyError, "nonexistent"),
        ("a", "c", {}, ValueError, "No rows with values in both"),
        ("a", "a", {}, ValueError, "source and target must be different columns"),
        (
            "a",
            "b",
            {"labels_with_counts": "pct"},
            ValueError,
            "labels_with_counts='pct' must be True, False or 'percent'",
        ),
    ],
)
def test_sankey_invalid_inputs(
    source: str, target: str, kwargs: dict[str, Any], err: type[Exception], match: str
) -> None:
    """Missing or identical columns, columns without shared rows and invalid label
    modes raise.
    """
    df_in = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [None, None]})
    with pytest.raises(err, match=match):
        pmv.sankey(df_in, source, target, **kwargs)


SHARED = {"src": ["A", "A", "A", "B"], "tgt": ["C", "C", "A", "C"]}
WITH_NAN = {"src": ["a", "a", "a", "b", None], "tgt": ["x", "x", None, "y", "y"]}


@pytest.mark.parametrize(
    ("data", "labels_with_counts", "expected_labels", "expected_links"),
    [
        (
            SHARED,
            True,
            ["A: 3", "B: 1", "C: 3", "A: 1"],
            [(0, 2, 2), (0, 3, 1), (1, 2, 1)],
        ),
        (SHARED, "percent", ["A: 75.0%", "B: 25.0%", "C: 75.0%", "A: 25.0%"], None),
        (SHARED, False, ["A", "B", "C", "A"], None),
        # rows missing either value count toward neither labels nor links
        (WITH_NAN, True, ["a: 2", "b: 1", "x: 2", "y: 1"], [(0, 2, 2), (1, 3, 1)]),
        (WITH_NAN, "percent", ["a: 66.7%", "b: 33.3%", "x: 66.7%", "y: 33.3%"], None),
    ],
)
def test_sankey_flow_data(
    data: dict[str, list[Any]],
    labels_with_counts: bool | Literal["percent"],
    expected_labels: list[str],
    expected_links: list[tuple[int, int, int]] | None,
) -> None:
    """A value in both columns gets one node per side, so no link is a self-loop."""
    flow = sankey_flow_data(
        pd.DataFrame(data), "src", "tgt", labels_with_counts=labels_with_counts
    )
    assert [node["label"] for node in flow["nodes"]] == expected_labels
    node_columns = [node["metadata"]["column"] for node in flow["nodes"]]
    assert node_columns == ["src"] * 2 + ["tgt"] * 2
    if expected_links:
        links = [(lnk["source"], lnk["target"], lnk["value"]) for lnk in flow["links"]]
        assert links == expected_links


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


def test_sankey_flow_data_skips_unused_categories() -> None:
    """Categorical columns link only observed pairs, so all-NaN rows raise."""
    df_cat = pd.DataFrame(
        {
            "src": pd.Categorical(["a", None], categories=["a", "b"]),
            "tgt": pd.Categorical(["x", "x"], categories=["x", "y"]),
        }
    )
    flow = sankey_flow_data(df_cat, "src", "tgt", labels_with_counts=False)
    assert [node["label"] for node in flow["nodes"]] == ["a", "x"]
    assert flow["links"] == [{"source": 0, "target": 1, "value": 1}]
    with pytest.raises(ValueError, match="No rows with values in both"):
        pmv.sankey(df_cat.iloc[1:], "src", "tgt")
