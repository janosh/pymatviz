"""Sankey diagram for comparing distributions in two dataframe columns."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymatviz.process_data import sankey_flow_data
from pymatviz.widgets.sankey import SankeyWidget


if TYPE_CHECKING:
    from typing import Any, Literal

    import pandas as pd


def sankey_from_2_df_cols(
    df: pd.DataFrame,
    cols: list[str],
    *,
    labels_with_counts: bool | Literal["percent"] = True,
    **kwargs: Any,
) -> SankeyWidget:
    """Plot the co-occurrence of values in two dataframe columns as a Sankey diagram.

    Each distinct value in the source column becomes a node on the left, each in
    the target column a node on the right. Links carry the number of rows holding
    each (source, target) pair. A value present in both columns gets one node per
    side. Node metadata records the column each node comes from.

    Args:
        df (pd.DataFrame): Pandas dataframe.
        cols (list[str]): 2-tuple of source and target column names. Source
            corresponds to left, target to right side of the diagram.
        labels_with_counts (bool | "percent", optional): Whether to append value
            counts (or percentages of rows with values in both columns) to node
            labels. Rows missing either value are left out. Defaults to True.
        **kwargs: SankeyWidget traits, e.g. ``orientation``, ``node_align``,
            ``link_color_mode``, ``value_format``, ``style``.

    Raises:
        ValueError: If len(cols) != 2 or no row has values in both columns.

    Returns:
        SankeyWidget: The Sankey diagram.
    """
    flow_data = sankey_flow_data(df, cols, labels_with_counts=labels_with_counts)
    if not flow_data["links"]:
        raise ValueError(f"No rows with values in both {cols=} to draw flows from")
    return SankeyWidget(flow_data, **kwargs)
