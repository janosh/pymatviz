"""Sankey diagram for comparing distributions in two dataframe columns."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymatviz.widgets.sankey import SankeyWidget


if TYPE_CHECKING:
    from typing import Any, Literal

    import pandas as pd


def sankey(
    df: pd.DataFrame,
    source: str,
    target: str,
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
        source (str): Column whose values become the left-side nodes.
        target (str): Column whose values become the right-side nodes.
        labels_with_counts (bool | "percent", optional): Whether to append value
            counts (or percentages of rows with values in both columns) to node
            labels. Rows missing either value are left out. Defaults to True.
        **kwargs: SankeyWidget traits, e.g. ``orientation``, ``node_align``,
            ``link_color_mode``, ``value_format``, ``style``.

    Raises:
        ValueError: If no row has values in both columns.

    Returns:
        SankeyWidget: The Sankey diagram.
    """
    flow_data = sankey_flow_data(
        df, source, target, labels_with_counts=labels_with_counts
    )
    if not flow_data["links"]:
        raise ValueError(
            f"No rows with values in both {source=} and {target=} to draw flows from"
        )
    return SankeyWidget(flow_data, **kwargs)


def sankey_flow_data(
    df: pd.DataFrame,
    source: str,
    target: str,
    *,
    labels_with_counts: bool | Literal["percent"] = True,
) -> dict[str, list[dict[str, Any]]]:
    """Count (source, target) value pairs in two DataFrame columns as Sankey data.

    Only rows with values in both columns form links. Node counts (and percentage
    denominators) come from those same rows, so each node's label matches the total
    flow through it.

    Args:
        df (pd.DataFrame): DataFrame containing source and target columns.
        source (str): Source (left side) column name.
        target (str): Target (right side) column name.
        labels_with_counts (bool | "percent"): Whether to append counts to labels.
            If "percent", shows percentages of linked rows instead. Defaults to True.

    Returns:
        dict[str, list[dict]]: SankeyWidget data with keys
            - "nodes": ``{"id", "label", "metadata": {"column"}}`` per distinct value,
              source-column nodes first. A value in both columns gets one node per
              side, so no link is a self-loop.
            - "links": ``{"source", "target", "value"}`` with node ids and row counts.
    """
    cols = [source, target]

    # value_counts drops rows with NaN in either column. Its index levels keep each
    # column's dtype (to_numpy would coerce mixed int/float columns to one), and
    # unlike reset_index, they can't clash with a column named "count".
    flows = df[cols].value_counts()
    source, target = (flows.index.get_level_values(lvl).tolist() for lvl in (0, 1))
    value = flows.tolist()

    # Each column is one side of the diagram, so a value in both gets two nodes.
    # Dicts keep first-seen order and each column's own value types.
    side_counts: list[dict[Any, int]] = [{}, {}]
    for src_val, tgt_val, count in zip(source, target, value, strict=True):
        for counts, val in zip(side_counts, (src_val, tgt_val), strict=True):
            counts[val] = counts.get(val, 0) + count

    as_percent = labels_with_counts == "percent"
    n_linked = sum(value)
    nodes: list[dict[str, Any]] = []
    node_ids: list[dict[Any, int]] = [{}, {}]
    for col, counts, ids in zip(cols, side_counts, node_ids, strict=True):
        for val, count in counts.items():
            label = str(val)
            if as_percent:
                label += f": {count / n_linked:.1%}"
            elif labels_with_counts:
                label += f": {count:d}"
            ids[val] = len(nodes)
            nodes.append(
                {"id": len(nodes), "label": label, "metadata": {"column": col}}
            )

    links = [
        {"source": node_ids[0][src_val], "target": node_ids[1][tgt_val], "value": count}
        for src_val, tgt_val, count in zip(source, target, value, strict=True)
    ]
    return {"nodes": nodes, "links": links}
