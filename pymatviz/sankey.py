"""Sankey diagram for comparing distributions in two dataframe columns."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

from pymatviz.process_data import sankey_flow_data


if TYPE_CHECKING:
    from typing import Any, Literal

    import pandas as pd


def sankey_from_2_df_cols(
    df: pd.DataFrame,
    cols: list[str],
    *,
    labels_with_counts: bool | Literal["percent"] = True,
    annotate_columns: bool | dict[str, Any] = True,
    **kwargs: Any,
) -> go.Figure:
    """Plot two columns of a dataframe as a Plotly Sankey diagram.

    Args:
        df (pd.DataFrame): Pandas dataframe.
        cols (list[str]): 2-tuple of source and target column names. Source
            corresponds to left, target to right side of the diagram.
        labels_with_counts (bool, optional): Whether to append value counts to node
            labels. Defaults to True.
        annotate_columns (bool, dict[str, Any], optional): Whether to use the column
            names as annotations vertically centered on the left and right sides of
            the diagram. If a dict, passed as **kwargs to
            plotly.graph_objects.Figure.add_annotation. Defaults to True.
        **kwargs: Additional keyword arguments passed to plotly.graph_objects.Sankey.

    Raises:
        ValueError: If len(cols) != 2.

    Returns:
        Figure: Plotly figure containing the Sankey diagram.
    """
    flow_data = sankey_flow_data(df, cols, labels_with_counts=labels_with_counts)

    fig = go.Figure()
    fig.add_sankey(
        node=dict(
            pad=20,
            thickness=30,
            line=dict(color="black", width=0.5),
            label=flow_data["labels"],
        ),
        link=dict(
            source=flow_data["source_indices"],
            target=flow_data["target_indices"],
            value=flow_data["value"],
        ),
        **kwargs,
    )

    if annotate_columns:
        # Add column labels as annotations
        anno_kwargs = annotate_columns if isinstance(annotate_columns, dict) else {}
        xshift = anno_kwargs.pop("xshift", 35)
        fig.layout.margin = dict(l=xshift, r=xshift)
        for idx, col in enumerate(cols):
            anno_defaults = dict(
                y=0.5,
                xref="paper",
                yref="paper",
                xshift=(-1 if idx == 0 else 1) * xshift,
                textangle=-90,
                showarrow=False,
                font_size=20,
            )

            fig.add_annotation(
                x=idx, text=f"<b>{col}</b>", **anno_defaults | anno_kwargs
            )

    return fig
