"""Sankey diagram for comparing distributions in two dataframe columns."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go


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
    if len(cols) != 2:
        raise ValueError(
            f"{cols=} should specify exactly two columns: (source_col, target_col)"
        )

    source, target, value = (
        df[list(cols)].value_counts().reset_index().to_numpy().T.tolist()
    )

    if labels_with_counts:
        as_percent = labels_with_counts == "percent"
        source_counts = df[cols[0]].value_counts(normalize=as_percent).to_dict()
        target_counts = df[cols[1]].value_counts(normalize=as_percent).to_dict()
        fmt = ".1%" if as_percent else "d"
        label = [f"{x}: {source_counts[x]:{fmt}}" for x in source] + [
            f"{x}: {target_counts[x]:{fmt}}" for x in target
        ]
    else:
        label = source + target

    fig = go.Figure()
    fig.add_sankey(
        node=dict(
            pad=20,
            thickness=30,
            line=dict(color="black", width=0.5),
            label=label,
        ),
        link=dict(
            # indices in source, target, value correspond to labels
            source=[source.index(x) for x in source],
            target=[len(source) + target.index(x) for x in target],
            value=value,
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
