from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import plotly.graph_objects as go


if TYPE_CHECKING:
    import pandas as pd


def sankey_from_2_df_cols(
    df: pd.DataFrame,
    cols: list[str],
    labels_with_counts: bool | Literal["percent"] = True,
    **kwargs: Any,
) -> go.Figure:
    """Plot two columns of a dataframe as a Plotly Sankey diagram.

    Args:
        df (pd.DataFrame): Pandas dataframe.
        cols (list[str]): 2-tuple of source and target column names. Source
            corresponds to left, target to right side of the diagram.
        labels_with_counts (bool, optional): Whether to append value counts to node
            labels. Defaults to True.
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

    sankey = go.Sankey(
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

    return go.Figure(data=[sankey])
