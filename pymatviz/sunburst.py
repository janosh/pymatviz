from __future__ import annotations

from typing import Any, Literal, Sequence

import pandas as pd
import plotly.express as px
from plotly.graph_objs._figure import Figure
from pymatgen.symmetry.groups import SpaceGroup

from pymatviz.utils import get_crystal_sys


def spacegroup_sunburst(
    spacegroups: Sequence[int | str] | pd.DataFrame,
    spg_col: str = None,
    show_values: Literal["value", "percent", False] = False,
    **kwargs: Any,
) -> Figure:
    """Generate a sunburst plot with crystal systems as the inner ring for a list of
    international space group numbers.

    Hint: To hide very small labels, set a uniformtext minsize and mode='hide'.
    fig.update_layout(uniformtext=dict(minsize=9, mode="hide"))

    Args:
        spacegroups (list[int] | pd.DataFrame): A sequence of space group strings or
            numbers or a dataframe. If dataframe, be sure to specify spg_col.
        spg_col (str): The name of the column that holds the space group numbers.
            Defaults to None.
        show_values ("value" | "percent" | False): Whether to display values below each
            labels on the sunburst.

    Returns:
        Figure: The Plotly figure.
    """
    if isinstance(spacegroups, pd.DataFrame):
        if spg_col is None:
            raise ValueError(
                "if 1st arg is a DataFrame, spg_col must be specified as 2nd arg"
            )
        series = spacegroups[spg_col]
    else:
        series = pd.Series(spacegroups)

    df = pd.DataFrame(series.value_counts().reset_index())
    df.columns = ["spacegroup", "count"]

    try:
        df["crystal_sys"] = [get_crystal_sys(x) for x in df.spacegroup]
    except ValueError:  # column must be space group strings
        df["crystal_sys"] = [SpaceGroup(x).crystal_system for x in df.spacegroup]

    if "color_discrete_sequence" not in kwargs:
        kwargs["color_discrete_sequence"] = px.colors.qualitative.G10

    fig = px.sunburst(df, path=["crystal_sys", "spacegroup"], values="count", **kwargs)

    if show_values == "percent":
        fig.data[0].textinfo = "label+percent entry"
    elif show_values == "value":
        fig.data[0].textinfo = "label+value"
    elif show_values is not False:
        raise ValueError(f"Invalid {show_values=}")

    fig.update_layout(
        margin=dict(l=10, r=10, b=10, pad=10),
        paper_bgcolor="rgba(0, 0, 0, 0)",
    )

    return fig
