from __future__ import annotations

from typing import Any, Literal, Sequence

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure

from pymatviz.utils import get_crystal_sys


def spacegroup_sunburst(
    spacegroups: Sequence[int] | pd.DataFrame,
    sgp_col: str = None,
    show_values: Literal["value", "percent", False] = False,
    **kwargs: Any,
) -> Figure:
    """Generate a sunburst plot with crystal systems as the inner ring for a list of
    international space group numbers.

    Args:
        spacegroups (list[int] | pd.DataFrame): A sequence of space group numbers or a
            dataframe. If dataframe, be sure to specify sgp_col.
        sgp_col (str): The name of the column that holds the space group numbers.
            Defaults to None.
        show_values ("value" | "percent" | False): Whether to display values below each
            labels on the sunburst.

    Returns:
        Figure: The Plotly figure.
    """
    if isinstance(spacegroups, pd.DataFrame):
        assert (
            sgp_col is not None
        ), "if 1st arg is a DataFrame, sgp_col must be specified"
        series = spacegroups[sgp_col]
    else:
        series = pd.Series(spacegroups)

    df = pd.DataFrame({"spacegroup": range(230)})
    df["cryst_sys"] = [get_crystal_sys(spg) for spg in range(1, 231)]

    df["values"] = series.value_counts().reindex(range(230), fill_value=0)

    if "color_discrete_sequence" not in kwargs:
        kwargs["color_discrete_sequence"] = px.colors.qualitative.G10

    fig = px.sunburst(df, path=["cryst_sys", "spacegroup"], values="values", **kwargs)

    if show_values == "percent":
        fig.data[0].textinfo = "label+percent entry"
    elif show_values == "value":
        fig.data[0].textinfo = "label+value"
    elif show_values is not False:
        raise ValueError(f"Invalid {show_values=}")

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10, pad=10),
        paper_bgcolor="rgba(0, 0, 0, 0)",
    )

    return fig
