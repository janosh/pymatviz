from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
import plotly.express as px
from pymatgen.core import Structure
from pymatgen.symmetry.groups import SpaceGroup

from pymatviz.utils import get_crystal_sys


if TYPE_CHECKING:
    from collections.abc import Sequence

    import plotly.graph_objects as go


def spacegroup_sunburst(
    data: Sequence[int | str] | pd.Series,
    show_counts: Literal["value", "percent", False] = False,
    **kwargs: Any,
) -> go.Figure:
    """Generate a sunburst plot with crystal systems as the inner ring for a list of
    international space group numbers.

    Hint: To hide very small labels, set a uniformtext minsize and mode='hide'.
    fig.update_layout(uniformtext=dict(minsize=9, mode="hide"))

    Args:
        data (list[int] | pd.Series): A sequence (list, tuple, pd.Series) of
            space group strings or numbers (from 1 - 230) or pymatgen structures.
        show_counts ("value" | "percent" | False): Whether to display values below each
            labels on the sunburst.
        color_discrete_sequence (list[str]): A list of 7 colors, one for each crystal
            system. Defaults to plotly.express.colors.qualitative.G10.
        **kwargs: Additional keyword arguments passed to plotly.express.sunburst.

    Returns:
        Figure: The Plotly figure.
    """
    if isinstance(next(iter(data)), Structure):
        # if 1st sequence item is structure, assume all are
        series = pd.Series(
            struct.get_space_group_info()[1]  # type: ignore[union-attr]
            for struct in data
        )
    else:
        series = pd.Series(data)

    df = pd.DataFrame(series.value_counts().reset_index())
    df.columns = ["spacegroup", "count"]

    try:  # assume column contains integers as space group numbers
        df["crystal_sys"] = [get_crystal_sys(x) for x in df.spacegroup]
    except ValueError:  # column must be strings of space group symbols
        df["crystal_sys"] = [SpaceGroup(x).crystal_system for x in df.spacegroup]

    kwargs.setdefault("color_discrete_sequence", px.colors.qualitative.G10)

    fig = px.sunburst(df, path=["crystal_sys", "spacegroup"], values="count", **kwargs)

    if show_counts == "percent":
        fig.data[0].textinfo = "label+percent entry"
    elif show_counts == "value":
        fig.data[0].textinfo = "label+value"
    elif show_counts is not False:
        raise ValueError(f"Invalid {show_counts=}")

    fig.update_layout(
        margin=dict(l=10, r=10, b=10, pad=10),
        paper_bgcolor="rgba(0, 0, 0, 0)",
    )

    return fig
