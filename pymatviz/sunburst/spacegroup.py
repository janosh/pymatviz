"""Sunburst plot of crystal systems."""

from collections.abc import Sequence
from typing import Any, Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import pymatviz as pmv
from pymatviz.enums import Key
from pymatviz.sunburst.helpers import _limit_slices
from pymatviz.typing import ShowCounts


def spacegroup_sunburst(
    data: Sequence[int | str] | pd.Series,
    *,
    show_counts: ShowCounts = "value",
    max_slices: int | None = None,
    max_slices_mode: Literal["other", "drop"] = "other",
    **kwargs: Any,
) -> go.Figure:
    """Generate a sunburst plot with crystal systems as the inner ring for a list of
    international space group numbers.

    Hint: To hide very small labels, set a uniformtext minsize and mode='hide'.
    fig.update_layout(uniformtext=dict(minsize=9, mode="hide"))

    Args:
        data (list[int] | pd.Series): A sequence (list, tuple, pd.Series) of
            space group strings or numbers (from 1 - 230) or pymatgen structures.
        show_counts ("value" | "percent" | "value+percent" | False): Whether to display
            values below each labels on the sunburst.
        max_slices (int | None): Maximum number of space groups to show
            for each crystal system. If None (default), all space groups are shown. If
            positive integer, only the top N space groups by count are shown.
        max_slices_mode ("other" | "drop"): How to handle spacegroups beyond max_slices:
            - "other": Combine remaining space groups into an "Other" slice (default)
            - "drop": Discard remaining space groups entirely
        color_discrete_sequence (list[str]): A list of 7 colors, one for each crystal
            system. Defaults to plotly.express.colors.qualitative.G10.
        **kwargs: Additional keyword arguments passed to plotly.express.sunburst.

    Returns:
        Figure: The Plotly figure.
    """
    if type(next(iter(data))).__qualname__ in ("Structure", "Atoms"):
        # if 1st sequence item is pymatgen structure or ASE Atoms, assume all are
        from moyopy import MoyoDataset
        from moyopy.interface import MoyoAdapter

        series = pd.Series(
            MoyoDataset(MoyoAdapter.from_py_obj(struct)).number  # type: ignore[arg-type]
            for struct in data
        )
    else:
        series = pd.Series(data)

    df_spg_counts = pd.DataFrame(series.value_counts().reset_index())
    df_spg_counts.columns = [Key.spg_num, "count"]

    try:  # assume column contains integers as space group numbers
        df_spg_counts[Key.crystal_system] = df_spg_counts[Key.spg_num].map(
            pmv.utils.spg_to_crystal_sys
        )

    except (ValueError, TypeError):  # assume column is strings of space group symbols
        df_spg_counts[Key.crystal_system] = df_spg_counts[Key.spg_num].map(
            pmv.utils.spg_num_to_from_symbol
        )

    # Limit the number of space groups per crystal system if requested
    df_spg_counts = _limit_slices(
        df_spg_counts,
        group_col=Key.crystal_system,
        count_col="count",
        max_slices=max_slices,
        max_slices_mode=max_slices_mode,
        other_label="Other",
        child_col_for_other_label=Key.spg_num,
    )

    sunburst_defaults = dict(color_discrete_sequence=px.colors.qualitative.G10)

    fig = px.sunburst(
        df_spg_counts,
        path=[Key.crystal_system, Key.spg_num],
        values="count",
        **sunburst_defaults | kwargs,
    )

    if show_counts == "percent":
        fig.data[0].textinfo = "label+percent entry"
    elif show_counts == "value":
        fig.data[0].textinfo = "label+value"
        fig.data[0].texttemplate = "%{label}<br>N=%{value:.2f}"
    elif show_counts == "value+percent":
        fig.data[0].textinfo = "label+value+percent entry"
        fig.data[0].texttemplate = "%{label}<br>N=%{value:.2f}<br>%{percentEntry}"
    elif show_counts is not False:
        raise ValueError(
            f"Invalid {show_counts=}, must be 'value', 'percent', 'value+percent', "
            "or False"
        )

    fig.layout.margin = dict(l=10, r=10, b=10, pad=10)
    fig.layout.paper_bgcolor = "rgba(0, 0, 0, 0)"

    return fig
