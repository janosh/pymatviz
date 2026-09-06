"""Bar plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from pymatviz.enums import Key
from pymatviz.process_data import normalize_spacegroups
from pymatviz.utils import si_fmt_int, spg_to_crystal_sys


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Literal

    from pymatgen.core import Structure


def spacegroup_bar(
    data: Sequence[int | float | str | Structure] | pd.Series,
    *,
    show_counts: bool = True,
    xticks: Literal["all", "crys_sys_edges"] | int = 20,
    show_empty_bins: bool = False,
    log: bool = False,
    **kwargs: Any,
) -> go.Figure:
    """Plot a histogram of spacegroups shaded by crystal system using Plotly.

    Args:
        data (Sequence[int | float | str | Structure] | pd.Series): Space group symbols,
            integral numbers (1-230), or pymatgen structures. All inputs use numerically
            ordered space group numbers as x-axis labels.
        show_counts (bool, optional): Whether to count the number of items
            in each crystal system. Defaults to True.
        xticks ("all" | "crys_sys_edges" | int, optional): Where to add x-ticks. An
            integer will add ticks below that number of tallest bars. Defaults to 20.
            "all" will show below all bars, "crys_sys_edges" only at the edge from one
            crystal system to another.
        show_empty_bins (bool, optional): Whether to include a zero-height bar for every
            missing space group, for any input format. Defaults to False.
        log (bool, optional): Whether to log scale the y-axis. Defaults to False.
        **kwargs: Keywords passed to plotly.express.bar().

    Returns:
        go.Figure: Plotly Figure object.

    Raises:
        ValueError: If data is empty.
    """
    if len(data) == 0:
        raise ValueError("spacegroup_bar requires non-empty data")

    series = normalize_spacegroups(data)

    count_col = "Counts"
    df_data = series.value_counts(sort=False).to_frame(name=count_col)

    crystal_sys_colors = {
        "triclinic": "red",
        "monoclinic": "teal",
        "orthorhombic": "blue",
        "tetragonal": "green",
        "trigonal": "orange",
        "hexagonal": "purple",
        "cubic": "darkred",
    }

    df_data = df_data.reindex(range(1, 231), fill_value=0)
    if not show_empty_bins:
        df_data = df_data[df_data[count_col] > 0]
    df_data[Key.crystal_system] = df_data.index.map(spg_to_crystal_sys)

    # count rows per crystal system
    crys_sys_counts = df_data.groupby(Key.crystal_system)[[count_col]].sum()
    crys_sys_counts["width"] = df_data.value_counts(Key.crystal_system)
    crys_sys_counts["color"] = pd.Series(crystal_sys_colors)

    # sort by key order in dict crys_colors
    crys_sys_counts = crys_sys_counts.loc[
        [x for x in crystal_sys_colors if x in crys_sys_counts.index]
    ]

    fig_title = f"{count_col} per crystal system" if show_counts else None
    x_values = df_data.index if show_empty_bins else pd.RangeIndex(len(df_data))
    x_range = (x_values[0] - 0.5, x_values[-1] + 0.5)

    fig = px.bar(
        df_data,
        x=x_values,
        y=count_col,
        color=df_data[Key.crystal_system],
        color_discrete_map=crystal_sys_colors,
        **kwargs,
    )
    # add vertical lines between crystal systems and fill area with color
    x0 = x1 = x_range[0]
    for idx, (crys_sys, count, width, color) in enumerate(crys_sys_counts.itertuples()):
        prev_width = x1 - x0 if idx > 0 else 0
        x1 = x0 + width
        anno = dict(
            text=crys_sys, font_size=14, x=(x0 + x1) / 2, textangle=90, xanchor="center"
        )
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=color,
            opacity=0.15,
            line=dict(width=1),
            annotation=anno,
        )
        # add percent annotation
        if show_counts:
            fig.add_annotation(
                text=f"{si_fmt_int(count)} ({count / len(data):.0%})",
                x=(x0 + x1) / 2,
                y=1,
                # shift count up if bar is so narrow it overlaps with neighbors
                yshift=16 if (width + prev_width < 15 and idx % 2 == 1) else 0,
                showarrow=False,
                font_size=12,
                yref="paper",
                yanchor="bottom",
            )
        x0 += width

    fig.layout.showlegend = False
    fig.layout.title = dict(text=fig_title, x=0.5)
    fig.layout.xaxis.update(
        showgrid=False, title="International Spacegroup Number", range=x_range
    )
    count_max = df_data[count_col].max()
    y_max = np.log10(count_max * 1.05) if log else count_max * 1.05
    fig.layout.yaxis.update(range=(0, y_max), type="log" if log else None)
    fig.layout.margin = dict(l=0, r=0, t=40, b=0)

    if isinstance(xticks, int):
        x_indices = df_data[count_col].reset_index(drop=True).nlargest(xticks).index
    elif xticks == "crys_sys_edges":
        x_indices = crys_sys_counts.width.cumsum() - 1
    elif xticks == "all":
        x_indices = range(len(df_data))
    else:
        raise ValueError(f"Invalid {xticks=}, must be int, 'all' or 'crys_sys_edges'")

    fig.update_xaxes(
        tickvals=x_values[x_indices], ticktext=df_data.index[x_indices], tickangle=90
    )

    return fig
