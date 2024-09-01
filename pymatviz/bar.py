"""Bar plots."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import transforms
from matplotlib.ticker import FixedLocator
from pymatgen.core import Structure
from pymatgen.symmetry.groups import SpaceGroup

from pymatviz.enums import Key
from pymatviz.utils import PLOTLY, Backend, crystal_sys_from_spg_num, si_fmt_int


def spacegroup_bar(
    data: Sequence[int | str | Structure] | pd.Series,
    *,
    show_counts: bool = True,
    xticks: Literal["all", "crys_sys_edges"] | int = 20,
    show_empty_bins: bool = False,
    ax: plt.Axes | None = None,
    backend: Backend = PLOTLY,
    text_kwargs: dict[str, Any] | None = None,
    log: bool = False,
    **kwargs: Any,
) -> plt.Axes | go.Figure:
    """Plot a histogram of spacegroups shaded by crystal system.

    Args:
        data (list[int | str | Structure] | pd.Series): Space group strings or numbers
            (from 1 - 230) or pymatgen structures.
        show_counts (bool, optional): Whether to count the number of items
            in each crystal system. Defaults to True.
        xticks ("all" | "crys_sys_edges" | int, optional): Where to add x-ticks. An
            integer will add ticks below that number of tallest bars. Defaults to 20.
            "all" will show below all bars, "crys_sys_edges" only at the edge from one
            crystal system to another.
        show_empty_bins (bool, optional): Whether to include a 0-height bar for missing
            space groups missing from the data. Currently only implemented for numbers,
            not symbols. Defaults to False.
        ax (Axes, optional): matplotlib Axes on which to plot. Defaults to None.
        backend ("matplotlib" | "plotly", optional): Which backend to use for plotting.
            Defaults to "plotly".
        text_kwargs (dict, optional): Keyword arguments passed to
            matplotlib.Axes.text(). Defaults to None. Has no effect if backend is
            "plotly".
        log (bool, optional): Whether to log scale the y-axis. Defaults to False.
        kwargs: Keywords passed to pd.Series.plot.bar() or plotly.express.bar().

    Returns:
        plt.Axes | go.Figure: matplotlib Axes or plotly Figure depending on backend.
    """
    if isinstance(next(iter(data)), Structure):
        # if 1st sequence item is structure, assume all are
        data = cast(Sequence[Structure], data)
        series = pd.Series(struct.get_space_group_info()[1] for struct in data)
    else:
        series = pd.Series(data)

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

    if df_data.index.inferred_type == "integer":  # assume index is space group numbers
        df_data = df_data.reindex(range(1, 231), fill_value=0).sort_index()
        if not show_empty_bins:
            df_data = df_data.query(f"{count_col} > 0")
        df_data[Key.crystal_system] = [
            crystal_sys_from_spg_num(x) for x in df_data.index
        ]

        xlim = (df_data.index.min() - 0.5, df_data.index.max() + 0.5)
        x_label = "International Spacegroup Number"

    else:  # assume index is space group symbols
        # TODO: figure out how to implement show_empty_bins for space group symbols
        # if show_empty_bins:
        #     idx = [SpaceGroup.from_int_number(x).symbol for x in range(1, 231)]
        #     df = df.reindex(idx, fill_value=0)
        df_data = df_data[df_data[count_col] > 0]
        df_data[Key.crystal_system] = [
            SpaceGroup(x).crystal_system for x in df_data.index
        ]

        # sort df by crystal system going from smallest to largest spacegroup numbers
        # e.g. triclinic (1-2) comes first, cubic (195-230) last
        sys_order = dict(
            zip(crystal_sys_colors, range(len(crystal_sys_colors)), strict=True)
        )
        df_data = df_data.loc[
            df_data[Key.crystal_system].map(sys_order).sort_values().index
        ]

        x_label = "International Spacegroup Symbol"

    # count rows per crystal system
    crys_sys_counts = df_data.groupby(Key.crystal_system).sum(count_col)
    crys_sys_counts["width"] = df_data.value_counts(Key.crystal_system)
    crys_sys_counts["color"] = pd.Series(crystal_sys_colors)

    # sort by key order in dict crys_colors
    crys_sys_counts = crys_sys_counts.loc[
        [x for x in crystal_sys_colors if x in crys_sys_counts.index]
    ]
    xlim = (0, len(df_data) - 1)

    fig_title = f"{count_col} per crystal system" if show_counts else None
    if backend == PLOTLY:
        df_plot = df_data if show_empty_bins else df_data.reset_index()

        fig = px.bar(
            df_plot,
            x=df_plot.index,
            y=count_col,
            color=df_data[Key.crystal_system],
            color_discrete_map=crystal_sys_colors,
            **kwargs,
        )
        # add vertical lines between crystal systems and fill area with color
        x0 = x1 = 0
        for idx, (crys_sys, count, width, color) in enumerate(
            crys_sys_counts.itertuples()
        ):
            prev_width = x1 - x0 if idx > 0 else 0
            x1 = x0 + width
            anno = dict(
                text=crys_sys,
                font=dict(size=14),
                x=(x0 + x1) / 2,
                textangle=90,
                xanchor="center",
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
                    font=dict(size=12),
                    yref="paper",
                    yanchor="bottom",
                )
            x0 += width

        fig.layout.showlegend = False
        fig.layout.title = dict(text=fig_title, x=0.5)
        fig.layout.xaxis.update(showgrid=False, title=x_label, range=xlim)
        count_max = df_data[count_col].max()
        y_max = np.log10(count_max * 1.05) if log else count_max * 1.05
        fig.layout.yaxis.update(range=(0, y_max), type="log" if log else None)
        fig.layout.margin = dict(l=0, r=0, t=40, b=0)

        if isinstance(xticks, int):
            # get x_locs of n=xticks tallest bars
            x_indices = df_data.reset_index()[count_col].nlargest(xticks).index
            tick_text = df_data.iloc[x_indices].index
        elif xticks == "crys_sys_edges":
            # add x_locs of n=xticks tallest bars
            x_indices = crys_sys_counts.width.cumsum()
            tick_text = df_data.index[x_indices - 1]
        elif xticks == "all":
            x_indices = df_data.reset_index().index
            tick_text = df_data.index
        else:
            raise ValueError(
                f"Invalid {xticks=}, must be int, 'all' or 'crys_sys_edges'"
            )

        fig.update_xaxes(tickvals=x_indices, ticktext=tick_text, tickangle=90)

        return fig

    ax = ax or plt.gca()
    # keep this above df.plot.bar()! order matters
    ax.set(ylabel=count_col, xlim=xlim)

    defaults = dict(width=0.9)  # set default histogram bar width
    df_data[count_col].plot.bar(figsize=[16, 4], ax=ax, **defaults | kwargs)

    ax.set_title(fig_title, fontdict={"fontsize": 18}, pad=30)
    ax.set(xlabel=x_label)

    # https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/fill_between_demo
    transform = transforms.blended_transform_factory(ax.transData, ax.transAxes)

    # add crystal system labels and dividers
    x0 = 0
    for crys_sys, count, width, color in crys_sys_counts.itertuples():
        x1 = x0 + width

        for patch in ax.patches[0 if x0 == 1 else x0 : x1 + 1]:
            patch.set_facecolor(color)

        text_kwds = dict(transform=transform, horizontalalignment="center") | (
            text_kwargs or {}
        )
        crys_sys_anno_kwds = dict(
            rotation=90, va="top", ha="right", fontdict={"fontsize": 14}
        )
        ax.text(*[(x0 + x1) / 2, 0.95], crys_sys, **crys_sys_anno_kwds | text_kwds)
        if show_counts:
            ax.text(
                *[(x0 + x1) / 2, 1.02],
                f"{si_fmt_int(count)} ({count / len(data):.0%})",
                **dict(fontdict={"fontsize": 12}) | text_kwds,
            )

        ax.fill_between(
            [x0 - 0.5, x1 - 0.5],
            *[0, 1],
            facecolor=color,
            alpha=0.1,
            transform=transform,
            edgecolor="black",
        )
        x0 += width

    ax.yaxis.grid(visible=True)
    ax.xaxis.grid(visible=False)
    ax.set_ylim(0, None)
    if log:
        ax.set_yscale("log")

    if xticks == "crys_sys_edges" or isinstance(xticks, int):
        if isinstance(xticks, int):
            # get x_locs of n=xticks tallest bars
            x_indices = df_data.reset_index().sort_values(count_col).tail(xticks).index
        else:
            # add x_locs of n=xticks tallest bars
            x_indices = crys_sys_counts.width.cumsum()

        major_loc = FixedLocator(x_indices)

        ax.xaxis.set_major_locator(major_loc)
    plt.xticks(rotation=90)
    return ax
