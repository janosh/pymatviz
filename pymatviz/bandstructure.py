from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go


if TYPE_CHECKING:
    from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine


def plot_band_structure(
    bs: BandStructureSymmLine, line_color: str = "blue", **kwargs: Any
) -> go.Figure:
    """Plot  band structure using Plotly with improvements.

    Args:
        bs: BandStructureSymmLine object for band structure data.
        line_color: Color for the lines in the plot.
        **kwargs: Passed to Plotly's Figure.add_scatter method.

    Returns:
        A Plotly figure object.
    """
    fig = go.Figure()

    # Extracting band and distance data
    for band in range(bs.nb_bands):
        fig.add_scatter(
            x=bs.distance,
            y=bs.bands[band],
            mode="lines",
            line=dict(color=line_color),
            showlegend=False,
            **kwargs,
        )

    # Adding vertical lines for high-symmetry points and setting the x-axis range
    high_symm_distances = [
        bs.distance[bs.branches[i]["start_index"]] for i in range(len(bs.branches))
    ]
    for dist in high_symm_distances:
        fig.add_vline(x=dist, line=dict(color="black", width=1))

    # Adjust the x-axis and y-axis range to include zero and to start from the first
    # high-symmetry point
    x_axis_range = [min(high_symm_distances), max(bs.distance)]
    y_axis_range = [np.amin(bs.bands), np.amax(bs.bands)]

    # set some default values for the figure
    fig.layout.title = "Band Structure"
    fig.layout.xaxis.title = "Wave Vector"
    fig.layout.yaxis.title = "Frequency (THz)"
    fig.layout.xaxis = dict(
        range=x_axis_range,
        tickmode="array",
        tickvals=high_symm_distances,
        ticktext=[
            bs.qpoints[bs.branches[i]["start_index"]].label
            for i in range(len(bs.branches))
        ],
        # show plot area border line on top
        mirror=True,
    )
    fig.layout.yaxis = dict(range=y_axis_range, mirror=True)
    fig.layout.margin = dict(t=40)  # Adjust top margin to ensure title fits

    return fig
