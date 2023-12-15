from __future__ import annotations

from typing import Any, Union

import plotly.express as px
import plotly.graph_objects as go
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.util.string import htmlify


AnyBandStructure = Union[BandStructureSymmLine, PhononBandStructureSymmLine]


def pretty_sym_point(symbol: str) -> str:
    """Convert a symbol to a pretty-printed version."""
    # htmlify maps S0 -> S<sub>0</sub> but leaves S_0 as is so we remove underscores
    return (
        htmlify(symbol.replace("_", ""))
        .replace("GAMMA", "Γ")
        .replace("DELTA", "Δ")
        .replace("SIGMA", "Σ")
    )


def get_ticks(bs: PhononBandStructureSymmLine) -> tuple[list[float], list[str]]:
    """Get all ticks and labels for a band structure plot.

    Returns:
        tuple[list[float], list[str]]: Ticks and labels for the x-axis of a band
            structure plot.
    """
    ticks_x_pos = []
    tick_labels: list[str] = []
    prev_label = bs.qpoints[0].label
    prev_branch = bs.branches[0]["name"]

    for idx, point in enumerate(bs.qpoints):
        if point.label is None:
            continue
        ticks_x_pos += [bs.distance[idx]]

        branches = (
            branch["name"]
            for branch in bs.branches
            if branch["start_index"] <= idx <= branch["end_index"]
        )
        this_branch = next(branches, None)

        if point.label != prev_label and prev_branch != this_branch:
            tick_labels.pop()
            ticks_x_pos.pop()
            tick_labels += [f"{prev_label or ''}|{point.label}"]
        else:
            tick_labels += [point.label]

        prev_label = point.label
        prev_branch = this_branch

    tick_labels = list(map(pretty_sym_point, tick_labels))
    return ticks_x_pos, tick_labels


def plot_band_structure(
    band_structs: PhononBandStructureSymmLine | dict[str, PhononBandStructureSymmLine],
    line_kwds: dict[str, Any] | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Plot single or multiple pymatgen band structures using Plotly, focusing on the
    minimum set of overlapping branches.

    Warning: Only tested with phonon band structures so far but plan is to extend to
    electronic band structures.

    Args:
        band_structs (PhononBandStructureSymmLine | dict[str, PhononBandStructure]):
            Single BandStructureSymmLine or PhononBandStructureSymmLine object or a dict
            with labels mapped to multiple such objects.
        line_kwds (dict[str, Any]): Passed to Plotly's Figure.add_scatter method.

        **kwargs: Passed to Plotly's Figure.add_scatter method.

    Returns:
        go.Figure: Plotly figure object.
    """
    fig = go.Figure()
    line_kwds = line_kwds or {}

    if not isinstance(band_structs, dict):  # normalize input to dictionary
        band_structs = {"": band_structs}

    # find common branches by normalized branch names
    common_branches: set[str] = set()
    for bs in band_structs.values():
        branches = {pretty_sym_point(branch["name"]) for branch in bs.branches}
        if not common_branches:
            common_branches = branches
        else:
            common_branches = common_branches & branches

    if not common_branches:
        raise ValueError("No common branches found among the band structures.")

    # plotting only the common branches for each band structure
    first_bs = None
    colors = px.colors.qualitative.Plotly
    line_styles = ("solid", "dot", "dash", "longdash", "dashdot", "longdashdot")

    for bs_idx, (label, bs) in enumerate(band_structs.items()):
        color = colors[bs_idx % len(colors)]
        line_defaults = dict(color=color, width=2)
        line_style = line_styles[bs_idx % len(line_styles)]
        # 1st bands determine x-axis scale (there are usually slight scale differences
        # between bands)
        first_bs = first_bs or bs
        for branch_idx, branch in enumerate(bs.branches):
            start_idx = branch["start_index"]
            end_idx = branch["end_index"] + 1  # Include the end point
            # using the same first_bs x-axis for all band structures to avoid band
            # shifting
            distances = first_bs.distance[start_idx:end_idx]
            for band in range(bs.nb_bands):
                frequencies = bs.bands[band][start_idx:end_idx]
                # group traces for toggling and set legend name only for 1st band
                fig.add_scatter(
                    x=distances,
                    y=frequencies,
                    mode="lines",
                    line=line_defaults | line_kwds,
                    legendgroup=label,
                    name=label,
                    showlegend=branch_idx == band == 0,
                    line_dash=line_style,
                    **kwargs,
                )

    # add x-axis labels and vertical lines for common high-symmetry points
    first_bs = next(iter(band_structs.values()))
    x_ticks, x_labels = get_ticks(first_bs)
    fig.layout.xaxis.update(tickvals=x_ticks, ticktext=x_labels, tickangle=0)

    # remove 0 to avoid duplicate vertical line, looks like graphical artifact
    for x_pos in {*x_ticks} - {0}:
        fig.add_vline(x=x_pos, line=dict(color="black", width=1))

    fig.layout.xaxis.title = "Wave Vector"
    fig.layout.yaxis.title = "Frequency (THz)"
    fig.layout.margin = dict(t=5, b=5, l=5, r=5)

    y_min, y_max = (
        min(min(bs.bands.ravel()) for bs in band_structs.values()),
        max(max(bs.bands.ravel()) for bs in band_structs.values()),
    )
    if y_min >= -0.01:  # only set y_min=0 if no imaginary frequencies
        fig.layout.yaxis.range = (0, 1.05 * y_max)
    else:
        # no need for y=0 line if y_min = 0
        fig.add_hline(y=0, line=dict(color="black", width=1))

    axes_kwds = dict(linecolor="black", gridcolor="lightgray")
    fig.layout.xaxis.update(**axes_kwds)
    fig.layout.yaxis.update(**axes_kwds)

    # move legend to best position
    fig.layout.legend.update(
        x=0.005,
        y=0.99,
        orientation="h",
        yanchor="top",
        bgcolor="rgba(255, 255, 255, 0.6)",
        bordercolor="rgba(0, 0, 0, 0.2)",
        borderwidth=1,
    )

    # scale font size with figure size
    fig.layout.font.size = 16 * (fig.layout.width or 800) / 800

    return fig
