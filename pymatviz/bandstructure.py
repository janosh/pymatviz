from __future__ import annotations

from typing import Any, Union

import plotly.express as px
import plotly.graph_objects as go
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine


AnyBandStructure = Union[BandStructureSymmLine, PhononBandStructureSymmLine]


def plot_band_structure(
    band_structs: AnyBandStructure | dict[str, AnyBandStructure], **kwargs: Any
) -> go.Figure:
    """Plot single or multiple phonon band structures using Plotly, focusing on the
    minimum set of overlapping branches.

    Args:
        band_structs (AnyBandStructure | dict[str, AnyBandStructure]): Single
            BandStructureSymmLine or PhononBandStructureSymmLine object or a dictionary
            with labels mapped to multiple such objects.
        **kwargs: Passed to Plotly's Figure.add_scatter method.

    Returns:
        A Plotly figure object.
    """
    fig = go.Figure()

    # Normalize input to a dictionary
    if not isinstance(band_structs, dict):
        band_structs = {"": band_structs}
    colors = iter(px.colors.qualitative.Plotly)

    # Find common branches by normalized branch names
    common_branches: set[str] = set()
    for bs in band_structs.values():
        branches = {branch["name"].replace("GAMMA", "Γ") for branch in bs.branches}
        if not common_branches:
            common_branches = branches
        else:
            common_branches = common_branches & branches

    if not common_branches:
        raise ValueError("No common branches found among the band structures.")

    # Plotting only the common branches for each band structure
    for label, bs in band_structs.items():
        color = next(colors)
        first_trace = True
        for b in bs.branches:
            normalized_name = b["name"].replace("GAMMA", "Γ")
            if normalized_name in common_branches:
                start_index = b["start_index"]
                end_index = b["end_index"] + 1  # Include the end point
                distances = bs.distance[start_index:end_index]
                for band in range(bs.nb_bands):
                    frequencies = bs.bands[band][start_index:end_index]
                    # Group traces for toggling and set legend name only for 1st band
                    fig.add_scatter(
                        x=distances,
                        y=frequencies,
                        mode="lines",
                        line=dict(color=color),
                        legendgroup=label,
                        name=label if first_trace else None,
                        showlegend=first_trace,
                        **kwargs,
                    )
                    first_trace = False

    # Add vertical lines for common high-symmetry points
    first_bs = next(iter(band_structs.values()))
    high_symm_points_xs = []
    high_symm_points = set()
    for b in first_bs.branches:
        normalized_name = b["name"].replace("GAMMA", "Γ").replace(r"$\Gamma$", "Γ")
        if normalized_name in common_branches:
            high_symm_points.add(
                first_bs.qpoints[b["start_index"]].label.replace("GAMMA", "Γ")
            )
            high_symm_points_xs.append(first_bs.distance[b["start_index"]])

    for x_pos in high_symm_points_xs:
        fig.add_vline(x=x_pos, line=dict(color="black", width=1))

    fig.layout.title = "Band Structure"
    fig.layout.xaxis.title = "Wave Vector"
    fig.layout.yaxis.title = "Frequency (THz)"
    fig.layout.xaxis = dict(
        tickmode="array", tickvals=high_symm_points_xs, ticktext=list(high_symm_points)
    )
    fig.layout.margin = dict(t=40, b=0, l=0, r=0)

    return fig
