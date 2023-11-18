from __future__ import annotations

from typing import Union

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine


AnyBandStructure = Union[BandStructureSymmLine, PhononBandStructureSymmLine]


def plot_band_structure(
    band_structs: AnyBandStructure | dict[str, AnyBandStructure]
) -> go.Figure:
    """Plot single or multiple phonon band structures using Plotly, focusing on the
    minimum set of overlapping branches.

    Args:
        band_structs (AnyBandStructure | dict[str, AnyBandStructure]): Single
            BandStructureSymmLine or PhononBandStructureSymmLine object or a dictionary
            with labels mapped to multiple such objects.

    Returns:
        A Plotly figure object.
    """
    fig = go.Figure()

    # Normalize input to a dictionary
    if not isinstance(band_structs, dict):
        band_structs = {"": band_structs}
    colors = iter(px.colors.qualitative.Plotly)

    # Find common branches by normalized branch names
    common_branches = None
    for bs in band_structs.values():
        branches = {b["name"].replace("GAMMA", "Γ") for b in bs.branches}
        common_branches = (
            branches if common_branches is None else common_branches & branches
        )

    if not common_branches:
        raise ValueError("No common branches found among the band structures.")

    # Plotting only the common branches for each band structure
    for label, bs in band_structs.items():
        color = next(colors)
        for b in bs.branches:
            normalized_name = b["name"].replace("GAMMA", "Γ")
            if normalized_name in common_branches:
                start_index = b["start_index"]
                end_index = b["end_index"] + 1  # Include the end point
                distances = bs.distance[start_index:end_index]
                for band in range(bs.nb_bands):
                    frequencies = bs.bands[band][start_index:end_index]
                    # Group traces for toggling and set legend name only for the first band
                    legend_name = label if band == 0 else None
                    fig.add_trace(
                        go.Scatter(
                            x=distances,
                            y=frequencies,
                            mode="lines",
                            line=dict(color=color),
                            legendgroup=label,
                            name=legend_name,
                            showlegend=band == 0,  # Show legend only for the first band
                        )
                    )

    # Add vertical lines for common high-symmetry points
    first_bs = next(iter(band_structs.values()))
    high_symm_distances = [
        first_bs.distance[first_bs.branches[i]["start_index"]]
        for i in range(len(first_bs.branches))
        if first_bs.branches[i]["name"].replace("GAMMA", "Γ") in common_branches
    ]

    for dist in high_symm_distances:
        fig.add_vline(x=dist, line=dict(color="black", width=1))

    # Set x-axis and y-axis range
    all_frequencies = np.concatenate(
        [bs.bands.flatten() for bs in band_structs.values()]
    )
    fig.update_layout(
        title="Phonon Band Structure",
        xaxis_title="Wave Vector",
        yaxis_title="Frequency (THz)",
        xaxis=dict(
            tickmode="array",
            tickvals=high_symm_distances,
            ticktext=[b.replace("GAMMA", "Γ") for b in common_branches],
        ),
        yaxis=dict(range=[np.amin(all_frequencies), np.amax(all_frequencies)]),
        plot_bgcolor="white",
        margin=dict(t=40),  # Adjust top margin to ensure title fits
    )

    # Enable toggling traces by clicking on the legend
    fig.update_layout(legend=dict(itemclick="toggleothers"))

    return fig
