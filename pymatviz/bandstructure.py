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

    if isinstance(band_structs, PhononBandStructureSymmLine):
        band_structs = {"Single BS": band_structs}

    # Define a list of colors for each band structure
    # symbol_iter = iter(SymbolValidator().values[2::6])
    colors = iter(px.colors.qualitative.Dark24)

    # Find the common branches
    common_branches = None
    for bs in band_structs.values():
        branches = {b["name"] for b in bs.branches}
        if common_branches is None:
            common_branches = branches
        else:
            common_branches &= branches
    if common_branches is None:
        raise ValueError("No common branches found among the band structures.")

    # Plotting only the common branches for each band structure
    for label, bs in band_structs.items():
        line_color = next(colors, "blue")  # Use blue if colorway is exhausted
        for band in range(bs.nb_bands):
            for branch in bs.branches:
                if branch["name"] in common_branches:
                    start_index = branch["start_index"]
                    end_index = branch["end_index"] + 1  # Include the end point
                    distances = bs.distance[start_index:end_index]
                    frequencies = bs.bands[band, start_index:end_index]
                    fig.add_trace(
                        go.Scatter(
                            x=distances,
                            y=frequencies,
                            mode="lines",
                            line=dict(color=line_color),
                            name=label,
                        )
                    )

    # Adding vertical lines at high-symmetry points for the first band structure
    first_bs = next(iter(band_structs.values()))
    high_symm_distances = [
        first_bs.distance[first_bs.branches[i]["start_index"]]
        for i in range(len(first_bs.branches))
        if first_bs.branches[i]["name"] in common_branches
    ]

    for dist in high_symm_distances:
        fig.add_vline(x=dist, line=dict(color="black", width=1))

    # Customizing the plot appearance
    fig.update_layout(
        title="Phonon Band Structure",
        xaxis_title="Wave Vector",
        yaxis_title="Frequency (THz)",
        xaxis=dict(
            range=[min(high_symm_distances), max(high_symm_distances)],
            tickmode="array",
            tickvals=high_symm_distances,
            ticktext=[
                first_bs.qpoints[first_bs.branches[i]["start_index"]].label.replace(
                    "GAMMA", "Γ"
                )
                for i in range(len(first_bs.branches))
                if first_bs.branches[i]["name"] in common_branches
            ],
        ),
        yaxis=dict(range=[np.amin(first_bs.bands), np.amax(first_bs.bands)]),
        plot_bgcolor="white",
        margin=dict(t=40),  # Adjust top margin to ensure title fits
    )

    # Enable legend toggle only for multiple band structures
    if len(band_structs) > 1:
        fig.update_layout(legend=dict(itemclick="toggleothers"))

    return fig
