"""Radial distribution functions (RDFs) of pymatgen structures using plotly.

The main function, pairwise_rdfs, generates a plotly figure with facets for each
pair of elements in the given structure. It supports customization of cutoff distance,
bin size, specific element pairs to plot, reference line.

Example usage:
    structure = Structure(...)  # Create or load a pymatgen Structure
    fig = pairwise_rdfs(structure, bin_size=0.1)
    fig.show()
"""

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pymatgen.core import Structure
from scipy.signal import find_peaks


def calculate_rdf(
    structure: Structure,
    center_species: str,
    neighbor_species: str,
    cutoff: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the radial distribution function (RDF) for a given pair of species.

    The RDF is normalized by the number of pairs and the shell volume density, which
    makes the RDF approach 1 for large separations in a homogeneous system.

    Args:
        structure (Structure): A pymatgen Structure object.
        center_species (str): Symbol of the central species.
        neighbor_species (str): Symbol of the neighbor species.
        cutoff (float): Maximum distance for RDF calculation.
        n_bins (int): Number of bins for RDF calculation.

    Returns:
        tuple[np.ndarray, np.ndarray]: Arrays of (radii, g(r)) values.
    """
    bin_size = cutoff / n_bins
    radii = np.linspace(0, cutoff, n_bins + 1)[1:]
    rdf = np.zeros(n_bins)

    center_indices = [
        i for i, site in enumerate(structure) if site.specie.symbol == center_species
    ]
    neighbor_indices = [
        i for i, site in enumerate(structure) if site.specie.symbol == neighbor_species
    ]

    for center_idx in center_indices:
        for neighbor_idx in neighbor_indices:
            if center_idx != neighbor_idx:
                distance = structure.get_distance(center_idx, neighbor_idx)
                if distance < cutoff:
                    rdf[int(distance / bin_size)] += 1

    # Normalize RDF by the number of center-neighbor pairs and shell volumes
    rdf = rdf / (len(center_indices) * len(neighbor_indices))
    shell_volumes = 4 * np.pi * radii**2 * bin_size
    rdf = rdf / (shell_volumes / structure.volume)

    return radii, rdf


def find_last_significant_peak(
    radii: np.ndarray, rdf: np.ndarray, prominence: float = 0.1
) -> float:
    """Find the position of the last significant peak in the RDF."""
    peaks, properties = find_peaks(rdf, prominence=prominence, distance=5)
    if peaks.size > 0:
        # Sort peaks by prominence and select the last significant one
        sorted_peaks = peaks[np.argsort(properties["prominences"])]
        return radii[sorted_peaks[-1]]
    return radii[-1]


def element_pair_rdfs(
    structure: Structure,
    cutoff: float = 15,
    n_bins: int = 75,
    bin_size: float | None = None,
    element_pairs: list[tuple[str, str]] | None = None,
    reference_line: dict[str, Any] | None = None,
    n_cols: int = 3,
) -> go.Figure:
    """Generate a plotly figure of pairwise radial distribution functions (RDFs) for
    all (or a subset of) element pairs in a structure.

    The RDF is the probability of finding a neighbor at a distance r from a central
    atom. Basically a histogram of pair-wise particle distances.

    Args:
        structure (Structure): pymatgen Structure.
        cutoff (float, optional): Maximum distance for RDF calculation. Default is 15 Å.
        n_bins (int, optional): Number of bins for RDF calculation. Default is 75.
        bin_size (float, optional): Size of bins for RDF calculation. If specified, it
            overrides n_bins. Default is None.
        element_pairs (list[tuple[str, str]], optional): Element pairs to plot.
            If None, all pairs are plotted.
        reference_line (dict, optional): Keywords for reference line at g(r)=1 drawn
            with Figure.add_hline(). If None (default), no reference line is drawn.
        n_cols (int, optional): Number of columns for subplot layout. Defaults to 3.

    Returns:
        go.Figure: A plotly figure with facets for each pairwise RDF.

    Raises:
        ValueError: If the structure contains no sites, if invalid element pairs are
            provided, or if both n_bins and bin_size are specified.
    """
    if not structure.sites:
        raise ValueError("input structure contains no sites")

    if n_bins != 75 and bin_size is not None:
        raise ValueError(
            f"Cannot specify both {n_bins=} and {bin_size=}. Pick one or the other."
        )

    uniq_elements = sorted({site.specie.symbol for site in structure})
    element_pairs = element_pairs or [
        (e1, e2) for e1 in uniq_elements for e2 in uniq_elements if e1 <= e2
    ]
    element_pairs = sorted(element_pairs)

    if extra_elems := {e1 for e1, _e2 in element_pairs} - set(uniq_elements):
        raise ValueError(
            f"Elements {extra_elems} in element_pairs are not present in the structure"
        )

    # Calculate pairwise RDFs
    if bin_size is not None:
        n_bins = int(cutoff / bin_size)
    elem_pair_rdfs = {
        pair: calculate_rdf(structure, *pair, cutoff, n_bins) for pair in element_pairs
    }

    # Determine subplot layout
    n_pairs = len(element_pairs)
    actual_cols = min(n_cols, n_pairs)
    n_rows = (n_pairs + actual_cols - 1) // actual_cols

    # Create the plotly figure with facets
    fig = make_subplots(
        rows=n_rows,
        cols=actual_cols,
        subplot_titles=[f"{e1}-{e2}" for e1, e2 in element_pairs],
        vertical_spacing=0.25 / n_rows,
        horizontal_spacing=0.15 / actual_cols,
    )

    # Add RDF traces to the figure
    for idx, (pair, (radii, rdf)) in enumerate(elem_pair_rdfs.items()):
        row, col = divmod(idx, actual_cols)
        row += 1
        col += 1

        fig.add_scatter(
            x=radii,
            y=rdf,
            mode="lines",
            name=f"{pair[0]}-{pair[1]}",
            line=dict(color="royalblue"),
            showlegend=False,
            row=row,
            col=col,
            hovertemplate="r = %{x:.2f} Å<br>g(r) = %{y:.2f}<extra></extra>",
        )

        # if one of the last n_col subplots, add x-axis label
        if idx >= n_pairs - actual_cols:
            fig.update_xaxes(title_text="r (Å)", row=row, col=col)

        # Add reference line if specified
        if reference_line is not None:
            defaults = dict(line_dash="dash", line_color="red")
            fig.add_hline(y=1, row=row, col=col, **defaults | reference_line)

    # set subplot height/width and x/y axis labels
    fig.update_layout(height=200 * n_rows, width=350 * actual_cols)
    fig.update_yaxes(title=dict(text="g(r)", standoff=0.1), col=1)

    return fig
