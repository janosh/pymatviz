"""Radial distribution functions (RDFs) of pymatgen structures using plotly.

The main function, pairwise_rdfs, generates a plotly figure with facets for each
pair of elements in the given structure. It supports customization of cutoff distance,
bin size, specific element pairs to plot, reference line.

Example usage:
    structure = Structure(...)  # Create or load a pymatgen Structure
    fig = pairwise_rdfs(structure, bin_size=0.1)
    fig.show()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly
from plotly.subplots import make_subplots
from pymatgen.core import Structure
from pymatgen.optimization.neighbors import find_points_in_spheres
from scipy.signal import find_peaks


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    import plotly.graph_objects as go
    from pymatgen.util.typing import PbcLike


def calculate_rdf(
    structure: Structure,
    center_species: str,
    neighbor_species: str,
    cutoff: float,
    n_bins: int,
    pbc: PbcLike = (True, True, True),
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
        pbc (tuple[int, int, int], optional): Periodic boundary conditions as any
            3-tuple of 0s/1s. Defaults to (1, 1, 1).

    Returns:
        tuple[np.ndarray, np.ndarray]: Arrays of (radii, g(r)) values.
    """
    bin_size = cutoff / n_bins
    radii = np.linspace(0, cutoff, n_bins + 1)[1:]
    rdf = np.zeros_like(radii)

    # Get indices of center and neighbor species
    center_indices = [
        i for i, site in enumerate(structure) if site.specie.symbol == center_species
    ]
    neighbor_indices = [
        i for i, site in enumerate(structure) if site.specie.symbol == neighbor_species
    ]

    # If there are no center atoms or neighbor atoms, return an empty RDF
    if not center_indices or not neighbor_indices:
        return radii, rdf  # Return zeros if no centers or neighbors

    center_neighbors = find_points_in_spheres(
        all_coords=structure.cart_coords,
        center_coords=structure.cart_coords[center_indices],
        r=cutoff,
        # Convert bools to ints (needed for cython code)
        pbc=np.array([*map(int, pbc)]),
        lattice=structure.lattice.matrix,
    )

    # Filter distances for the specific neighbor species and bin them
    neighbor_set = set(neighbor_indices)
    for idx1, idx2, _, dist in zip(*center_neighbors, strict=True):
        if idx2 in neighbor_set and center_indices[idx1] != idx2 and 0 < dist < cutoff:
            bin_index = min(int(dist / bin_size), n_bins - 1)
            rdf[bin_index] += 1

    # Normalize RDF by the number of center-neighbor pairs and shell volumes
    n_center = len(center_indices)
    n_neighbor = len(neighbor_indices)
    if center_species == neighbor_species:
        normalization = n_center * (n_neighbor - 1)  # Exclude self-interactions
    else:
        normalization = n_center * n_neighbor

    if normalization == 0:  # Avoid division by zero
        return radii, np.zeros_like(radii)

    # Spherical shell volume = surface area (4πr²) times thickness (bin_size)
    rdf /= normalization
    shell_volumes = 4 * np.pi * radii**2 * bin_size
    rdf /= shell_volumes / structure.volume

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
    structures: Structure | Sequence[Structure] | dict[str, Structure],
    cutoff: float = 15,
    n_bins: int = 75,
    bin_size: float | None = None,
    element_pairs: list[tuple[str, str]] | None = None,
    reference_line: dict[str, Any] | None = None,
    colors: Sequence[str] | None = None,
    line_styles: Sequence[str] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Generate a plotly figure of pairwise radial distribution functions (RDFs) for
    all (or a subset of) element pairs in one or multiple structures.

    Args:
        structures: Can be one of the following:
            - single pymatgen Structure
            - list of pymatgen Structures
            - dictionary mapping labels to Structures
        cutoff (float, optional): Maximum distance for RDF calculation. Default is 15 Å.
        n_bins (int, optional): Number of bins for RDF calculation. Default is 75.
        bin_size (float, optional): Size of bins for RDF calculation. If specified, it
            overrides n_bins. Default is None.
        element_pairs (list[tuple[str, str]], optional): Element pairs to plot.
            If None, all pairs present in any structure are plotted.
        reference_line (dict, optional): Keywords for reference line at g(r)=1 drawn
            with Figure.add_hline(). If None (default), no reference line is drawn.
        colors (Sequence[str], optional): colors for each structure's RDF line. Defaults
            to plotly.colors.qualitative.Plotly.
        line_styles (Sequence[str], optional): line styles for each structure's RDF
            line. Will be used for all element pairs present in that structure.
            Defaults to ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"].
        subplot_kwargs (dict, optional): Passed to plotly.make_subplots. Use this to
            e.g. set subplot_titles, rows/cols or row/column spacing to customize the
            subplot layout.

    Returns:
        go.Figure: A plotly figure with facets for each pairwise RDF, comparing one or
            multiple structures.

    Raises:
        ValueError: If no structures are provided, if structures have no sites,
            if invalid element pairs are provided, or if both n_bins and bin_size are
            specified.
    """
    # Normalize input to a dictionary of structures
    if isinstance(structures, Structure):
        structures = {"": structures}
    elif isinstance(structures, list | tuple) and all(
        isinstance(struct, Structure) for struct in structures
    ):
        structures = {struct.formula: struct for struct in structures}
    elif not isinstance(structures, dict):
        raise TypeError(f"Invalid input format for {structures=}")

    if not structures:
        raise ValueError("No structures provided")

    for key, struct in structures.items():
        if not struct.sites:
            raise ValueError(
                f"input structure{f' {key}' if key else ''} contains no sites"
            )

    if n_bins != 75 and bin_size is not None:
        raise ValueError(
            f"Cannot specify both {n_bins=} and {bin_size=}. Pick one or the other."
        )

    # Determine all unique elements across all structures
    all_elements = set.union(
        *(struct.chemical_system_set for struct in structures.values())
    )

    # Determine element pairs to plot
    if element_pairs is None:
        element_pairs = [
            (el1, el2) for el1 in all_elements for el2 in all_elements if el1 <= el2
        ]
    else:
        # Check if all elements in element_pairs are present in at least one structure
        pair_elements = {elem for pair in element_pairs for elem in pair}
        if extra_elems := pair_elements - set(all_elements):
            raise ValueError(
                f"Elements {extra_elems} in element_pairs not present in any structure"
            )

    element_pairs = sorted(element_pairs)

    # Calculate pairwise RDFs for all structures
    if bin_size is not None:
        n_bins = int(cutoff / bin_size)

    elem_pair_rdfs: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = {
        pair: [
            calculate_rdf(struct, *pair, cutoff, n_bins)
            for struct in structures.values()
        ]
        for pair in element_pairs
    }

    # Determine subplot layout
    n_pairs = len(element_pairs)
    subplot_kwargs = subplot_kwargs or {}
    actual_cols = min(subplot_kwargs.pop("cols", 3), n_pairs)
    n_rows = (n_pairs + actual_cols - 1) // actual_cols

    # Create the plotly figure with facets
    subplot_defaults = dict(
        rows=n_rows,
        cols=actual_cols,
        subplot_titles=[f"{el1}-{el2}" for el1, el2 in element_pairs],
        vertical_spacing=0.15 / n_rows,
        horizontal_spacing=0.15 / actual_cols,
    )
    fig = make_subplots(**subplot_defaults | subplot_kwargs)

    # Set default colors and line styles if not provided
    line_styles = line_styles or "solid dot dash longdash dashdot longdashdot".split()
    colors = colors or plotly.colors.qualitative.Plotly
    labels = list(structures)

    # Add RDF traces to the figure
    for subplot_idx, (_elem_pair, rdfs) in enumerate(elem_pair_rdfs.items()):
        row, col = divmod(subplot_idx, actual_cols)

        for trace_idx, (radii, rdf) in enumerate(rdfs):
            color = colors[trace_idx % len(colors)]
            line_style = line_styles[trace_idx % len(line_styles)]
            label = labels[trace_idx]
            fig.add_scatter(
                x=radii,
                y=rdf,
                mode="lines",
                name=label,
                line=dict(color=color, dash=line_style),
                legendgroup=label,
                showlegend=subplot_idx == 0,  # Show legend only for the first subplot
                row=row + 1,
                col=col + 1,
                hovertemplate=f"{label}<br>r = %{{x:.2f}} Å<br>g(r) = %{{y:.2f}}"
                "<extra></extra>",
            )

        # Add x-axis label
        fig.update_xaxes(title_text="r (Å)", row=row, col=col)

    # Add reference line if specified
    if reference_line is not None:
        defaults = dict(line_dash="dash", line_color="gray", opacity=0.7)
        fig.add_hline(y=1, **defaults | reference_line)

    # Set subplot height/width and y-axis labels
    fig.update_layout(height=300 * n_rows, width=450 * actual_cols)
    fig.update_yaxes(title_text="g(r)", col=1)

    # show legend centered above subplots only if multiple structures were passed
    if len(structures) > 1:
        fig.layout.legend = dict(
            orientation="h",
            xanchor="center",
            x=0.5,
            y=1.02,
            yanchor="bottom",
            font_size=14,
        )

    return fig
