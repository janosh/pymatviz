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

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pymatviz.process_data import normalize_structures
from pymatviz.rdf.helpers import calculate_rdf


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    import numpy as np

    from pymatviz.typing import AnyStructure


def element_pair_rdfs(
    structures: AnyStructure | Sequence[AnyStructure] | dict[str, AnyStructure],
    cutoff: float | None = None,
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
        structures (AnyStructure | Sequence[AnyStructure] | dict[str, AnyStructure]):
            Can be one of the following:
            - single pymatgen Structure or ASE Atoms object
            - sequence (list, tuple) of pymatgen Structures or ASE Atoms objects
            - dictionary mapping labels to pymatgen Structures or ASE Atoms objects
            - pandas Series of pymatgen Structures or ASE Atoms objects
        cutoff (float | None, optional): Maximum distance for RDF calculation.
            If None, defaults to twice the longest lattice vector length across all
            structures (up to 15A). If negative, its absolute value is used as a scaling
            factor for the longest lattice vector length (e.g. -1.5 means 1.5x the
            longest lattice vector). Default is None.
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
        TypeError: If input structures are not pymatgen Structures or ASE Atoms.
    """
    struct_dict = normalize_structures(structures)

    for key, struct in struct_dict.items():
        if not struct.sites:
            raise ValueError(
                f"input structure{f' {key}' if key else ''} contains no sites"
            )

    # Calculate dynamic cutoff if not specified or negative
    max_cell_len = max(max(struct.lattice.abc) for struct in struct_dict.values())
    if cutoff is None:
        cutoff = min(15, 2 * max_cell_len)
    elif cutoff < 0:
        cutoff = abs(cutoff) * max_cell_len
    if not isinstance(cutoff, int | float):
        raise TypeError(f"Invalid {cutoff=}")

    if n_bins != 75 and bin_size is not None:
        raise ValueError(
            f"Cannot specify both {n_bins=} and {bin_size=}. Pick one or the other."
        )

    # Determine all unique elements across all structures
    all_elements = set.union(
        *(struct.chemical_system_set for struct in struct_dict.values())
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
            calculate_rdf(struct, pair[0], pair[1], cutoff=cutoff, n_bins=n_bins)
            for struct in struct_dict.values()
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
    colors = colors or plotly.colors.qualitative.Plotly
    line_styles = line_styles or (
        "solid",
        "dot",
        "dash",
        "longdash",
        "dashdot",
        "longdashdot",
    )
    labels = list(struct_dict)

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
                # Only show legend for first subplot and if multiple structures
                showlegend=subplot_idx == 0 and len(struct_dict) > 1,
                row=row + 1,
                col=col + 1,
                hovertemplate=f"{label}<br>r = %{{x:.2f}} Å<br>g(r) = %{{y:.2f}}"
                "<extra></extra>",
            )

    fig.update_xaxes(title_text="r [Å]", title_standoff=9, row=n_rows)
    fig.update_yaxes(title_text="g(r)", title_standoff=9, col=1)
    fig.update_layout(height=300 * n_rows, width=450 * actual_cols)

    # Add reference line if specified
    if reference_line is not None:
        hline_defaults = dict(line_dash="dash", line_color="gray", opacity=0.7)
        fig.add_hline(y=1, **hline_defaults | reference_line)

    # show legend centered above subplots only if multiple structures were passed
    if len(struct_dict) > 1:
        fig.layout.showlegend = True
        fig.layout.legend.update(
            orientation="h",
            xanchor="center",
            x=0.5,
            y=1.02,
            yanchor="bottom",
            font_size=14,
        )
    else:
        # Hide legend for single structure
        fig.layout.showlegend = False

    return fig


def full_rdf(
    structures: AnyStructure | Sequence[AnyStructure] | dict[str, AnyStructure],
    cutoff: float = 15,
    n_bins: int = 75,
    bin_size: float | None = None,
    reference_line: dict[str, Any] | None = None,
    colors: Sequence[str] | None = None,
    line_styles: Sequence[str] | None = None,
) -> go.Figure:
    """Generate a plotly figure of full radial distribution functions (RDFs) for
    one or multiple structures.

    Args:
        structures (AnyStructure | Sequence[AnyStructure] | dict[str, AnyStructure]):
            Can be one of the following:
            - single pymatgen Structure or ASE Atoms object
            - list of pymatgen Structures or ASE Atoms objects
            - dictionary mapping labels to pymatgen Structures or ASE Atoms objects.
        cutoff (float, optional): Maximum distance for RDF calculation. Default is 15 Å.
        n_bins (int, optional): Number of bins for RDF calculation. Default is 75.
        bin_size (float, optional): Size of bins for RDF calculation. If specified, it
            overrides n_bins. Default is None.
        reference_line (dict, optional): Keywords for reference line at g(r)=1 drawn
            with Figure.add_hline(). If None (default), no reference line is drawn.
        colors (Sequence[str], optional): colors for each structure's RDF line. Defaults
            to plotly.colors.qualitative.Plotly.
        line_styles (Sequence[str], optional): line styles for each structure's RDF
            line. Defaults to ["solid", "dot", "dash", "longdash", "dashdot",
            "longdashdot"].

    Returns:
        go.Figure: A plotly figure with full RDFs for one or multiple structures.

    Raises:
        ValueError: If no structures are provided, if structures have no sites,
            or if both n_bins and bin_size are specified.
    """
    struct_dict = normalize_structures(structures)

    for key, struct in struct_dict.items():
        if not struct.sites:
            raise ValueError(
                f"input structure{f' {key}' if key else ''} contains no sites"
            )

    if n_bins != 75 and bin_size is not None:
        raise ValueError(
            f"Cannot specify both {n_bins=} and {bin_size=}. Pick one or the other."
        )

    # Calculate full RDFs for all structures
    if bin_size is not None:
        n_bins = int(cutoff / bin_size)

    rdfs = {
        label: calculate_rdf(struct, cutoff=cutoff, n_bins=n_bins)
        for label, struct in struct_dict.items()
    }

    fig = go.Figure()

    colors = colors or plotly.colors.qualitative.Plotly
    line_styles = line_styles or (
        "solid",
        "dot",
        "dash",
        "longdash",
        "dashdot",
        "longdashdot",
    )

    for idx, (label, (radii, rdf)) in enumerate(rdfs.items()):
        fig.add_scatter(
            x=radii,
            y=rdf,
            mode="lines",
            name=label,
            line=dict(
                color=colors[idx % len(colors)],
                dash=line_styles[idx % len(line_styles)],
            ),
            hovertemplate=f"{label}<br>r = %{{x:.2f}} Å<br>g(r) = %{{y:.2f}}"
            "<extra></extra>",
        )

    fig.update_layout(xaxis_title="r [Å]", yaxis_title="g(r)")

    if reference_line is not None:
        hline_defaults = dict(line_dash="dash", line_color="gray", opacity=0.7)
        fig.add_hline(y=1, **hline_defaults | reference_line)

    # show legend centered above subplots only if multiple structures were passed
    if len(struct_dict) > 1:
        fig.layout.showlegend = True
        fig.layout.legend.update(
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
        )
    else:
        fig.layout.showlegend = False

    return fig
