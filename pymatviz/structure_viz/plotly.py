"""2D plots of pymatgen structures with matplotlib.

structure_2d() and its helpers get_rot_matrix() and unit_cell_to_lines() were
inspired by ASE https://wiki.fysik.dtu.dk/ase/ase/visualize/visualize.html#matplotlib.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from itertools import product
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SymmetryUndeterminedError

from pymatviz.enums import ElemColorScheme
from pymatviz.structure_viz.helpers import (
    NO_SYM_MSG,
    UNIT_CELL_EDGES,
    _angles_to_rotation_matrix,
    add_site_to_plot,
    generate_subplot_title,
    get_atomic_radii,
    get_elem_colors,
    get_image_atoms,
    get_structures,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any, Literal

    import plotly.graph_objects as go
    from pymatgen.core import Structure


def structure_2d_plotly(
    struct: Structure | Sequence[Structure],
    *,
    rotation: str = "10x,8y,3z",
    atomic_radii: float | dict[str, float] | None = None,
    atom_size: float = 40,
    elem_colors: ElemColorScheme | dict[str, str] = ElemColorScheme.jmol,
    scale: float = 1,
    show_unit_cell: bool | dict[str, Any] = True,
    show_sites: bool | dict[str, Any] = True,
    show_image_sites: bool | dict[str, Any] = True,
    site_labels: Literal["symbol", "species", False]
    | dict[str, str]
    | Sequence[str] = "species",
    standardize_struct: bool | None = None,
    n_cols: int = 3,
    subplot_title: Callable[[Structure, str | int], str | dict[str, Any]] | None = None,
) -> go.Figure:
    """Plot pymatgen structures in 2D with Plotly.

    Args:
        struct (Structure | Sequence[Structure]): Pymatgen Structure(s) to plot.
        rotation (str, optional): Euler angles in degrees in the form '10x,20y,30z'
            describing angle at which to view structure. Defaults to "10x,8y,3z".
        atomic_radii (float | dict[str, float], optional): Either a scaling factor for
            default radii or map from element symbol to atomic radii. Defaults to None.
        atom_size (float, optional): Scaling factor for atom sizes. Defaults to 40.
        elem_colors (ElemColorScheme | dict[str, str], optional): Element color scheme
            or custom color map. Defaults to ElemColorScheme.jmol.
        scale (float, optional): Scaling of the plotted atoms and lines. Defaults to 1.
        show_unit_cell (bool | dict[str, Any], optional): Whether to plot unit cell. If
            a dict, will be used to customize unit cell appearance. Defaults to True.
        show_sites (bool | dict[str, Any], optional): Whether to plot atomic sites. If
            a dict, will be used to customize site marker appearance. Defaults to True.
        show_image_sites (bool | dict[str, Any], optional): Whether to show image sites
            on unit cell edges and surfaces. If a dict, will be used to customize how
            image sites are rendered. Defaults to True.
        site_labels ("symbol" | "species" | dict[str, str] | Sequence):
            How to annotate lattice sites. Defaults to "species".
        standardize_struct (bool, optional): Whether to standardize the structure.
            Defaults to None.
        n_cols (int, optional): Number of columns for subplots. Defaults to 4.
        subplot_title (Callable[[Structure, str | int], str | dict], optional):
            Function to generate subplot titles. Defaults to None.

    Returns:
        go.Figure: Plotly figure with the plotted structure(s).
    """
    structures = get_structures(struct)

    n_structs = len(structures)
    n_cols = min(n_cols, n_structs)
    n_rows = math.ceil(n_structs / n_cols)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[" " for _ in range(n_structs)],
        vertical_spacing=0,
        horizontal_spacing=0,
    )

    _elem_colors = get_elem_colors(elem_colors)
    _atomic_radii = get_atomic_radii(atomic_radii)

    for idx, (struct_key, struct_i) in enumerate(structures.items(), start=1):
        row = (idx - 1) // n_cols + 1
        col = (idx - 1) % n_cols + 1

        # Standardize structure if needed
        if standardize_struct is None:
            standardize_struct = any(any(site.frac_coords < 0) for site in struct_i)
        if standardize_struct:
            try:
                spg_analyzer = SpacegroupAnalyzer(struct_i)
                struct_i = spg_analyzer.get_conventional_standard_structure()  # noqa: PLW2901
            except SymmetryUndeterminedError:
                warnings.warn(NO_SYM_MSG, UserWarning, stacklevel=2)

        # Apply rotation
        rotation_matrix = _angles_to_rotation_matrix(rotation)
        rotated_coords = np.dot(struct_i.cart_coords, rotation_matrix)

        # Plot atoms
        if show_sites:
            site_kwargs = dict(line=dict(width=0.3, color="gray"))
            if isinstance(show_sites, dict):
                site_kwargs |= show_sites

            for site_idx, (site, coords) in enumerate(
                zip(struct_i, rotated_coords, strict=False)
            ):
                add_site_to_plot(
                    fig,
                    site,
                    coords,
                    site_idx,
                    site_labels,
                    _elem_colors,
                    _atomic_radii,
                    atom_size,
                    scale,
                    site_kwargs,
                    is_3d=False,  # Explicitly set to False for 2D plot
                    row=row,
                    col=col,
                )

                # Add image sites
                if show_image_sites:
                    image_site_kwargs = dict(
                        size=_atomic_radii[site.species.elements[0].symbol]
                        * scale
                        * atom_size
                        * 0.8,
                        color=_elem_colors.get(site.species.elements[0].symbol, "gray"),
                        opacity=0.5,
                    )
                    if isinstance(show_image_sites, dict):
                        image_site_kwargs |= show_image_sites

                    image_atoms = get_image_atoms(site, struct_i.lattice)
                    if image_atoms:  # Only proceed if there are image atoms
                        rotated_image_atoms = np.dot(image_atoms, rotation_matrix)

                        for image_coords in rotated_image_atoms:
                            add_site_to_plot(
                                fig,
                                site,
                                image_coords,
                                site_idx,
                                site_labels,
                                _elem_colors,
                                _atomic_radii,
                                atom_size,
                                scale,
                                image_site_kwargs,
                                is_image=True,
                                is_3d=False,  # Explicitly set to False for 2D plot
                                row=row,
                                col=col,
                            )

        # Plot unit cell
        if show_unit_cell:
            corners = np.array(list(product((0, 1), (0, 1), (0, 1))))
            cell_vertices = np.dot(
                np.dot(corners, struct_i.lattice.matrix), rotation_matrix
            )
            unit_cell_kwargs = dict(color="black", width=1, dash=None)
            if isinstance(show_unit_cell, dict):
                unit_cell_kwargs |= show_unit_cell

            for start, end in UNIT_CELL_EDGES:
                hover_text = (
                    f"Start: ({', '.join(f'{c:.3g}' for c in cell_vertices[start])}) "
                    f"[{', '.join(f'{c:.3g}' for c in corners[start])}]<br>"
                    f"End: ({', '.join(f'{c:.3g}' for c in cell_vertices[end])}) "
                    f"[{', '.join(f'{c:.3g}' for c in corners[end])}]"
                )
                fig.add_scatter(
                    x=[cell_vertices[start, 0], cell_vertices[end, 0]],
                    y=[cell_vertices[start, 1], cell_vertices[end, 1]],
                    mode="lines",
                    line=unit_cell_kwargs,
                    text=hover_text,
                    hoverinfo="text",
                    showlegend=False,
                    row=row,
                    col=col,
                )

        # Set subplot titles
        anno = generate_subplot_title(struct_i, struct_key, idx, subplot_title)
        subtitle_y_pos = 1 - (row - 1) / n_rows - 0.02
        anno |= dict(y=subtitle_y_pos, yanchor="top")
        fig.layout.annotations[idx - 1].update(**anno)

    # Update layout
    fig.layout.height = 300 * n_rows
    fig.layout.width = 300 * n_cols
    fig.layout.showlegend = False
    fig.layout.paper_bgcolor = "rgba(0,0,0,0)"
    fig.layout.plot_bgcolor = "rgba(0,0,0,0)"
    fig.layout.margin = dict(l=10, r=10, t=40, b=10)
    common_kwargs = dict(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        scaleratio=1,
        constrain="domain",
    )
    fig.update_xaxes(**common_kwargs, scaleanchor="y")
    fig.update_yaxes(**common_kwargs, scaleanchor="x")

    return fig


def structure_3d_plotly(
    struct: Structure | Sequence[Structure],
    *,
    atomic_radii: float | dict[str, float] | None = None,
    atom_size: float = 20,
    elem_colors: ElemColorScheme | dict[str, str] = ElemColorScheme.jmol,
    scale: float = 1,
    show_unit_cell: bool | dict[str, Any] = True,
    show_sites: bool | dict[str, Any] = True,
    show_image_sites: bool = True,
    site_labels: Literal["symbol", "species", False]
    | dict[str, str]
    | Sequence[str] = "species",
    standardize_struct: bool | None = None,
    n_cols: int = 3,
    subplot_title: Callable[[Structure, str | int], str | dict[str, Any]] | None = None,
) -> go.Figure:
    """Plot pymatgen structures in 3D with Plotly.

    Args:
        struct (Structure | Sequence[Structure]): Pymatgen Structure(s) to plot.
        atomic_radii (float | dict[str, float], optional): Either a scaling factor for
            default radii or map from element symbol to atomic radii. Defaults to None.
        atom_size (float, optional): Scaling factor for atom sizes. Defaults to 20.
        elem_colors (ElemColorScheme | dict[str, str], optional): Element color scheme
            or custom color map. Defaults to ElemColorScheme.jmol.
        scale (float, optional): Scaling of the plotted atoms and lines. Defaults to 1.
        show_unit_cell (bool | dict[str, Any], optional): Whether to plot unit cell. If
            a dict, will be used to customize unit cell appearance. Defaults to True.
        show_sites (bool | dict[str, Any], optional): Whether to plot atomic sites. If
            a dict, will be used to customize site marker appearance. Defaults to True.
        show_image_sites (bool | dict[str, Any], optional): Whether to show image sites
            on unit cell edges and surfaces. If a dict, will be used to customize how
            image sites are rendered. Defaults to True.
        site_labels ("symbol" | "species" | dict[str, str] | Sequence):
            How to annotate lattice sites. Defaults to "species".
        standardize_struct (bool, optional): Whether to standardize the structure.
            Defaults to None.
        n_cols (int, optional): Number of columns for subplots. Defaults to 3.
        subplot_title (Callable[[Structure, str | int], str | dict], optional):
            Function to generate subplot titles. Defaults to None.

    Returns:
        go.Figure: Plotly figure with the plotted 3D structure(s).
    """
    structures = get_structures(struct)

    n_structs = len(structures)
    n_cols = min(n_cols, n_structs)
    n_rows = (n_structs - 1) // n_cols + 1

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=[[{"type": "scene"} for _ in range(n_cols)] for _ in range(n_rows)],
        subplot_titles=[" " for _ in range(n_structs)],
    )

    _elem_colors = get_elem_colors(elem_colors)
    _atomic_radii = get_atomic_radii(atomic_radii)

    for idx, (struct_key, struct_i) in enumerate(structures.items(), start=1):
        # Standardize structure if needed
        if standardize_struct is None:
            standardize_struct = any(any(site.frac_coords < 0) for site in struct_i)
        if standardize_struct:
            try:
                spg_analyzer = SpacegroupAnalyzer(struct_i)
                struct_i = spg_analyzer.get_conventional_standard_structure()  # noqa: PLW2901
            except SymmetryUndeterminedError:
                warnings.warn(NO_SYM_MSG, UserWarning, stacklevel=2)

        # Plot atoms
        if show_sites:
            site_kwargs = dict(line=dict(width=0.3, color="gray"))
            if isinstance(show_sites, dict):
                site_kwargs |= show_sites

            for site_idx, site in enumerate(struct_i):
                add_site_to_plot(
                    fig,
                    site,
                    site.coords,
                    site_idx,
                    site_labels,
                    _elem_colors,
                    _atomic_radii,
                    atom_size,
                    scale,
                    site_kwargs,
                    is_3d=True,
                    scene=f"scene{idx}",
                )

                # Add image sites
                if show_image_sites:
                    image_site_kwargs = dict(
                        size=_atomic_radii[site.species.elements[0].symbol]
                        * scale
                        * atom_size
                        * 0.8,
                        color=_elem_colors.get(site.species.elements[0].symbol, "gray"),
                        opacity=0.5,
                    )
                    if isinstance(show_image_sites, dict):
                        image_site_kwargs |= show_image_sites

                    image_atoms = get_image_atoms(site, struct_i.lattice)
                    if image_atoms:  # Only proceed if there are image atoms
                        for image_coords in image_atoms:
                            add_site_to_plot(
                                fig,
                                site,
                                image_coords,
                                site_idx,
                                site_labels,
                                _elem_colors,
                                _atomic_radii,
                                atom_size,
                                scale,
                                image_site_kwargs,
                                is_image=True,
                                is_3d=True,
                                scene=f"scene{idx}",
                            )

        # Plot unit cell
        if show_unit_cell:
            corners = np.array(list(product((0, 1), (0, 1), (0, 1))))
            cell_vertices = np.dot(corners, struct_i.lattice.matrix)
            unit_cell_kwargs = dict(color="black", width=2)
            if isinstance(show_unit_cell, dict):
                unit_cell_kwargs |= show_unit_cell

            for start, end in UNIT_CELL_EDGES:
                hover_text = (
                    f"Start: ({', '.join(f'{c:.3g}' for c in cell_vertices[start])}) "
                    f"[{', '.join(f'{c:.3g}' for c in corners[start])}]<br>"
                    f"End: ({', '.join(f'{c:.3g}' for c in cell_vertices[end])}) "
                    f"[{', '.join(f'{c:.3g}' for c in corners[end])}]"
                )
                fig.add_scatter3d(
                    x=[cell_vertices[start, 0], cell_vertices[end, 0]],
                    y=[cell_vertices[start, 1], cell_vertices[end, 1]],
                    z=[cell_vertices[start, 2], cell_vertices[end, 2]],
                    mode="lines",
                    line=unit_cell_kwargs,
                    text=hover_text,
                    hoverinfo="text",
                    showlegend=False,
                    scene=f"scene{idx}",
                )

        # Set subplot titles
        anno = generate_subplot_title(struct_i, struct_key, idx, subplot_title)
        row = (idx - 1) // n_cols + 1
        subtitle_y_pos = 1 - (row - 1) / n_rows - 0.02
        anno |= dict(y=subtitle_y_pos, yanchor="top")
        fig.layout.annotations[idx - 1].update(**anno)

        # Update 3D scene properties
        no_axes_kwargs = dict(
            showticklabels=False, showgrid=False, zeroline=False, visible=False
        )

        fig.update_scenes(
            xaxis=no_axes_kwargs,
            yaxis=no_axes_kwargs,
            zaxis=no_axes_kwargs,
            aspectmode="data",
            bgcolor="rgba(90,90,90,0.01)",  # Transparent background
        )

    # Calculate subplot positions with 2% gap
    gap = 0.01
    for idx in range(1, n_structs + 1):
        row = (idx - 1) // n_cols + 1
        col = (idx - 1) % n_cols + 1

        x_start = (col - 1) / n_cols + gap / 2
        x_end = col / n_cols - gap / 2
        y_start = 1 - row / n_rows + gap / 2  # Invert y-axis to match row order
        y_end = 1 - (row - 1) / n_rows - gap / 2

        domain = dict(x=[x_start, x_end], y=[y_start, y_end])
        fig.update_layout({f"scene{idx}": dict(domain=domain, aspectmode="data")})

    # Update overall layout
    fig.update_layout(
        height=400 * n_rows,
        width=400 * n_cols,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
        margin=dict(l=0, r=0, t=0, b=0),  # Minimize margins
    )

    return fig
