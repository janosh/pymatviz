"""Create interactive hoverable 2D and 3D plots of pymatgen structures with plotly."""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import numpy as np
from plotly.subplots import make_subplots
from pymatgen.analysis.local_env import CrystalNN, NearNeighbors
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pymatviz.enums import ElemColorScheme, SiteCoords
from pymatviz.structure_viz.helpers import (
    NO_SYM_MSG,
    _angles_to_rotation_matrix,
    draw_bonds,
    draw_site,
    draw_unit_cell,
    draw_vector,
    get_atomic_radii,
    get_elem_colors,
    get_first_matching_site_prop,
    get_image_sites,
    get_structures,
    get_subplot_title,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any, Literal

    import ase.atoms
    import plotly.graph_objects as go
    from pymatgen.core import PeriodicSite, Structure

    from pymatviz.typing import ColorType


def structure_2d_plotly(
    struct: Structure
    | Sequence[Structure]
    | ase.atoms.Atoms
    | Sequence[ase.atoms.Atoms],
    *,
    rotation: str = "10x,8y,3z",
    atomic_radii: float | dict[str, float] | None = None,
    atom_size: float = 40,
    elem_colors: ElemColorScheme | dict[str, ColorType] = ElemColorScheme.jmol,
    scale: float = 1,
    show_unit_cell: bool | dict[str, Any] = True,
    show_sites: bool | dict[str, Any] = True,
    show_image_sites: bool | dict[str, Any] = True,
    show_bonds: bool | NearNeighbors = False,
    site_labels: Literal["symbol", "species", False]
    | dict[str, str]
    | Sequence[str] = "species",
    standardize_struct: bool | None = None,
    n_cols: int = 3,
    subplot_title: Callable[[Structure, str | int], str | dict[str, Any]] | None = None,
    show_site_vectors: str | Sequence[str] = ("force", "magmom"),
    vector_kwargs: dict[str, dict[str, Any]] | None = None,
    hover_text: SiteCoords
    | Callable[[PeriodicSite], str] = SiteCoords.cartesian_fractional,
    bond_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Plot pymatgen structures in 2D with Plotly.

    Args:
        struct (Structure | Sequence[Structure]): Pymatgen Structure(s) to plot.
        rotation (str, optional): Rotation angles in degrees in the form '10x,20y,30z'
            from which to view the structure. Defaults to "10x,8y,3z".
        atomic_radii (float | dict[str, float], optional): Either a scaling factor for
            default radii or map from element symbol to atomic radii. Defaults to None.
        atom_size (float, optional): Scaling factor for atom sizes. Defaults to 40.
        elem_colors (ElemColorScheme | dict[str, ColorType], optional): Element color
            scheme or custom color map. Defaults to ElemColorScheme.jmol.
        scale (float, optional): Scaling of the plotted atoms and lines. Defaults to 1.
        show_unit_cell (bool | dict[str, Any], optional): Whether to plot unit cell. If
            a dict, will be used to customize unit cell appearance. The dict should have
            a "node"/"edge" key to customize node/edge appearance. Defaults to True.
        show_sites (bool | dict[str, Any], optional): Whether to plot atomic sites. If
            a dict, will be used to customize site marker appearance. Defaults to True.
        show_image_sites (bool | dict[str, Any], optional): Whether to show image sites
            on unit cell edges and surfaces. If a dict, will be used to customize how
            image sites are rendered. Defaults to True.
        show_bonds (bool | NearNeighbors, optional): Whether to draw bonds between
            sites. If True, uses CrystalNN with a search_cutoff of 10 Å to determine
            nearest neighbors, including those in neighboring cells. If a NearNeighbors
            object, uses that to determine nearest neighbors.
            Defaults to False (since still experimental).
        site_labels ("symbol" | "species" | dict[str, str] | Sequence):
            How to annotate lattice sites. Defaults to "species".
        standardize_struct (bool, optional): Whether to standardize the structure.
            Defaults to None.
        n_cols (int, optional): Number of columns for subplots. Defaults to 4.
        subplot_title (Callable[[Structure, str | int], str | dict] | False, optional):
            Function to generate subplot titles. Defaults to
            lambda struct_i, idx: f"{idx}. {struct_i.formula} (spg={spg_num})". Set to
            False to hide all subplot titles.
        show_site_vectors (str | Sequence[str], optional): Whether to show vector site
            quantities such as forces or magnetic moments as arrow heads originating
            from each site. Pass the key (or sequence of keys) to look for in site
            properties. Defaults to ("force", "magmom"). If not found as a site
            property, will look for it in the structure properties as well and assume
            the key points at a (N, 3) array with N the number of sites. If multiple
            keys are provided, it plots the first key found in site properties or
            structure properties in any of the passed structures (if a dict of
            structures was passed). But it will only plot one vector per site and it
            will use the same key for all sites and across all structures.
        vector_kwargs (dict[str, dict[str, Any]], optional): For customizing vector
            arrows. Keys are property names (e.g., "force", "magmom"), values are
            dictionaries of arrow customization options. Use key "scale" to adjust
            vector length.
        hover_text (SiteCoords | Callable, optional): Controls the hover tooltip
            template. Can be SiteCoords.cartesian, SiteCoords.fractional,
            SiteCoords.cartesian_fractional, or a callable that takes a site and
            returns a custom string. Defaults to SiteCoords.cartesian_fractional.
        bond_kwargs (dict[str, Any], optional): For customizing bond lines. Keys are
            line properties (e.g., "color", "width"), values are the corresponding
            values. Defaults to None.

    Returns:
        go.Figure: Plotly figure showing the 2D structure(s).
    """
    structures = get_structures(struct)

    n_structs = len(structures)
    n_cols = min(n_cols, n_structs)
    n_rows = math.ceil(n_structs / n_cols)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        # needed to avoid IndexError on fig.layout.annotations[idx - 1].update(anno)
        subplot_titles=[" " for _ in range(n_structs)],
        vertical_spacing=0,
        horizontal_spacing=0,
    )

    _elem_colors = get_elem_colors(elem_colors)
    _atomic_radii = get_atomic_radii(atomic_radii)

    if isinstance(show_site_vectors, str):
        show_site_vectors = [show_site_vectors]

    # Determine which vector property to plot (calling outside loop ensures we plot the
    # same prop for all sites in all structures)
    vector_prop = get_first_matching_site_prop(
        list(structures.values()),
        show_site_vectors,
        warn_if_none=show_site_vectors != ("force", "magmom"),
        # check that value is actually a 3-component vector. needs to handle both
        # (N, 3) and (3,) cases
        filter_callback=lambda _prop, value: (np.array(value).shape or [None])[-1] == 3,
    )

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
            except ValueError:
                warnings.warn(NO_SYM_MSG, UserWarning, stacklevel=2)

        # Apply rotation
        rotation_matrix = _angles_to_rotation_matrix(rotation)
        rotated_coords = np.dot(struct_i.cart_coords, rotation_matrix)

        visible_image_atoms: set[tuple[float, float, float]] = set()

        # First, populate visible_image_atoms with all image sites that will be drawn
        if show_image_sites and show_sites:
            for site in struct_i:
                image_atoms = get_image_sites(site, struct_i.lattice)
                if len(image_atoms) > 0:
                    for image_coords in image_atoms:
                        # Apply the same rotation to image atoms as to regular atoms
                        rotated_image_coords = np.dot(image_coords, rotation_matrix)
                        visible_image_atoms.add(tuple(rotated_image_coords))

        # Draw bonds
        if show_bonds:
            draw_bonds(
                fig=fig,
                structure=struct_i,
                nn=CrystalNN() if show_bonds is True else show_bonds,
                is_3d=False,
                bond_kwargs=bond_kwargs,
                row=row,
                col=col,
                visible_image_atoms=visible_image_atoms,
                rotation_matrix=rotation_matrix,
                elem_colors=_elem_colors,
            )

        # Plot atoms and vectors
        if show_sites:
            for site_idx, (site, coords) in enumerate(
                zip(struct_i, rotated_coords, strict=False)
            ):
                draw_site(
                    fig,
                    site,
                    coords,
                    site_idx,
                    site_labels,
                    _elem_colors,
                    _atomic_radii,
                    atom_size,
                    scale,
                    {} if show_sites is True else show_sites,
                    is_3d=False,
                    row=row,
                    col=col,
                    name=f"site{site_idx}",
                    hover_text=hover_text,
                )

                # Add vector arrows
                if vector_prop:
                    vector = None
                    if vector_prop in site.properties:
                        vector = np.array(site.properties[vector_prop])
                    elif vector_prop in struct_i.properties:
                        vector = struct_i.properties[vector_prop][site_idx]

                    if vector is not None and np.any(vector):
                        draw_vector(
                            fig,
                            coords,
                            vector,
                            is_3d=False,
                            arrow_kwargs=(vector_kwargs or {}).get(vector_prop, {}),
                            row=row,
                            col=col,
                            name=f"vector{site_idx}",
                        )

                # Add image sites
                if show_image_sites:
                    image_atoms = get_image_sites(site, struct_i.lattice)
                    if len(image_atoms) > 0:
                        rotated_image_atoms = np.dot(image_atoms, rotation_matrix)

                        for image_coords in rotated_image_atoms:
                            draw_site(
                                fig,
                                site,
                                image_coords,
                                site_idx,
                                site_labels,
                                _elem_colors,
                                _atomic_radii,
                                atom_size,
                                scale,
                                {} if show_image_sites is True else show_image_sites,
                                is_image=True,
                                is_3d=False,
                                row=row,
                                col=col,
                            )

        # Plot unit cell
        if show_unit_cell:
            draw_unit_cell(
                fig,
                struct_i,
                unit_cell_kwargs={} if show_unit_cell is True else show_unit_cell,
                is_3d=False,
                row=row,
                col=col,
                rotation_matrix=rotation_matrix,
            )

        # Set subplot titles
        anno = get_subplot_title(struct_i, struct_key, idx, subplot_title)
        subtitle_y_pos = 1 - (row - 1) / n_rows - 0.02
        fig.layout.annotations[idx - 1].update(
            **dict(y=subtitle_y_pos, yanchor="top") | anno
        )

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
    fig.update_xaxes(**common_kwargs)
    fig.update_yaxes(**common_kwargs)

    # Need to set scaleanchor on each subplot individually to keep them individually
    # pan and zoomable, else zooming in on one will zoom all
    for idx in range(1, n_structs + 1):
        key = idx if idx > 1 else ""
        fig.layout[f"xaxis{key}"].scaleanchor = f"y{key}"
        fig.layout[f"yaxis{key}"].scaleanchor = f"x{key}"

    return fig


def structure_3d_plotly(
    struct: Structure
    | Sequence[Structure]
    | ase.atoms.Atoms
    | Sequence[ase.atoms.Atoms],
    *,
    atomic_radii: float | dict[str, float] | None = None,
    atom_size: float = 20,
    elem_colors: ElemColorScheme | dict[str, ColorType] = ElemColorScheme.jmol,
    scale: float = 1,
    show_unit_cell: bool | dict[str, Any] = True,
    show_sites: bool | dict[str, Any] = True,
    show_image_sites: bool | dict[str, Any] = True,
    show_bonds: bool | NearNeighbors = False,
    site_labels: Literal["symbol", "species", False]
    | dict[str, str]
    | Sequence[str] = "species",
    standardize_struct: bool | None = None,
    n_cols: int = 3,
    subplot_title: Callable[[Structure, str | int], str | dict[str, Any]]
    | None
    | Literal[False] = None,
    show_site_vectors: str | Sequence[str] = ("force", "magmom"),
    vector_kwargs: dict[str, dict[str, Any]] | None = None,
    hover_text: SiteCoords
    | Callable[[PeriodicSite], str] = SiteCoords.cartesian_fractional,
    bond_kwargs: dict[str, Any] | None = None,
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
            a dict, will be used to customize unit cell appearance. The dict should have
            a "node"/"edge" key to customize node/edge appearance. Defaults to True.
        show_sites (bool | dict[str, Any], optional): Whether to plot atomic sites. If
            a dict, will be used to customize site marker appearance. Defaults to True.
        show_image_sites (bool | dict[str, Any], optional): Whether to show image sites
            on unit cell edges and surfaces. If a dict, will be used to customize how
            image sites are rendered. Defaults to True.
        show_bonds (bool | NearNeighbors, optional): Whether to draw bonds between
            sites. If True, uses CrystalNN with a search_cutoff of 10 Å to determine
            nearest neighbors, including those in neighboring cells. If a NearNeighbors
            object, uses that to determine nearest neighbors.
            Defaults to False (since still experimental).
        site_labels ("symbol" | "species" | dict[str, str] | Sequence):
            How to annotate lattice sites. Defaults to "species".
        standardize_struct (bool, optional): Whether to standardize the structure.
            Defaults to None.
        n_cols (int, optional): Number of columns for subplots. Defaults to 3.
        subplot_title (Callable[[Structure, str | int], str | dict] | False, optional):
            Function to generate subplot titles. Defaults to
            lambda struct_i, idx: f"{idx}. {struct_i.formula} (spg={spg_num})". Set to
            False to hide all subplot titles.
        show_site_vectors (str | Sequence[str], optional): Whether to show vector site
            quantities such as forces or magnetic moments as arrow heads originating
            from each site. Pass the key (or sequence of keys) to look for in site
            properties. Defaults to ("force", "magmom"). If not found as a site
            property, will look for it in the structure properties as well and assume
            the key points at a (N, 3) array with N the number of sites. If multiple
            keys are provided, it plots the first key found in site properties or
            structure properties in any of the passed structures (if a dict of
            structures was passed). But it will only plot one vector per site and it
            will use the same key for all sites and across all structures.
        vector_kwargs (dict[str, dict[str, Any]], optional): For customizing vector
            arrows. Keys are property names (e.g., "force", "magmom"), values are
            dictionaries of arrow customization options. Use key "scale" to adjust
            vector length.
        hover_text (SiteCoords | Callable, optional): Controls the hover tooltip
            template. Can be SiteCoords.cartesian, SiteCoords.fractional,
            SiteCoords.cartesian_fractional, or a callable that takes a site and
            returns a custom string. Defaults to SiteCoords.cartesian_fractional.
        bond_kwargs (dict[str, Any], optional): For customizing bond lines. Keys are
            line properties (e.g., "color", "width"), values are the corresponding
            values. Defaults to None.

    Returns:
        go.Figure: Plotly figure showing the 3D structure(s).
    """
    structures = get_structures(struct)

    n_structs = len(structures)
    n_cols = min(n_cols, n_structs)
    n_rows = (n_structs - 1) // n_cols + 1

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=[[{"type": "scene"} for _ in range(n_cols)] for _ in range(n_rows)],
        # needed to avoid IndexError on fig.layout.annotations[idx - 1].update(anno)
        subplot_titles=[" " for _ in range(n_structs)],
    )

    _elem_colors = get_elem_colors(elem_colors)
    _atomic_radii = get_atomic_radii(atomic_radii)

    if isinstance(show_site_vectors, str):
        show_site_vectors = [show_site_vectors]

    # Determine which vector property to plot (calling outside loop ensures we plot the
    # same prop for all sites in all structures)
    vector_prop = get_first_matching_site_prop(
        list(structures.values()),
        show_site_vectors,
        warn_if_none=show_site_vectors != ("force", "magmom"),
        # check that value is actually a 3-component vector. needs to handle both
        # (N, 3) and (3,) cases
        filter_callback=lambda _prop, value: (np.array(value).shape or [None])[-1] == 3,
    )

    for idx, (struct_key, struct_i) in enumerate(structures.items(), start=1):
        # Standardize structure if needed
        if standardize_struct is None:
            standardize_struct = any(any(site.frac_coords < 0) for site in struct_i)
        if standardize_struct:
            try:
                spg_analyzer = SpacegroupAnalyzer(struct_i)
                struct_i = spg_analyzer.get_conventional_standard_structure()  # noqa: PLW2901
            except ValueError:
                warnings.warn(NO_SYM_MSG, UserWarning, stacklevel=2)

        visible_image_atoms: set[tuple[float, float, float]] = set()

        # First, populate visible_image_atoms with all image sites that will be drawn
        if show_image_sites and show_sites:
            for site in struct_i:
                image_atoms = get_image_sites(site, struct_i.lattice)
                if len(image_atoms) > 0:
                    for image_coords in image_atoms:
                        visible_image_atoms.add(tuple(image_coords))

        # Draw bonds
        if show_bonds:
            draw_bonds(
                fig=fig,
                structure=struct_i,
                nn=CrystalNN() if show_bonds is True else show_bonds,
                is_3d=True,
                bond_kwargs=bond_kwargs,
                scene=f"scene{idx}",
                visible_image_atoms=visible_image_atoms,
                elem_colors=_elem_colors,
            )

        # Plot atoms and vectors
        if show_sites:
            for site_idx, site in enumerate(struct_i):
                draw_site(
                    fig,
                    site,
                    site.coords,
                    site_idx,
                    site_labels,
                    _elem_colors,
                    _atomic_radii,
                    atom_size,
                    scale,
                    {} if show_sites is True else show_sites,
                    is_3d=True,
                    scene=f"scene{idx}",
                    name=f"site{site_idx}",
                    hover_text=hover_text,
                )

                # Add vector arrows
                if vector_prop:
                    vector = None
                    if vector_prop in site.properties:
                        vector = np.array(site.properties[vector_prop])
                    elif vector_prop in struct_i.properties:
                        vector = struct_i.properties[vector_prop][site_idx]

                    if vector is not None and np.any(vector):
                        draw_vector(
                            fig,
                            site.coords,
                            vector,
                            is_3d=True,
                            arrow_kwargs=(vector_kwargs or {}).get(vector_prop, {}),
                            scene=f"scene{idx}",
                            name=f"vector{site_idx}",
                        )

                # Add image sites
                if show_image_sites:
                    image_atoms = get_image_sites(site, struct_i.lattice)
                    if len(image_atoms) > 0:
                        for image_coords in image_atoms:
                            draw_site(
                                fig,
                                site,
                                image_coords,
                                site_idx,
                                site_labels,
                                _elem_colors,
                                _atomic_radii,
                                atom_size,
                                scale,
                                {} if show_image_sites is True else show_image_sites,
                                is_image=True,
                                is_3d=True,
                                scene=f"scene{idx}",
                            )

        # Plot unit cell
        if show_unit_cell:
            draw_unit_cell(
                fig,
                struct_i,
                unit_cell_kwargs={} if show_unit_cell is True else show_unit_cell,
                is_3d=True,
                scene=f"scene{idx}",
            )

        # Set subplot titles
        if subplot_title is not False:
            anno = get_subplot_title(struct_i, struct_key, idx, subplot_title)
            if "y" not in anno:
                row = (idx - 1) // n_cols + 1
                subtitle_y_pos = 1 - (row - 1) / n_rows - 0.02
                anno["y"] = subtitle_y_pos
            if "yanchor" not in anno:
                anno["yanchor"] = "top"
            fig.layout.annotations[idx - 1].update(anno)

        # Update 3D scene properties
        no_axes_kwargs = dict(
            showticklabels=False, showgrid=False, zeroline=False, visible=False
        )

        fig.update_scenes(
            xaxis=no_axes_kwargs,
            yaxis=no_axes_kwargs,
            zaxis=no_axes_kwargs,
            aspectmode="data",
            bgcolor="rgba(90, 90, 90, 0.01)",  # Transparent background
        )

    # Calculate subplot positions with small gap
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
    fig.layout.height = 400 * n_rows
    fig.layout.width = 400 * n_cols
    fig.layout.showlegend = False
    fig.layout.paper_bgcolor = "rgba(0,0,0,0)"  # Transparent background
    fig.layout.plot_bgcolor = "rgba(0,0,0,0)"  # Transparent background
    fig.layout.margin = dict(l=0, r=0, t=0, b=0)  # Minimize margins

    return fig
