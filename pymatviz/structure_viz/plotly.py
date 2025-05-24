"""Create interactive hoverable 2D and 3D plots of pymatgen structures with plotly."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pymatgen.analysis.local_env import CrystalNN, NearNeighbors

from pymatviz.enums import ElemColorScheme, SiteCoords
from pymatviz.process_data import normalize_structures
from pymatviz.structure_viz.helpers import (
    _angles_to_rotation_matrix,
    _draw_element_legend,
    _get_site_symbol,
    _prep_augmented_structure_for_bonding,
    _standardize_struct,
    draw_bonds,
    draw_site,
    draw_unit_cell,
    draw_vector,
    generate_site_label,
    get_atomic_radii,
    get_elem_colors,
    get_first_matching_site_prop,
    get_image_sites,
    get_site_hover_text,
    get_subplot_title,
)
from pymatviz.typing import BOTTOM_RIGHT
from pymatviz.utils import pick_max_contrast_color


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any, Literal

    import ase.atoms
    from pymatgen.core import PeriodicSite, Structure

    from pymatviz.typing import ColorType


def structure_2d_plotly(
    struct: Structure
    | Sequence[Structure]
    | ase.Atoms
    | Sequence[ase.Atoms]
    | dict[str, Structure | ase.Atoms],
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
    site_labels: Literal["symbol", "species", "legend", False]
    | dict[str, str]
    | Sequence[str] = "legend",
    standardize_struct: bool | None = None,
    n_cols: int = 3,
    subplot_title: Callable[[Structure, str | int], str | dict[str, Any]] | None = None,
    show_site_vectors: str | Sequence[str] = ("force", "magmom"),
    vector_kwargs: dict[str, dict[str, Any]] | None = None,
    hover_text: SiteCoords
    | Callable[[PeriodicSite], str] = SiteCoords.cartesian_fractional,
    hover_float_fmt: str | Callable[[float], str] = ".4",
    bond_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
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
        site_labels ("symbol" | "species" | "legend" | False | dict | Sequence):
            How to annotate lattice sites. Defaults to "legend" (show an element
            color legend). Other options: "symbol", "species", False (no labels),
            a dict mapping site index to label, or a sequence of labels.
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
        hover_float_fmt (str | Callable[[float], str], optional): Float formatting for
            hover coordinates. Can be an f-string format like ".4" (default) or a
            callable that takes a float and returns a string. Defaults to ".4".
        bond_kwargs (dict[str, Any], optional): For customizing bond lines. Keys are
            line properties (e.g., "color", "width"), values are the corresponding
            values. Defaults to None.
        legend_kwargs (dict[str, Any], optional): Controls for the element legend,
            active if site_labels="legend". Keys can be "font_size" (default 12),
            "box_size_px" (default 18), "item_gap_px" (default 3),
            "margin_frac" (default 0.04), and "corner" (default "bottom-right").

    Returns:
        go.Figure: Plotly figure showing the 2D structure(s).
    """
    structures = normalize_structures(struct)

    n_structs = len(structures)
    n_cols = min(n_cols, n_structs)
    n_rows = math.ceil(n_structs / (n_cols or 1))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        # Avoid IndexError on fig.layout.annotations[idx - 1].update(anno)
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
        filter_callback=lambda _prop, value: (np.array(value).shape or [None])[-1] == 3,
    )

    legends_to_draw = []

    for idx, (struct_key, raw_struct_i) in enumerate(structures.items(), start=1):
        row = (idx - 1) // n_cols + 1
        col = (idx - 1) % n_cols + 1

        struct_i = _standardize_struct(raw_struct_i, standardize_struct)

        rotation_matrix = _angles_to_rotation_matrix(rotation)
        rotated_coords_all_sites = np.dot(struct_i.cart_coords, rotation_matrix)

        # For bonding, consider primary and image sites if show_image_sites is active
        # The actual plotting of image sites is handled later by draw_site if show_sites
        augmented_structure = _prep_augmented_structure_for_bonding(
            struct_i, show_image_sites and show_sites
        )
        if show_bonds:
            draw_bonds(
                fig=fig,
                structure=augmented_structure,  # Pass augmented structure
                nn=CrystalNN() if show_bonds is True else show_bonds,
                is_3d=False,
                bond_kwargs=bond_kwargs,
                row=row,
                col=col,
                rotation_matrix=rotation_matrix,
                elem_colors=_elem_colors,
            )

        if site_labels == "legend":
            legend_item = dict(
                struct=struct_i, colors=_elem_colors, subplot_idx=idx, is_3d=False
            )
            legends_to_draw.append(legend_item)

        if show_sites:  # Plot atoms, vectors, and image sites
            for site_idx_loop, (site, rotated_site_coords_3d) in enumerate(
                zip(struct_i, rotated_coords_all_sites, strict=False)
            ):
                draw_site(  # Draw primary site
                    fig=fig,
                    site=site,
                    coords=rotated_site_coords_3d,  # Pass 3D rotated coords
                    site_idx=site_idx_loop,
                    site_labels=site_labels,
                    elem_colors=_elem_colors,
                    atomic_radii=_atomic_radii,
                    atom_size=atom_size,
                    scale=scale,
                    site_kwargs={} if show_sites is True else show_sites,
                    is_3d=False,  # draw_site will project to 2D
                    row=row,
                    col=col,
                    name=f"site-{struct_key}-{site_idx_loop}",
                    hover_text=hover_text,
                    float_fmt=hover_float_fmt,
                )

                if vector_prop:  # Add vector arrows for the primary site
                    vector = None
                    if vector_prop in site.properties:
                        vector = np.array(site.properties[vector_prop])
                    # Ensure site_idx_loop is valid for struct_i.properties[vector_prop]
                    elif vector_prop in struct_i.properties and site_idx_loop < len(
                        struct_i.properties[vector_prop]
                    ):
                        vector = struct_i.properties[vector_prop][site_idx_loop]

                    if vector is not None and np.any(vector):
                        # Rotate the vector for 2D projection
                        rotated_vector = np.dot(vector, rotation_matrix)
                        draw_vector(
                            fig,
                            rotated_site_coords_3d,
                            rotated_vector,
                            is_3d=False,
                            arrow_kwargs=(vector_kwargs or {}).get(vector_prop, {}),
                            row=row,
                            col=col,
                            name=f"vector-{struct_key}-{site_idx_loop}",
                        )

                # Add image sites for the current primary site
                # This uses the global show_image_sites argument.
                if show_image_sites:
                    image_cart_coords_arrays = get_image_sites(site, struct_i.lattice)
                    if len(image_cart_coords_arrays) > 0:
                        rotated_image_atoms_coords_3d_list = np.dot(
                            image_cart_coords_arrays, rotation_matrix
                        )
                        for image_idx, current_rotated_image_coords_3d in enumerate(
                            rotated_image_atoms_coords_3d_list
                        ):
                            draw_site(
                                fig=fig,
                                site=site,
                                coords=current_rotated_image_coords_3d,
                                site_idx=site_idx_loop,
                                site_labels=site_labels,
                                elem_colors=_elem_colors,
                                atomic_radii=_atomic_radii,
                                atom_size=atom_size,
                                scale=scale,
                                site_kwargs={}
                                if show_image_sites is True
                                else show_image_sites,
                                is_image=True,
                                is_3d=False,
                                row=row,
                                col=col,
                                name=f"image-{struct_key}-{site_idx_loop}-{image_idx}",
                                float_fmt=hover_float_fmt,
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
    fig.layout.margin = dict(l=0, r=0, t=30, b=0)
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

    legend_defaults = dict(
        font_size=12,
        box_size_px=18,
        item_gap_px=3,
        margin_frac=0.04,
        corner=BOTTOM_RIGHT,
    )

    for legend_data in legends_to_draw:
        _draw_element_legend(
            fig=fig,
            struct_i=legend_data["struct"],
            _elem_colors=legend_data["colors"],
            subplot_idx=legend_data["subplot_idx"],
            is_3d=legend_data["is_3d"],
            **legend_defaults | (legend_kwargs or {}),
        )

    return fig


def structure_3d_plotly(
    struct: Structure
    | Sequence[Structure]
    | ase.Atoms
    | Sequence[ase.Atoms]
    | dict[str, Structure | ase.Atoms],
    *,
    atomic_radii: float | dict[str, float] | None = None,
    atom_size: float = 20,
    elem_colors: ElemColorScheme | dict[str, ColorType] = ElemColorScheme.jmol,
    scale: float = 1,
    show_unit_cell: bool | dict[str, Any] = True,
    show_sites: bool | dict[str, Any] = True,
    show_image_sites: bool | dict[str, Any] = True,
    show_bonds: bool | NearNeighbors = False,
    site_labels: Literal["symbol", "species", "legend", False]
    | dict[str, str]
    | Sequence[str] = "legend",
    standardize_struct: bool | None = None,
    n_cols: int = 3,
    subplot_title: Callable[[Structure, str | int], str | dict[str, Any]]
    | None
    | Literal[False] = None,
    show_site_vectors: str | Sequence[str] = ("force", "magmom"),
    vector_kwargs: dict[str, dict[str, Any]] | None = None,
    hover_text: SiteCoords
    | Callable[[PeriodicSite], str] = SiteCoords.cartesian_fractional,
    hover_float_fmt: str | Callable[[float], str] = ".4",
    bond_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
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
        site_labels ("symbol" | "species" | "legend" | False | dict | Sequence):
            How to annotate lattice sites. Defaults to "legend" (show an element
            color legend). Other options: "symbol", "species", False (no labels),
            a dict mapping site index to label, or a sequence of labels.
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
        hover_float_fmt (str | Callable[[float], str], optional): Float formatting for
            hover coordinates. Can be an f-string format like ".4" (default) or a
            callable that takes a float and returns a string. Defaults to ".4".
        bond_kwargs (dict[str, Any], optional): For customizing bond lines. Keys are
            line properties (e.g., "color", "width"), values are the corresponding
            values. Defaults to None.
        legend_kwargs (dict[str, Any], optional): Controls for the element legend,
            active if site_labels="legend". Keys can be "font_size" (default 12),
            "box_size_px" (default 18), "item_gap_px" (default 3),
            "margin_frac" (default 0.04), and "corner" (default "bottom-right").

    Returns:
        go.Figure: Plotly figure showing the 3D structure(s).
    """
    structures = normalize_structures(struct)

    n_structs = len(structures)
    n_cols = min(n_cols, n_structs)
    n_rows = (n_structs - 1) // n_cols + 1

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=[[{"type": "scene"} for _ in range(n_cols)] for _ in range(n_rows)],
        # Avoid IndexError on fig.layout.annotations[idx - 1].update(anno)
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
        filter_callback=lambda _prop, value: (np.array(value).shape or [None])[-1] == 3,
    )

    legends_to_draw = []

    for idx, (struct_key, raw_struct_i) in enumerate(structures.items(), start=1):
        struct_i = _standardize_struct(raw_struct_i, standardize_struct)

        # Prepare augmented structure: original + image sites for consistent processing
        # This augmented_structure is used for collecting all site data for the single
        # 3D trace and for bond calculations.
        augmented_structure = _prep_augmented_structure_for_bonding(
            struct_i, show_image_sites and show_sites
        )

        all_site_x, all_site_y, all_site_z = [], [], []
        all_site_colors, all_site_sizes, all_site_hover_texts = [], [], []
        all_site_labels_list: list[str | None] = []
        all_site_textfont_colors = []

        # Plot atoms and vectors
        if show_sites:
            # Iterate over all sites in the augmented_structure (primary + images)
            for site_idx_loop, site in enumerate(augmented_structure.sites):
                all_site_x.append(site.coords[0])
                all_site_y.append(site.coords[1])
                all_site_z.append(site.coords[2])

                symbol = _get_site_symbol(site)
                site_base_color = _elem_colors.get(symbol, "gray")

                # Convert color to string format
                if isinstance(site_base_color, tuple) and len(site_base_color) == 3:
                    r, g, b = (
                        int(c * 255) if isinstance(c, float) and 0 <= c <= 1 else int(c)
                        for c in site_base_color
                    )
                    display_color_str = f"rgb({r},{g},{b})"
                else:
                    display_color_str = (
                        str(site_base_color)
                        if isinstance(site_base_color, str)
                        else "rgb(128,128,128)"
                    )

                all_site_colors.append(display_color_str)
                all_site_textfont_colors.append(
                    pick_max_contrast_color(display_color_str)
                )

                radius = _atomic_radii.get(symbol, 1) * scale
                all_site_sizes.append(radius * atom_size)

                # Use helper for hover text
                all_site_hover_texts.append(
                    get_site_hover_text(site, hover_text, site.species, hover_float_fmt)
                )

                # Use helper for site label
                if site_labels == "legend":
                    all_site_labels_list.append(None)
                else:
                    all_site_labels_list.append(
                        generate_site_label(site_labels, site_idx_loop, site)
                    )

            site_kwargs = {} if show_sites is True else show_sites
            marker_defaults = {
                "size": all_site_sizes,
                "color": all_site_colors,
                "sizemode": "diameter",
            }
            if "marker" in site_kwargs:
                site_kwargs["marker"].update(marker_defaults)
            else:
                site_kwargs["marker"] = marker_defaults

            trace_3d = go.Scatter3d(
                x=all_site_x,
                y=all_site_y,
                z=all_site_z,
                mode="markers" + ("+text" if any(all_site_labels_list) else ""),
                text=all_site_labels_list if any(all_site_labels_list) else None,
                textposition="top center",
                textfont={"color": all_site_textfont_colors},
                hovertext=all_site_hover_texts,
                hoverinfo="text",
                name=f"site-{struct_key}",
                **site_kwargs,
            )
            fig.add_trace(
                trace_3d, row=(idx - 1) // n_cols + 1, col=(idx - 1) % n_cols + 1
            )

            for site_idx_loop, site_in_original_struct in enumerate(struct_i):
                if vector_prop:
                    vector = None
                    # Check properties on the original site object
                    if vector_prop in site_in_original_struct.properties:
                        vector = np.array(
                            site_in_original_struct.properties[vector_prop]
                        )
                    # Check structure-level properties, using original site_idx_loop
                    elif vector_prop in struct_i.properties and site_idx_loop < len(
                        struct_i.properties[vector_prop]
                    ):
                        vector = struct_i.properties[vector_prop][site_idx_loop]

                    if vector is not None and np.any(vector):
                        draw_vector(
                            fig,
                            site_in_original_struct.coords,
                            vector,
                            is_3d=True,
                            arrow_kwargs=(vector_kwargs or {}).get(vector_prop, {}),
                            scene=f"scene{idx}",
                            name=f"vector{site_idx_loop}",
                        )

        if site_labels == "legend":
            legend_item = dict(
                struct=struct_i, colors=_elem_colors, subplot_idx=idx, is_3d=True
            )
            legends_to_draw.append(legend_item)

        # Draw bonds using the augmented structure after sites are processed
        if show_bonds:
            plotted_sites_coords: set[tuple[float, float, float]] | None = None
            if show_sites and all_site_x:  # Ensure all_site_x is populated
                # Rounded to 5 decimal places for robust comparison
                plotted_sites_coords = {
                    tuple(np.round(coord, 5))
                    for coord in zip(all_site_x, all_site_y, all_site_z, strict=False)
                }

            draw_bonds(
                fig=fig,
                structure=augmented_structure,  # Pass the augmented structure
                nn=CrystalNN() if show_bonds is True else show_bonds,
                is_3d=True,
                bond_kwargs=bond_kwargs,
                scene=f"scene{idx}",
                elem_colors=_elem_colors,
                plotted_sites_coords=plotted_sites_coords,
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
    fig.layout.margin = dict(l=0, r=0, t=30, b=0)  # Minimize margins

    legend_defaults = dict(
        font_size=12,
        box_size_px=18,
        item_gap_px=3,
        margin_frac=0.04,
        corner=BOTTOM_RIGHT,
    )
    for legend_data in legends_to_draw:
        _draw_element_legend(
            fig=fig,
            struct_i=legend_data["struct"],
            _elem_colors=legend_data["colors"],
            subplot_idx=legend_data["subplot_idx"],
            is_3d=legend_data["is_3d"],
            **legend_defaults | (legend_kwargs or {}),
        )

    return fig
