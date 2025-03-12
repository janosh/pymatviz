"""Helper functions for 2D and 3D plots of pymatgen structures with plotly."""

from __future__ import annotations

import functools
import itertools
import math
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Lattice, PeriodicSite, Species, Structure

from pymatviz.colors import ELEM_COLORS_ALLOY, ELEM_COLORS_JMOL, ELEM_COLORS_VESTA
from pymatviz.enums import ElemColorScheme, Key, SiteCoords
from pymatviz.utils import df_ptable
from pymatviz.utils.plotting import pick_max_contrast_color


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import Any, Literal

    import plotly.graph_objects as go
    from numpy.typing import ArrayLike
    from pymatgen.analysis.local_env import NearNeighbors

    from pymatviz.typing import ColorType


# fallback value (in nanometers) for covalent radius of an element
# see https://wikipedia.org/wiki/Atomic_radii_of_the_elements
missing_covalent_radius = 0.2
covalent_radii: pd.Series = df_ptable[Key.covalent_radius].fillna(
    missing_covalent_radius
)
NO_SYM_MSG = "Symmetry could not be determined, skipping standardization"
UNIT_CELL_EDGES = (
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
)


def _angles_to_rotation_matrix(
    angles: str, rotation: ArrayLike | None = None
) -> ArrayLike:
    """Convert Euler angles to a rotation matrix.

    Note the order of angles matters. 50x,40z != 40z,50x.

    Args:
        angles (str): Euler angles (in degrees) formatted as '-10y,50x,120z'
        rotation (np.array, optional): Initial rotation matrix. Use this if you already
            have a rotation and want to combine it with the rotation defined by angles.
            Defaults to identity matrix np.eye(3).

    Returns:
        np.array: 3d rotation matrix.
    """
    if rotation is None:
        rotation = np.eye(3)

    # Return initial rotation matrix if no angles
    if not angles:
        return rotation.copy()

    for angle in angles.split(","):
        radians = math.radians(float(angle[:-1]))
        xyz = angle[-1]
        dim = "xyz".index(xyz)
        sin = math.sin(radians)
        cos = math.cos(radians)
        if dim == 0:
            rotation = np.dot(rotation, [(1, 0, 0), (0, cos, sin), (0, -sin, cos)])
        elif dim == 1:
            rotation = np.dot(rotation, [(cos, 0, -sin), (0, 1, 0), (sin, 0, cos)])
        else:
            rotation = np.dot(rotation, [(cos, sin, 0), (-sin, cos, 0), (0, 0, 1)])
    return rotation


def get_image_sites(
    site: PeriodicSite, lattice: Lattice, tol: float = 0.03
) -> np.ndarray:
    """Get images for a given site in a lattice.

    Images are sites that are integer translations of the given site that are within a
    tolerance of the unit cell edges.

    Args:
        site (PeriodicSite): The site to get images for.
        lattice (Lattice): The lattice to get images for.
        tol (float): The tolerance for being on the unit cell edge. Defaults to 0.02.

    Returns:
        np.ndarray: Coordinates of all image sites.
    """
    coords_image_atoms: list[np.ndarray] = []

    # Generate all possible combinations of lattice vector offsets (except zero offset)
    offsets = set(itertools.product([-1, 0, 1], repeat=3)) - {(0, 0, 0)}

    for offset in offsets:
        new_frac = site.frac_coords + offset
        new_cart = lattice.get_cartesian_coords(new_frac)

        is_within_cell = all(-tol <= coord <= 1 + tol for coord in new_frac)
        is_on_edge = any(
            np.isclose(new_frac, 0, atol=tol) | np.isclose(new_frac, 1, atol=tol)
        )

        if is_within_cell and is_on_edge:
            coords_image_atoms += [new_cart]

    return np.array(coords_image_atoms)


def unit_cell_to_lines(cell: ArrayLike) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Convert lattice vectors to plot lines.

    Args:
        cell (np.array): Lattice vectors.

    Returns:
        tuple[np.array, np.array, np.array]:
        - Lines
        - z-indices that sort plot elements into out-of-plane layers
        - lines used to plot the unit cell
    """
    n_lines = n1 = 0
    segments = []
    for idx in range(3):
        norm = math.sqrt(sum(cell[idx] ** 2))
        segment = max(2, int(norm / 0.3))
        segments += [segment]
        n_lines += 4 * segment

    lines = np.empty((n_lines, 3))
    z_indices = np.empty(n_lines, dtype=int)
    unit_cell_lines = np.zeros((3, 3))

    for idx in range(3):
        segment = segments[idx]
        dd = cell[idx] / (4 * segment - 2)
        unit_cell_lines[idx] = dd
        point_array = np.arange(1, 4 * segment + 1, 4)[:, None] * dd
        z_indices[n1:] = idx
        for ii, jj in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            n2 = n1 + segment
            lines[n1:n2] = point_array + ii * cell[idx - 2] + jj * cell[idx - 1]
            n1 = n2

    return lines, z_indices, unit_cell_lines


def get_elem_colors(
    elem_colors: ElemColorScheme | Mapping[str, ColorType],
) -> dict[str, ColorType]:
    """Get element colors based on the provided scheme or custom dictionary."""
    if isinstance(elem_colors, dict):
        return elem_colors
    if str(elem_colors) == str(ElemColorScheme.jmol):
        return ELEM_COLORS_JMOL  # type: ignore[return-value]
    if str(elem_colors) == str(ElemColorScheme.vesta):
        return ELEM_COLORS_VESTA  # type: ignore[return-value]
    if str(elem_colors) == str(ElemColorScheme.alloy):
        return ELEM_COLORS_ALLOY  # type: ignore[return-value]
    raise ValueError(
        f"colors must be a dict or one of ('{', '.join(ElemColorScheme)}')"
    )


def get_atomic_radii(atomic_radii: float | dict[str, float] | None) -> dict[str, float]:
    """Get atomic radii based on the provided input."""
    if atomic_radii is None or isinstance(atomic_radii, float):
        scale = atomic_radii or 1
        return {elem: radius * scale for elem, radius in covalent_radii.items()}
    return atomic_radii


def generate_site_label(
    site_labels: Literal["symbol", "species", False] | dict[str, str] | Sequence[str],
    site_idx: int,
    majority_species: Species,
) -> str:
    """Generate a label for a site based on the provided labeling scheme."""
    if site_labels == "symbol":
        return str(majority_species.symbol)
    if site_labels == "species":
        return str(majority_species)
    if site_labels is False:
        return ""
    if isinstance(site_labels, dict):
        return site_labels.get(majority_species.symbol, "")
    if isinstance(site_labels, list | tuple):
        return site_labels[site_idx]
    raise ValueError(
        f"Invalid {site_labels=}. Must be one of "
        f"('symbol', 'species', False, dict, list)"
    )


def get_subplot_title(
    struct_i: Structure,
    struct_key: Any,
    idx: int,
    subplot_title: Callable[[Structure, str | int], str | dict[str, Any]] | None,
) -> dict[str, Any]:
    """Generate a subplot title based on the provided function or default logic."""
    title_dict: dict[str, str | float | dict[str, str | float]] = {}

    if callable(subplot_title):
        sub_title = subplot_title(struct_i, struct_key)
        if isinstance(sub_title, str | int | float):
            title_dict["text"] = str(sub_title)
        elif isinstance(sub_title, dict):
            title_dict |= sub_title
        else:
            raise TypeError(
                f"Invalid subplot_title, must be str or dict, got {sub_title}"
            )

    if not title_dict.get("text"):
        if isinstance(struct_key, int):
            spg_num = struct_i.get_symmetry_dataset()["number"]
            title_dict["text"] = f"{idx}. {struct_i.formula} (spg={spg_num})"
        elif isinstance(struct_key, str):
            title_dict["text"] = str(struct_key)
        else:
            raise TypeError(f"Invalid {struct_key=}. Must be an int or str.")

    return title_dict


def get_site_hover_text(
    site: PeriodicSite,
    hover_text: SiteCoords | Callable[[PeriodicSite], str],
    majority_species: Species,
) -> str:
    """Generate hover text for a site based on the hover template."""
    if callable(hover_text):
        return hover_text(site)

    cart_text = f"({', '.join(f'{c:.3g}' for c in site.coords)})"
    frac_text = f"[{', '.join(f'{c:.3g}' for c in site.frac_coords)}]"
    if hover_text == SiteCoords.cartesian:
        coords_text = cart_text
    elif hover_text == SiteCoords.fractional:
        coords_text = frac_text
    elif hover_text == SiteCoords.cartesian_fractional:
        coords_text = f"{cart_text} {frac_text}"
    else:
        raise ValueError(f"Invalid {hover_text=}")

    out_text = f"<b>Site: {majority_species}</b><br>Coordinates {coords_text}"

    if site.properties:
        out_text += "<br>Properties: " + ", ".join(
            f"{k}: {v}" for k, v in site.properties.items()
        )
    return out_text


def draw_site(
    fig: go.Figure,
    site: PeriodicSite,
    coords: np.ndarray,
    site_idx: int,
    site_labels: Any,
    _elem_colors: dict[str, ColorType],
    _atomic_radii: dict[str, float],
    atom_size: float,
    scale: float,
    site_kwargs: dict[str, Any],
    *,
    is_image: bool = False,
    is_3d: bool = False,
    row: int | None = None,
    col: int | None = None,
    scene: str | None = None,
    hover_text: SiteCoords
    | Callable[[PeriodicSite], str] = SiteCoords.cartesian_fractional,
    **kwargs: Any,
) -> None:
    """Add a site (regular or image) to the plot."""
    species = getattr(site, "specie", site.species)
    majority_species = (
        max(species, key=species.get) if isinstance(species, Composition) else species
    )
    site_radius = _atomic_radii[majority_species.symbol] * scale
    color = _elem_colors.get(majority_species.symbol, "gray")

    site_hover_text = get_site_hover_text(site, hover_text, majority_species)

    txt = generate_site_label(site_labels, site_idx, majority_species)

    marker = dict(
        size=site_radius * atom_size,
        color=color,
        opacity=0.8 if is_image else 1,
        line=dict(width=1, color="gray"),
    )
    marker.update(site_kwargs)

    # Calculate text color based on background color for maximum contrast
    text_color = pick_max_contrast_color(color)
    scatter_kwargs = dict(
        x=[coords[0]],
        y=[coords[1]],
        mode="markers+text" if txt else "markers",
        marker=marker,
        text=txt,
        textposition="middle center",
        textfont=dict(
            color=text_color,
            size=np.clip(atom_size * site_radius * (0.8 if is_image else 1), 10, 18),
        ),
        hoverinfo="text" if hover_text else None,
        hovertext=site_hover_text,
        hoverlabel=dict(namelength=-1),
        name=f"Image of {majority_species!s}" if is_image else str(majority_species),
        showlegend=False,
    )
    scatter_kwargs |= kwargs

    if is_3d:
        scatter_kwargs["z"] = [coords[2]]
        fig.add_scatter3d(**scatter_kwargs, scene=scene)
    else:
        fig.add_scatter(**scatter_kwargs, row=row, col=col)


def get_structures(
    struct: Structure | Sequence[Structure] | pd.Series | dict[Any, Structure],
) -> dict[Any, Structure]:
    """Convert pymatgen Structures or ASE Atoms or sequences of either to a dictionary
    of pymatgen Structures.
    """
    if isinstance(struct, Structure):
        return {0: struct}
    if isinstance(struct, pd.Series):
        return struct.to_dict()
    if isinstance(next(iter(struct), None), Structure):
        return dict(enumerate(struct))
    if isinstance(struct, dict) and {*map(type, struct.values())} == {Structure}:
        return struct

    def is_ase_atoms(struct: Any) -> bool:
        """Check if the input is an ASE Atoms object without importing ase."""
        cls_name = f"{type(struct).__module__}.{type(struct).__qualname__}"
        return cls_name in ("ase.atoms.Atoms", "pymatgen.io.ase.MSONAtoms")

    if is_ase_atoms(struct):  # detect single ASE Atoms object
        from pymatgen.io.ase import AseAtomsAdaptor

        return {0: AseAtomsAdaptor().get_structure(struct)}
    if is_ase_atoms(next(iter(struct), None)):  # detect sequence of ASE Atoms
        from pymatgen.io.ase import AseAtomsAdaptor

        return {
            idx: AseAtomsAdaptor().get_structure(atoms)
            for idx, atoms in enumerate(struct)
        }

    raise TypeError(f"Expected pymatgen Structure or Sequence of them, got {struct=}")


def draw_unit_cell(
    fig: go.Figure,
    structure: Structure,
    unit_cell_kwargs: dict[str, Any],
    *,
    is_3d: bool = True,
    row: int | None = None,
    col: int | None = None,
    scene: str | None = None,
    rotation_matrix: np.ndarray | None = None,
) -> go.Figure:
    """Draw the unit cell of a structure in a 2D or 3D Plotly figure."""
    corners = np.array(list(itertools.product((0, 1), (0, 1), (0, 1))))
    cart_corners = structure.lattice.get_cartesian_coords(corners)

    # Apply rotation to cartesian corners for 2D plots
    if not is_3d and rotation_matrix is not None:
        cart_corners = np.dot(cart_corners, rotation_matrix)

    alpha, beta, gamma = structure.lattice.angles

    trace_adder = (  # prefill args for add_scatter or add_scatter3d
        functools.partial(fig.add_scatter3d, scene=scene)
        if is_3d
        else functools.partial(fig.add_scatter, row=row, col=col)
    )

    # Add edges
    edge_defaults = dict(color="black", width=1, dash="dash")
    edge_kwargs = edge_defaults | unit_cell_kwargs.get("edge", {})
    for idx, (start, end) in enumerate(UNIT_CELL_EDGES):
        start_point = cart_corners[start]
        end_point = cart_corners[end]
        mid_point = (start_point + end_point) / 2
        edge_vector = end_point - start_point
        edge_len = np.linalg.norm(edge_vector)

        hover_text = (
            f"Length: {edge_len:.3g} Å<br>"
            f"Start: ({', '.join(f'{c:.3g}' for c in start_point)}) "
            f"[{', '.join(f'{c:.3g}' for c in corners[start])}]<br>"
            f"End: ({', '.join(f'{c:.3g}' for c in end_point)}) "
            f"[{', '.join(f'{c:.3g}' for c in corners[end])}]"
        )

        coords = dict(
            x=[start_point[0], mid_point[0], end_point[0]],
            y=[start_point[1], mid_point[1], end_point[1]],
        )
        if is_3d:
            coords["z"] = [start_point[2], mid_point[2], end_point[2]]
        trace_adder(
            **coords,
            mode="lines",
            line=edge_kwargs,
            hovertext=[None, hover_text, None],
            name=f"edge {idx}",
        )

    # Add corner spheres
    node_defaults = dict(size=3, color="black")
    node_kwargs = node_defaults | unit_cell_kwargs.get("node", {})
    for idx, (frac_coord, cart_coord) in enumerate(
        zip(corners, cart_corners, strict=True)
    ):
        adjacent_angles = []
        for _ in range(3):
            v1 = cart_corners[(idx + 1) % 8] - cart_coord
            v2 = cart_corners[(idx + 2) % 8] - cart_coord
            angle = np.degrees(
                np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            )
            adjacent_angles.append(angle)

        hover_text = (
            f"({', '.join(f'{c:.3g}' for c in cart_coord)}) "
            f"[{', '.join(f'{c:.3g}' for c in frac_coord)}]<br>"
            f"α = {alpha:.3g}°, β = {beta:.3g}°, γ = {gamma:.3g}°"  # noqa: RUF001
        )
        coords = dict(x=[cart_coord[0]], y=[cart_coord[1]])
        if is_3d:
            coords["z"] = [cart_coord[2]]
        trace_adder(
            **coords,
            mode="markers",
            marker=node_kwargs,
            hovertext=hover_text,
            name=f"node {idx}",
        )

    return fig


def draw_vector(
    fig: go.Figure,
    start: np.ndarray,
    vector: np.ndarray,
    *,
    is_3d: bool = False,
    arrow_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> None:
    """Add an arrow to represent a vector quantity on a Plotly figure.

    This function adds an arrow to a 2D or 3D Plotly figure to represent a vector
    quantity. In 3D, it uses a cone for the arrowhead and a line for the shaft.
    In 2D, it uses a scatter plot with an arrow marker.

    Args:
        fig (go.Figure): The Plotly figure to add the arrow to.
        start (np.ndarray): The starting point of the arrow (shape: (3,) for 3D,
            (2,) for 2D).
        vector (np.ndarray): The vector to be represented by the arrow (shape: (3,) for
            3D, (2,) for 2D).
        is_3d (bool, optional): Whether to add a 3D arrow. Defaults to False.
        arrow_kwargs (dict[str, Any] | None, optional): Additional keyword arguments
            for arrow customization. Supported keys:
            - color (str): Color of the arrow.
            - width (float): Width of the arrow shaft.
            - arrow_head_length (float): Length of the arrowhead (3D only).
            - arrow_head_angle (float): Angle of the arrowhead in degrees (3D only).
            - scale (float): Scaling factor for the vector length.
        **kwargs: Additional keyword arguments passed to the Plotly trace.

    Note:
        For 3D arrows, this function adds two traces to the figure: a cone for the
        arrowhead and a line for the shaft. For 2D arrows, it adds a single scatter
        trace with an arrow marker.
    """
    default_arrow_kwargs = dict(
        color="white",
        width=5,
        arrow_head_length=0.8,
        arrow_head_angle=30,
        scale=1.0,
    )
    arrow_kwargs = default_arrow_kwargs | (arrow_kwargs or {})

    # Apply scaling to the vector
    scaled_vector = vector * arrow_kwargs["scale"]
    end = start + scaled_vector

    if is_3d:
        # Add the shaft (line) first
        fig.add_scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode="lines",
            line=dict(color=arrow_kwargs["color"], width=arrow_kwargs["width"]),
            showlegend=False,
            **kwargs,
        )
        # Add the arrowhead (cone) at the end
        fig.add_cone(
            x=[end[0]],
            y=[end[1]],
            z=[end[2]],
            u=[scaled_vector[0]],
            v=[scaled_vector[1]],
            w=[scaled_vector[2]],
            colorscale=[[0, arrow_kwargs["color"]], [1, arrow_kwargs["color"]]],
            sizemode="absolute",
            sizeref=arrow_kwargs["arrow_head_length"],
            showscale=False,
            **kwargs,
        )
    else:
        fig.add_scatter(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            mode="lines+markers",
            marker=dict(
                symbol="arrow",
                size=10,
                color=arrow_kwargs["color"],
                angle=np.arctan2(scaled_vector[1], scaled_vector[0]),
                angleref="previous",
            ),
            line=dict(color=arrow_kwargs["color"], width=arrow_kwargs["width"]),
            showlegend=False,
            **kwargs,
        )


def get_first_matching_site_prop(
    structures: Sequence[Structure],
    prop_keys: Sequence[str],
    *,
    warn_if_none: bool = True,
    filter_callback: Callable[[str, Any], bool] | None = None,
) -> str | None:
    """Find the first property key that exists in any of the passed structures'
    properties or site properties. Will look in site.properties first, then
    structure.properties.

    Args:
        structures (Sequence[Structure]): pymatgen Structures to check.
        prop_keys (Sequence[str]): Property keys to look for.
        warn_if_none (bool, optional): Whether to warn if no matching property is found.
        filter_callback (Callable[[str, Any], bool] | None, optional): A function that
            takes the property key and value, and returns True if the property should be
            considered a match. If None, all properties are considered matches.

    Returns:
        str | None: The first matching property key found, or None if no match is found.
    """
    for prop in prop_keys:
        for struct in structures:
            if prop in struct.site_properties:
                for site in struct:
                    value = site.properties[prop]
                    if filter_callback is None or filter_callback(prop, value):
                        return prop
            elif prop in struct.properties:
                value = struct.properties[prop]
                if filter_callback is None or filter_callback(prop, value):
                    return prop

    if prop_keys and warn_if_none:
        warn_msg = f"None of {prop_keys=} found in any site or structure properties"
        warnings.warn(warn_msg, UserWarning, stacklevel=2)

    return None


def draw_bonds(
    fig: go.Figure,
    structure: Structure,
    nn: NearNeighbors,
    *,
    is_3d: bool = True,
    bond_kwargs: dict[str, Any] | None = None,
    row: int | None = None,
    col: int | None = None,
    scene: str | None = None,
    visible_image_atoms: set[tuple[float, float, float]] | None = None,
    rotation_matrix: np.ndarray | None = None,
    elem_colors: dict[str, ColorType] | None = None,
) -> None:
    """Draw bonds between atoms in the structure.

    Args:
        fig: Plotly figure to add bonds to
        structure: Pymatgen structure
        nn: NearNeighbors object to determine bonds
        is_3d: Whether the plot is 3D
        bond_kwargs: Customization options for bonds. If bond_kwargs["color"] is a
            tuple/list of colors, gradient coloring will be used. If color=True and
            elem_colors is provided, the gradient will use the colors of the connected
            atoms.
        row: Row index for 2D subplots
        col: Column index for 2D subplots
        scene: Scene name for 3D plots
        visible_image_atoms: Set of coordinates of visible image atoms
        rotation_matrix: Rotation matrix for 2D plots
        elem_colors: Element color map, used for gradient coloring if color=True
    """
    from matplotlib.colors import to_rgb
    from plotly.colors import find_intermediate_color

    default_bond_kwargs = dict(color=True, width=4)
    bond_kwargs = default_bond_kwargs | (bond_kwargs or {})
    elem_colors = elem_colors or {}

    def parse_color(color: str | ColorType) -> str:
        """Convert various color formats to RGB string."""
        if isinstance(color, str) and color.startswith("rgb"):
            return color
        if (
            isinstance(color, tuple | list)
            and len(color) == 3
            and all(0 <= val <= 1 for val in color)
        ):
            rgb_values = [int(255 * val) for val in color]
            return f"rgb({rgb_values[0]}, {rgb_values[1]}, {rgb_values[2]})"
        if color is True:  # Handle the case where color is True
            return "rgb(128, 128, 128)"  # Default to gray
        red, green, blue = to_rgb(color)
        return f"rgb({int(red * 255)}, {int(green * 255)}, {int(blue * 255)})"

    for site_idx, site in enumerate(structure):
        for neighbor in nn.get_nn_info(structure, site_idx):
            end_site = neighbor["site"]
            is_in_unit_cell = all(0 <= coord < 1 for coord in end_site.frac_coords)
            end_coords = tuple(
                np.dot(end_site.coords, rotation_matrix)
                if not is_3d and rotation_matrix is not None
                else end_site.coords
            )

            if not (
                is_in_unit_cell
                or (visible_image_atoms and end_coords in visible_image_atoms)
            ):
                continue

            start_coords = (
                np.dot(site.coords, rotation_matrix)
                if not is_3d and rotation_matrix is not None
                else site.coords
            )
            end_coords = (
                np.dot(end_site.coords, rotation_matrix)
                if not is_3d and rotation_matrix is not None
                else end_site.coords
            )

            # Set up colors for gradient or single color
            use_gradient = isinstance(bond_kwargs["color"], list | tuple) or (
                bond_kwargs["color"] is True and elem_colors
            )
            if use_gradient:
                if isinstance(bond_kwargs["color"], list | tuple):
                    colors = [parse_color(color) for color in bond_kwargs["color"]]
                else:  # color is True, use element colors
                    start_elem, end_elem = (
                        site.species.elements[0].symbol,
                        end_site.species.elements[0].symbol,
                    )
                    colors = [
                        parse_color(elem_colors.get(start_elem, "gray")),
                        parse_color(elem_colors.get(end_elem, "gray")),
                    ]
                n_segments = 10
            else:
                colors = [parse_color(bond_kwargs["color"])]
                n_segments = 1

            # Draw bond segments
            for segment_idx in range(n_segments):
                frac_start, frac_end = (
                    segment_idx / n_segments,
                    (segment_idx + 1) / n_segments,
                )
                segment_start = [
                    start + frac * (end - start)
                    for start, end, frac in zip(
                        start_coords, end_coords, [frac_start] * 3, strict=True
                    )
                ]
                segment_end = [
                    start + frac * (end - start)
                    for start, end, frac in zip(
                        start_coords, end_coords, [frac_end] * 3, strict=True
                    )
                ]

                if len(colors) == 1:
                    segment_color = colors[0]
                elif len(colors) == 2:
                    segment_color = find_intermediate_color(
                        colors[0],
                        colors[1],
                        (frac_start + frac_end) / 2,
                        colortype="rgb",
                    )
                else:
                    color_idx = min(
                        int((len(colors) - 1) * (frac_start + frac_end) / 2),
                        len(colors) - 2,
                    )
                    segment_color = find_intermediate_color(
                        colors[color_idx],
                        colors[color_idx + 1],
                        ((frac_start + frac_end) / 2 - color_idx / (len(colors) - 1))
                        * (len(colors) - 1),
                        colortype="rgb",
                    )

                trace_kwargs = dict(
                    mode="lines",
                    line=dict(
                        width=bond_kwargs.get("width", 2),
                        color=segment_color,
                        dash=bond_kwargs.get("dash"),
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                    name=f"bond {site_idx}-{neighbor['site_index']} segment "
                    f"{segment_idx}",
                )

                if is_3d:
                    fig.add_scatter3d(
                        x=[segment_start[0], segment_end[0]],
                        y=[segment_start[1], segment_end[1]],
                        z=[segment_start[2], segment_end[2]],
                        scene=scene,
                        **trace_kwargs,
                    )
                else:
                    fig.add_scatter(
                        x=[segment_start[0], segment_end[0]],
                        y=[segment_start[1], segment_end[1]],
                        row=row,
                        col=col,
                        **trace_kwargs,
                    )
