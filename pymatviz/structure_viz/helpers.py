"""2D plots of pymatgen structures with matplotlib.

structure_2d() and its helpers get_rot_matrix() and unit_cell_to_lines() were
inspired by ASE https://wiki.fysik.dtu.dk/ase/ase/visualize/visualize.html#matplotlib.
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pymatgen.core import Composition, Lattice, PeriodicSite, Structure

from pymatviz.colors import ELEM_COLORS_JMOL, ELEM_COLORS_VESTA
from pymatviz.enums import ElemColorScheme, Key
from pymatviz.utils import df_ptable, pick_bw_for_contrast


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any, Literal

    import plotly.graph_objects as go
    from numpy.typing import ArrayLike


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


def get_image_atoms(
    site: PeriodicSite, lattice: Lattice, tol: float = 0.02
) -> np.ndarray:
    """Get image atoms for a given site."""
    coords_image_atoms: list[np.ndarray] = []

    # If the site is at the lattice origin, return an empty array
    if np.allclose(site.frac_coords, (0, 0, 0), atol=tol):
        return np.array(coords_image_atoms)

    # Generate all possible combinations of lattice vector offsets
    offsets = list(itertools.product([0, 1], repeat=3))

    for offset in offsets:
        if offset == (0, 0, 0):
            continue  # Skip the original atom

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


def get_elem_colors(elem_colors: ElemColorScheme | dict[str, str]) -> dict[str, str]:
    """Get element colors based on the provided scheme or custom dictionary."""
    if str(elem_colors) == str(ElemColorScheme.jmol):
        return ELEM_COLORS_JMOL
    if str(elem_colors) == str(ElemColorScheme.vesta):
        return ELEM_COLORS_VESTA
    if isinstance(elem_colors, dict):
        return elem_colors
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
    major_elem_symbol: str,
    majority_species: str,
) -> str:
    """Generate a label for a site based on the provided labeling scheme."""
    if site_labels == "symbol":
        return str(major_elem_symbol)
    if site_labels == "species":
        return str(majority_species)
    if site_labels is False:
        return ""
    if isinstance(site_labels, dict):
        return site_labels.get(
            repr(major_elem_symbol), site_labels.get(major_elem_symbol, "")
        )
    if isinstance(site_labels, list | tuple):
        return site_labels[site_idx]
    raise ValueError(
        f"Invalid {site_labels=}. Must be one of "
        f"('symbol', 'species', False, dict, list)"
    )


def generate_subplot_title(
    struct_i: Structure,
    struct_key: Any,
    idx: int,
    subplot_title: Callable[[Structure, str | int], str | dict[str, Any]] | None,
) -> dict[str, Any]:
    """Generate a subplot title based on the provided function or default logic."""
    title_dict: dict[str, str | float | dict[str, str | float]] = {
        "font": {"color": "black"}
    }

    if callable(subplot_title):
        sub_title = subplot_title(struct_i, struct_key)
        if isinstance(sub_title, str | int | float):
            title_dict["text"] = str(sub_title)
        elif isinstance(sub_title, dict):
            title_dict |= sub_title
        else:
            raise TypeError(
                f"Invalid subplot_title={sub_title}. Must be a str or dict."
            )

    if not title_dict.get("text"):
        if isinstance(struct_key, int):
            spg_num = struct_i.get_space_group_info()[1]
            title_dict["text"] = f"{idx}. {struct_i.formula} (spg={spg_num})"
        elif isinstance(struct_key, str):
            title_dict["text"] = str(struct_key)
        else:
            raise TypeError(f"Invalid {struct_key=}. Must be an int or str.")

    return title_dict


def draw_site(
    fig: go.Figure,
    site: PeriodicSite,
    coords: np.ndarray,
    site_idx: int,
    site_labels: Any,
    _elem_colors: dict[str, str],
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
    **kwargs: Any,
) -> None:
    """Add a site (regular or image) to the plot."""
    species = getattr(site, "specie", site.species)
    majority_species = (
        max(species, key=species.get) if isinstance(species, Composition) else species
    )
    major_elem_symbol = majority_species.symbol
    site_radius = _atomic_radii[major_elem_symbol] * scale
    color = _elem_colors.get(major_elem_symbol, "gray")

    hover_text = (
        f"<b>Site: {majority_species}</b><br>"
        f"Coordinates ({', '.join(f'{c:.3g}' for c in site.coords)})<br>"
        f"[{', '.join(f'{c:.3g}' for c in site.frac_coords)}]"
    )

    if site.properties:
        hover_text += "<br>Properties: " + ", ".join(
            f"{k}: {v}" for k, v in site.properties.items()
        )

    txt = generate_site_label(
        site_labels, site_idx, major_elem_symbol, majority_species
    )

    marker = dict(
        size=site_radius * atom_size * (0.8 if is_image else 1),
        color=color,
        opacity=0.5 if is_image else 1,
    )
    marker.update(site_kwargs)

    scatter_kwargs = dict(
        x=[coords[0]],
        y=[coords[1]],
        mode="markers+text" if txt else "markers",
        marker=marker,
        text=txt,
        textposition="middle center",
        textfont=dict(
            color=pick_bw_for_contrast(color, text_color_threshold=0.5),
            size=np.clip(atom_size * site_radius * (0.8 if is_image else 1), 10, 18),
        ),
        hovertext=f"Image of {hover_text}" if is_image else hover_text,
        hoverinfo="text",
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
    """Convert various input types to a dictionary of structures."""
    if isinstance(struct, Structure):
        return {0: struct}
    if isinstance(struct, pd.Series):
        return struct.to_dict()
    if isinstance(next(iter(struct), None), Structure):
        return dict(enumerate(struct))
    if isinstance(struct, dict) and {*map(type, struct.values())} == {Structure}:
        return struct
    raise TypeError(f"Expected pymatgen Structure or Sequence of them, got {struct=}")


def _add_unit_cell(
    fig: go.Figure,
    structure: Structure,
    unit_cell_kwargs: dict[str, Any],
    *,
    is_3d: bool = True,
    row: int | None = None,
    col: int | None = None,
    scene: str | None = None,
) -> go.Figure:
    corners = np.array(list(itertools.product((0, 1), (0, 1), (0, 1))))
    cart_corners = structure.lattice.get_cartesian_coords(corners)

    alpha, beta, gamma = structure.lattice.angles

    def add_trace(
        x: float | Sequence[float],
        y: float | Sequence[float],
        z: float | Sequence[float] | None = None,
        mode: str = "lines",
        marker: dict[str, Any] | None = None,
        line: dict[str, Any] | None = None,
        hovertext: str | list[str | None] | None = None,
    ) -> None:
        trace_kwargs = dict(
            mode=mode,
            hoverinfo="text",
            hovertext=hovertext,
            showlegend=False,
            marker=marker,
            line=line,
        )

        if is_3d:
            fig.add_scatter3d(x=x, y=y, z=z, scene=scene, **trace_kwargs)
        else:
            fig.add_scatter(x=x, y=y, row=row, col=col, **trace_kwargs)

    # Add edges
    edge_defaults = dict(color="black", width=1, dash="dash")
    edge_kwargs = edge_defaults | unit_cell_kwargs.get("edge", {})
    for start, end in UNIT_CELL_EDGES:
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

        add_trace(
            x=[start_point[0], mid_point[0], end_point[0]],
            y=[start_point[1], mid_point[1], end_point[1]],
            z=[start_point[2], mid_point[2], end_point[2]] if is_3d else None,
            mode="lines",
            line=edge_kwargs,
            hovertext=[None, hover_text, None],
        )

    # Add corner spheres
    node_defaults = dict(size=3, color="black")
    node_kwargs = node_defaults | unit_cell_kwargs.get("node", {})
    for i, (frac_coord, cart_coord) in enumerate(
        zip(corners, cart_corners, strict=False)
    ):
        adjacent_angles = []
        for _ in range(3):
            v1 = cart_corners[(i + 1) % 8] - cart_coord
            v2 = cart_corners[(i + 2) % 8] - cart_coord
            angle = np.degrees(
                np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            )
            adjacent_angles.append(angle)

        hover_text = (
            f"({', '.join(f'{c:.3g}' for c in cart_coord)}) "
            f"[{', '.join(f'{c:.3g}' for c in frac_coord)}]<br>"
            f"α = {alpha:.3g}°, β = {beta:.3g}°, γ = {gamma:.3g}°"  # noqa: RUF001
        )

        add_trace(
            x=[cart_coord[0]],
            y=[cart_coord[1]],
            z=[cart_coord[2]] if is_3d else None,
            mode="markers",
            marker=node_kwargs,
            hovertext=hover_text,
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
