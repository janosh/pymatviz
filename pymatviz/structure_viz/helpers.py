"""2D plots of pymatgen structures with matplotlib.

structure_2d() and its helpers get_rot_matrix() and unit_cell_to_lines() were
inspired by ASE https://wiki.fysik.dtu.dk/ase/ase/visualize/visualize.html#matplotlib.
"""

from __future__ import annotations

import itertools
import math
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
    site: PeriodicSite, lattice: Lattice, tolerance: float = 0.02
) -> list[ArrayLike]:
    """Get image atoms for a given site.

    Args:
        site (PeriodicSite): The site to get image atoms for.
        lattice (Lattice): The lattice of the structure.
        tolerance (float, optional): Tolerance for considering an atom on the edge.
            Defaults to 0.02 (2%).

    Returns:
        list: List of cartesian coordinates of image atoms.
    """
    frac_coords = site.frac_coords
    image_atoms = []

    # Generate all possible combinations of lattice vector offsets
    offsets = list(itertools.product([0, 1], repeat=3))  # [0,0,0], [1,0,0], etc.

    for offset in offsets:
        if offset == (0, 0, 0):
            continue  # Skip the original atom

        # Apply offset to fractional coordinates
        new_frac = frac_coords + offset
        new_cart = lattice.get_cartesian_coords(new_frac)

        # Check if the new position is within or very close to the unit cell
        is_within_cell = all(-tolerance <= coord <= 1 + tolerance for coord in new_frac)
        is_on_edge = any(
            np.isclose(new_frac, 0, atol=tolerance)
            | np.isclose(new_frac, 1, atol=tolerance)
        )

        if is_within_cell and is_on_edge:
            image_atoms.append(new_cart)

    return image_atoms


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
        segments.append(segment)
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
        return 0.7 * df_ptable[Key.covalent_radius].fillna(0.2) * (atomic_radii or 1)
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
    if callable(subplot_title):
        sub_title = subplot_title(struct_i, idx)
        return dict(text=sub_title) if isinstance(sub_title, str) else sub_title
    if isinstance(struct_key, int):
        spg_num = struct_i.get_space_group_info()[1]
        sub_title = f"{struct_i.formula} (spg={spg_num})"
        return dict(text=f"{idx}. {sub_title}")
    return dict(text=struct_key)


def add_site_to_plot(
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

    marker = (
        dict(
            size=site_radius * atom_size * (0.8 if is_image else 1),
            color=color,
            opacity=0.5 if is_image else 1,
        )
        | site_kwargs
    )

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
