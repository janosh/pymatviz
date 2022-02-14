from __future__ import annotations

import math
from itertools import product
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch, Wedge
from matplotlib.path import Path
from pymatgen.core import Structure

from ml_matrics.utils import NumArray, covalent_radii, jmol_colors


def get_rot_matrix(angles: str, rotation: NumArray = np.identity(3)) -> NumArray:
    """Convert Euler angles to a rotation matrix.

    Note the order of angles matters. E.g. '50x,40z' != '40z,50x'.

    Args:
        angles (str): Euler angles (in degrees) of formatted as '50x,-10y,120z'
        rotation (NumArray, optional): Starting rotation matrix.
            Defaults to np.identity(3).

    Returns:
        ndarray: Rotation matrix.
    """
    if angles == "":
        return rotation.copy()  # return unit matrix if no angles

    for i, a in [
        ("xyz".index(s[-1]), math.radians(float(s[:-1]))) for s in angles.split(",")
    ]:
        s = math.sin(a)
        c = math.cos(a)
        if i == 0:
            rotation = np.dot(rotation, [(1, 0, 0), (0, c, s), (0, -s, c)])
        elif i == 1:
            rotation = np.dot(rotation, [(c, 0, -s), (0, 1, 0), (s, 0, c)])
        else:
            rotation = np.dot(rotation, [(c, s, 0), (-s, c, 0), (0, 0, 1)])
    return rotation


def unit_cell_to_lines(cell: NumArray) -> tuple[NumArray, NumArray, NumArray]:
    """Convert lattice vectors to plot lines.

    Args:
        cell (NumArray): Lattice vectors.

    Returns:
        tuple[NumArray, NumArray, NumArray]:
        - Lines
        - z-indices that sort plot elements into out-of-place layers
        - lines used to plot the unit cell
    """
    n_lines = 0
    segments = []
    for c in range(3):
        norm = math.sqrt(sum(cell[c] ** 2))
        segment = max(2, int(norm / 0.3))
        segments.append(segment)
        n_lines += 4 * segment

    lines = np.empty((n_lines, 3))
    z_indices = np.empty(n_lines, int)
    unit_cell_lines = np.zeros((3, 3))

    n1 = 0
    for c in range(3):
        segment = segments[c]
        dd = cell[c] / (4 * segment - 2)
        unit_cell_lines[c] = dd
        P = np.arange(1, 4 * segment + 1, 4)[:, None] * dd
        z_indices[n1:] = c
        for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            n2 = n1 + segment
            lines[n1:n2] = P + i * cell[c - 2] + j * cell[c - 1]
            n1 = n2

    return lines, z_indices, unit_cell_lines


def plot_structure_2d(
    struct: Structure,
    ax: plt.Axes = None,
    rotation: str = "",
    atomic_radii: float | dict[str, float] | None = None,
    colors: dict[str, str | list[float]] = None,
    scale: float = 1,
    offset: tuple[float, float] = (0, 0),
    show_unit_cell: bool = True,
    bbox: tuple[float, float, float, float] = None,
    maxwidth: int = None,
) -> plt.Axes:
    """Plot pymatgen structure object in 2d. Uses matplotlib.

    Args:
        struct (Structure): Must be pymatgen instance.
        ax (plt.Axes, optional): Matplotlib axes on which to plot. Defaults to None.
        rotation (str, optional): Euler angles in degrees in the form '10x,20y,30z'
            describing angle at which to view structure. Defaults to "".
        radii (float | dict[str, float], optional): Either a scaling factor for default
            radii or map from element symbol to atomic radii. Defaults to covalent
            radii.
        colors (dict[str, str | list[float]], optional): Map from element symbols to
            colors, either a named color (str) or rgb(a) values like (0.2, 0.3, 0.6).
            Defaults to JMol colors.
        scale (float, optional): Scaling of the plotted atoms and lines. Defaults to 1.
        offset (tuple[float, float], optional): (x, y) offset of the plotted atoms and
            lines. Defaults to (0, 0).
        show_unit_cell (bool, optional): Whether to draw unit cell. Defaults to True.
        bbox (tuple[float, float, float, float], optional): Bounding box for the plot.
            Defaults to None.
        maxwidth (int, optional): Maximum width of the plot. Defaults to None.

    Returns:
        plt.Axes: matplotlib Axes instance with plotted structure.
    """
    if ax is None:
        ax = plt.gca()

    # atom_nums = np.array(struct.atomic_numbers)
    elems = [str(site.species.elements[0]) for site in struct]

    if colors is None:
        colors = jmol_colors

    if isinstance(atomic_radii, (float, type(None))):
        atomic_radii = covalent_radii * (atomic_radii or 1)
    else:
        assert isinstance(atomic_radii, dict)
        # make sure all present elements are assigned a radius
        assert all(el in atomic_radii for el in elems)

    atomic_radii = cast(dict[str, float], atomic_radii)
    radii = np.array([atomic_radii[el] for el in elems])

    n_atoms = len(struct)
    rot_matrix = get_rot_matrix(rotation)
    unit_cell = struct.lattice.matrix

    if show_unit_cell:
        lines, z_indices, unit_cell_lines = unit_cell_to_lines(unit_cell)
        mult = np.array(list(product((0, 1), (0, 1), (0, 1))))
        cell_vertices = np.dot(mult, unit_cell)
        cell_vertices = np.dot(cell_vertices, rot_matrix)
    else:
        lines = np.empty((0, 3))
        z_indices = None
        unit_cell_lines = None
        cell_vertices = None

    n_lines = len(lines)

    positions = np.empty((n_atoms + n_lines, 3))
    site_coords = np.array([site.coords for site in struct])
    positions[:n_atoms] = site_coords
    positions[n_atoms:] = lines

    # determine which lines should be hidden behind other objects
    for idx in range(n_lines):
        this_layer = unit_cell_lines[z_indices[idx]]
        occlu_top = ((site_coords - lines[idx] + this_layer) ** 2).sum(1) < radii**2
        occlu_bot = ((site_coords - lines[idx] - this_layer) ** 2).sum(1) < radii**2
        if any(occlu_top & occlu_bot):
            z_indices[idx] = -1

    positions = np.dot(positions, rot_matrix)
    site_coords = positions[:n_atoms]

    if bbox is None:
        min_coords = (site_coords - radii[:, None]).min(0)
        max_coords = (site_coords + radii[:, None]).max(0)

        if show_unit_cell:
            min_coords = np.minimum(min_coords, cell_vertices.min(0))
            max_coords = np.maximum(max_coords, cell_vertices.max(0))

        means = (min_coords + max_coords) / 2
        coord_ranges = 1.05 * (max_coords - min_coords)
        width = scale * coord_ranges[0]

        if maxwidth and width > maxwidth:
            width = maxwidth
            scale = width / coord_ranges[0]

        height = scale * coord_ranges[1]
        offset = np.array(
            [scale * means[0] - width / 2, scale * means[1] - height / 2, 0]
        )
    else:
        width = (bbox[2] - bbox[0]) * scale
        height = (bbox[3] - bbox[1]) * scale
        offset = np.array([bbox[0], bbox[1], 0]) * scale

    positions *= scale
    positions -= offset

    if n_lines > 0:
        unit_cell_lines = np.dot(unit_cell_lines, rot_matrix)[:, :2] * scale

    # sort so we draw from back to front along out-of-plane (z-)axis
    for idx in positions[:, 2].argsort():
        xy = positions[idx, :2]
        if idx < n_atoms:
            start = 0
            # loop over all species on a site
            for elem, occ in struct[idx].species.items():
                elem = str(elem)
                wedge = Wedge(
                    xy,
                    atomic_radii[elem] * scale,
                    start,
                    start + 360 * occ,
                    facecolor=colors[elem],
                    edgecolor="black",
                )
                ax.add_patch(wedge)
                start += 360 * occ

        else:
            # draw unit cell
            idx -= n_atoms
            c = z_indices[idx]
            if c != -1:  # don't draw line if it should be obstructed by an atom
                hxy = unit_cell_lines[c]
                path = PathPatch(Path((xy + hxy, xy - hxy)))
                ax.add_patch(path)

    ax.set(xlim=[0, width], ylim=[0, height], aspect="equal")
    ax.axis("off")

    return ax
