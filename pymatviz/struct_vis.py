from __future__ import annotations

import math
from itertools import product
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch, Wedge
from matplotlib.path import Path
from pymatgen.core import Structure

from pymatviz.utils import NumArray, covalent_radii, jmol_colors


# plot_structure_2d() and its helphers get_rot_matrix() and unit_cell_to_lines() were
# inspired by ASE https://wiki.fysik.dtu.dk/ase/ase/visualize/visualize.html#matplotlib


def get_rot_matrix(angles: str, rotation: NumArray = np.eye(3)) -> NumArray:
    """Convert Euler angles to a rotation matrix.

    Note the order of angles matters. 50x,40z != 40z,50x.

    Args:
        angles (str): Euler angles (in degrees) formatted as '-10y,50x,120z'
        rotation (NumArray, optional): Initial rotation matrix. Defaults to identity
            matrix.

    Returns:
        NumArray: 3d rotation matrix.
    """
    if angles == "":
        return rotation.copy()  # return initial rotation matrix if no angles

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
        - z-indices that sort plot elements into out-of-plane layers
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
    rotation: str = "10x,10y,0z",
    atomic_radii: float | dict[str, float] | None = None,
    colors: dict[str, str | list[float]] = None,
    scale: float = 1,
    show_unit_cell: bool = True,
    site_labels: bool | dict[str, str | float] | list[str | float] = True,
    label_kwargs: dict[str, Any] = {"fontsize": 14},
) -> plt.Axes:
    """Plot pymatgen structure object in 2d. Uses matplotlib.

    Inspired by ASE's ase.visualize.plot.plot_atoms()
    https://wiki.fysik.dtu.dk/ase/ase/visualize/visualize.html#matplotlib
    pymatviz aims to give similar output to ASE but supports disordered structures and
    avoids the conversion hassle of AseAtomsAdaptor.get_atoms(pmg_struct).

    For example, these two snippets should give very similar output:
    ```py
    from pymatgen.ext.matproj import MPRester

    mp_19017 = MPRester().get_structure_by_material_id("mp-19017")

    # ASE
    from ase.visualize.plot import plot_atoms
    from pymatgen.io.ase import AseAtomsAdaptor

    plot_atoms(AseAtomsAdaptor().get_atoms(mp_19017), rotation="10x,10y,0z", radii=0.5)

    # pymatviz
    from pymatviz import plot_structure_2d

    plot_structure_2d(mp_19017)
    ```

    Args:
        struct (Structure): Must be pymatgen instance. ax (plt.Axes, optional):
        Matplotlib axes on which to plot. Defaults to None. rotation (str, optional):
        Euler angles in degrees in the form '10x,20y,30z'
            describing angle at which to view structure. Defaults to "".
        atomic_radii (float | dict[str, float], optional): Either a scaling factor for
            default radii or map from element symbol to atomic radii. Defaults to
            covalent radii.
        colors (dict[str, str | list[float]], optional): Map from element symbols to
            colors, either a named color (str) or rgb(a) values like (0.2, 0.3, 0.6).
            Defaults to JMol colors.
        scale (float, optional): Scaling of the plotted atoms and lines. Defaults to 1.
        show_unit_cell (bool, optional): Whether to draw unit cell. Defaults to True.
        site_labels (bool | dict[str, str | float] | list[str | float]): How to annotate
            lattice sites. If True, labels are element symbols. If a dict, should map
            element symbols to labels. If a list, must be same length as the number of
            sites in the crystal. Defaults to True.
        label_kwargs (dict, optional): Keyword arguments for matplotlib.text.Text.
            Defaults to {"fontsize": 14}.

    Returns:
        plt.Axes: matplotlib Axes instance with plotted structure.
    """
    if ax is None:
        ax = plt.gca()

    elems = [str(site.species.elements[0]) for site in struct]

    if isinstance(site_labels, list):
        assert len(site_labels) == len(
            struct
        ), "Length mismatch between site_labels and struct"

    if colors is None:
        colors = jmol_colors

    if atomic_radii is None or isinstance(atomic_radii, float):
        atomic_radii = 0.7 * covalent_radii * (atomic_radii or 1)
    else:
        assert isinstance(atomic_radii, dict)
        # make sure all present elements are assigned a radius
        missing = {el for el in elems if el not in atomic_radii}
        assert not missing, f"atomic_radii is missing keys: {missing}"

    radii = np.array([atomic_radii[el] for el in elems])  # type: ignore

    n_atoms = len(struct)
    rot_matrix = get_rot_matrix(rotation)
    unit_cell = struct.lattice.matrix

    if show_unit_cell:
        lines, z_indices, unit_cell_lines = unit_cell_to_lines(unit_cell)
        corners = np.array(list(product((0, 1), (0, 1), (0, 1))))
        cell_vertices = np.dot(corners, unit_cell)
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

    min_coords = (site_coords - radii[:, None]).min(0)
    max_coords = (site_coords + radii[:, None]).max(0)

    if show_unit_cell:
        min_coords = np.minimum(min_coords, cell_vertices.min(0))
        max_coords = np.maximum(max_coords, cell_vertices.max(0))

    means = (min_coords + max_coords) / 2
    coord_ranges = 1.05 * (max_coords - min_coords)

    offset = scale * (means - coord_ranges / 2)
    positions *= scale
    positions -= offset

    if n_lines > 0:
        unit_cell_lines = np.dot(unit_cell_lines, rot_matrix)[:, :2] * scale

    # sort so we draw from back to front along out-of-plane (z-)axis
    for idx in positions[:, 2].argsort():
        xy = positions[idx, :2]
        start = 0
        if idx < n_atoms:
            # loop over all species on a site (usually just 1 for ordered sites)
            for elem, occupancy in struct[idx].species.items():
                elem = str(elem)
                radius = atomic_radii[elem] * scale  # type: ignore
                wedge = Wedge(
                    xy,
                    radius,
                    360 * start,
                    360 * (start + occupancy),
                    facecolor=colors[elem],
                    edgecolor="black",
                )
                ax.add_patch(wedge)

                txt = elem
                if isinstance(site_labels, dict) and elem in site_labels:
                    txt = site_labels.get(elem, "")
                if isinstance(site_labels, list):
                    txt = site_labels[idx]

                if site_labels:
                    # place element symbol half way along outer wedge edge for
                    # disordered sites
                    half_way = 2 * np.pi * (start + occupancy / 2)
                    direction = np.array([math.cos(half_way), math.sin(half_way)])
                    text_offset = (
                        (radius + 0.3 * scale) * direction if occupancy < 1 else (0, 0)
                    )

                    txt_kwds = dict(ha="center", va="center", **label_kwargs)
                    ax.text(*(xy + text_offset), txt, **txt_kwds)

                start += occupancy
        else:  # draw unit cell
            idx -= n_atoms
            # only draw line if not obstructed by an atom
            if z_indices[idx] != -1:
                hxy = unit_cell_lines[z_indices[idx]]
                path = PathPatch(Path((xy + hxy, xy - hxy)))
                ax.add_patch(path)

    width, height, _ = scale * coord_ranges
    ax.set(xlim=[0, width], ylim=[0, height], aspect="equal")
    ax.axis("off")

    return ax
