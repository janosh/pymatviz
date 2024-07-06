"""2D plots of pymatgen structures with matplotlib.

plot_structure_2d() and its helpers get_rot_matrix() and unit_cell_to_lines() were
inspired by ASE https://wiki.fysik.dtu.dk/ase/ase/visualize/visualize.html#matplotlib.
"""

from __future__ import annotations

import math
import warnings
from itertools import product
from typing import TYPE_CHECKING, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch, Wedge
from matplotlib.path import Path
from pymatgen.analysis.local_env import CrystalNN, NearNeighbors
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pymatviz.colors import ELEM_COLORS_JMOL, ELEM_COLORS_VESTA
from pymatviz.enums import ElemColorScheme, Key
from pymatviz.utils import ExperimentalWarning, covalent_radii


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Literal

    from matplotlib.typing import ColorType
    from numpy.typing import ArrayLike


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


def plot_structure_2d(
    struct: Structure | Sequence[Structure],
    *,
    ax: plt.Axes | None = None,
    rotation: str = "10x,10y,0z",
    atomic_radii: float | dict[str, float] | None = None,
    elem_colors: ElemColorScheme | dict[str, str | ColorType] = ElemColorScheme.jmol,
    scale: float = 1,
    show_unit_cell: bool = True,
    show_bonds: bool | NearNeighbors = False,
    site_labels: bool
    | Literal["symbol", "species"]
    | dict[str, str | float]
    | Sequence[str | float] = True,
    label_kwargs: dict[str, Any] | None = None,
    bond_kwargs: dict[str, Any] | None = None,
    standardize_struct: bool | None = None,
    axis: bool | str = "off",
    n_cols: int = 4,
    subplot_kwargs: dict[str, Any] | None = None,
    subplot_title: Callable[[Structure, str | int], str] | None = None,
    **kwargs: Any,
) -> plt.Axes | tuple[plt.Figure, np.ndarray[plt.Axes]]:
    """Plot pymatgen structures in 2D with matplotlib.

    Inspired by ASE's ase.visualize.plot.plot_atoms()
    https://wiki.fysik.dtu.dk/ase/ase/visualize/visualize.html#matplotlib
    pymatviz aims to give similar output to ASE but supports disordered structures and
    avoids the conversion hassle of AseAtomsAdaptor.get_atoms(pmg_struct).

    For example, these two snippets should give very similar output:

    ```python
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

    Multiple structures in single figure example:

    ```py
    from pymatgen.ext.matproj import MPRester
    from pymatviz import plot_structure_2d

    structures = {
        (mp_id := f"mp-{idx}"): MPRester().get_structure_by_material_id(mp_id)
        for idx in range(1, 5)
    }
    plot_structure_2d(structures)
    ```

    Args:
        struct (Structure): Must be pymatgen instance.
        ax (plt.Axes, optional): Matplotlib axes on which to plot. Defaults to None.
        rotation (str, optional): Euler angles in degrees in the form '10x,20y,30z'
            describing angle at which to view structure. Defaults to "".
        atomic_radii (float | dict[str, float], optional): Either a scaling factor for
            default radii or map from element symbol to atomic radii. Defaults to
            covalent radii.
        elem_colors (dict[str, str | list[float]], optional): Map from element symbols
            to colors, either a named color (str) or rgb(a) values like (0.2, 0.3, 0.6).
            Defaults to JMol colors (https://jmol.sourceforge.net/jscolors).
        scale (float, optional): Scaling of the plotted atoms and lines. Defaults to 1.
        show_unit_cell (bool, optional): Whether to plot unit cell. Defaults to True.
        show_bonds (bool | NearNeighbors, optional): Whether to plot bonds. If True, use
            pymatgen.analysis.local_env.CrystalNN to infer the structure's connectivity.
            If False, don't plot bonds. If a subclass of
            pymatgen.analysis.local_env.NearNeighbors, use that to determine
            connectivity. Options include VoronoiNN, MinimumDistanceNN, OpenBabelNN,
            CovalentBondNN, dtc. Defaults to True.
        site_labels (bool | "symbol" | "species" | dict[str, str | float] | Sequence):
            How to annotate lattice sites.
            If True, labels are element species (symbol + oxidation
            state). If a dict, should map species strings (or element symbols but looks
            for species string first) to labels. If a list, must be same length as the
            number of sites in the crystal. If a string, must be "symbol" or
            "species". "symbol" hides the oxidation state, "species" shows it
            (equivalent to True). Defaults to True.
        label_kwargs (dict, optional): Keyword arguments for matplotlib.text.Text like
            {"fontsize": 14}. Defaults to None.
        bond_kwargs (dict, optional): Keyword arguments for the matplotlib.path.Path
            class used to plot chemical bonds. Allowed are edgecolor, facecolor, color,
            linewidth, linestyle, antialiased, hatch, fill, capstyle, joinstyle.
            Defaults to None.
        standardize_struct (bool, optional): Whether to standardize the structure using
            SpacegroupAnalyzer(struct).get_conventional_standard_structure() before
            plotting. Defaults to False unless any fractional coordinates are negative,
            i.e. any crystal sites are outside the unit cell. Set this to False to
            disable this behavior which speeds up plotting for many structures.
        axis (bool | str, optional): Whether/how to show plot axes. Defaults to "off".
            See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis for
            details.
        n_cols (int, optional): Number of columns for subplots. Defaults to 4.
        subplot_kwargs (dict, optional): Unused if only a single structure is passed.
            Keyword arguments for plt.subplots(). Defaults to None.
        subplot_title (Callable[[Structure, str | int], str], optional): Should return
            subplot titles. Receives the structure and its key or index when passed as
            a dict or pandas.Series. Defaults to None in which case the title is the
            structure's material id if available, otherwise its formula and space group.
        **kwargs: Unused.

    Raises:
        ValueError: On invalid site_labels.

    Returns:
        plt.Axes: matplotlib Axes instance with plotted structure.
    """
    if "colors" in kwargs:
        warnings.warn(
            "plot_structure_2d's colors keyword was deprecated on 2024-07-06 and will "
            "be removed soon. Use 'elem_colors' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        elem_colors = kwargs.pop("colors")
    if isinstance(struct, Structure):
        ax = ax or plt.gca()

        if subplot_kwargs is not None:
            warnings.warn(
                "subplot_kwargs are ignored when plotting a single structure",
                UserWarning,
                stacklevel=2,
            )

        if isinstance(site_labels, (list, tuple)) and len(site_labels) != len(struct):
            raise ValueError(
                f"If a list, site_labels ({len(site_labels)=}) must have same length as"
                f" the number of sites in the crystal ({len(struct)=})"
            )

        # Default behavior in case of no user input: standardize if any fractional
        # coordinates are negative
        has_sites_outside_unit_cell = any(any(site.frac_coords < 0) for site in struct)
        if standardize_struct is False and has_sites_outside_unit_cell:
            warnings.warn(
                "your structure has negative fractional coordinates but you passed "
                f"{standardize_struct=}, you may want to set standardize_struct=True",
                UserWarning,
                stacklevel=2,
            )
        elif standardize_struct is None:
            standardize_struct = has_sites_outside_unit_cell
        if standardize_struct:
            struct = SpacegroupAnalyzer(struct).get_conventional_standard_structure()

        # Get default colors
        if str(elem_colors) == str(ElemColorScheme.jmol):
            elem_colors = ELEM_COLORS_JMOL
        elif str(elem_colors) == str(ElemColorScheme.vesta):
            elem_colors = ELEM_COLORS_VESTA
        elif not isinstance(elem_colors, dict):
            valid_color_schemes = "', '".join(ElemColorScheme)
            raise ValueError(
                f"colors must be a dict or one of ('{valid_color_schemes}')"
            )

        # Get any element at each site, only used for occlusion calculation which won't
        # be perfect for disordered sites. Plotting wedges of different radii for
        # disordered sites is handled later.
        elements_at_sites = [str(site.species.elements[0].symbol) for site in struct]

        if atomic_radii is None or isinstance(atomic_radii, float):
            # atomic_radii is a scaling factor for the default set of radii
            atomic_radii = 0.7 * covalent_radii * (atomic_radii or 1)
        elif missing := set(elements_at_sites) - set(atomic_radii):
            # atomic_radii is assumed to be a map from element symbols to atomic radii
            # make sure all present elements are assigned a radius
            raise ValueError(f"atomic_radii is missing keys: {missing}")

        radii_at_sites = np.array(
            [atomic_radii[el] for el in elements_at_sites]  # type: ignore[index]
        )

        # Generate lines for unit cell
        rotation_matrix = _angles_to_rotation_matrix(rotation)
        unit_cell = struct.lattice.matrix

        if show_unit_cell:
            lines, z_indices, unit_cell_lines = unit_cell_to_lines(unit_cell)
            corners = np.array(list(product((0, 1), (0, 1), (0, 1))))
            cell_vertices = np.dot(corners, unit_cell)
            cell_vertices = np.dot(cell_vertices, rotation_matrix)
        else:
            lines = np.empty((0, 3))
            z_indices = None
            unit_cell_lines = None
            cell_vertices = None

        # Zip atoms and unit cell lines together
        n_atoms = len(struct)
        n_lines = len(lines)

        positions = np.empty((n_atoms + n_lines, 3))
        site_coords = np.array([site.coords for site in struct])
        positions[:n_atoms] = site_coords
        positions[n_atoms:] = lines

        # Determine which unit cell line should be hidden behind other objects
        for idx in range(n_lines):
            this_layer = unit_cell_lines[z_indices[idx]]

            occluded_top = ((site_coords - lines[idx] + this_layer) ** 2).sum(
                1
            ) < radii_at_sites**2

            occluded_bottom = ((site_coords - lines[idx] - this_layer) ** 2).sum(
                1
            ) < radii_at_sites**2

            if any(occluded_top & occluded_bottom):
                z_indices[idx] = -1

        # Apply rotation matrix
        positions = np.dot(positions, rotation_matrix)
        rotated_site_coords = positions[:n_atoms]

        # Normalize wedge positions
        min_coords = (rotated_site_coords - radii_at_sites[:, None]).min(0)
        max_coords = (rotated_site_coords + radii_at_sites[:, None]).max(0)

        if show_unit_cell:
            min_coords = np.minimum(min_coords, cell_vertices.min(0))
            max_coords = np.maximum(max_coords, cell_vertices.max(0))

        means = (min_coords + max_coords) / 2
        coord_ranges = 1.05 * (max_coords - min_coords)

        offset = scale * (means - coord_ranges / 2)
        positions *= scale
        positions -= offset

        # Rotate and scale unit cell lines
        if n_lines > 0:
            unit_cell_lines = np.dot(unit_cell_lines, rotation_matrix)[:, :2] * scale

        special_site_labels = ("symbol", "species")
        # Sort positions by 3rd dim (out-of-plane) to plot back-to-front along z-axis
        for idx in positions[:, 2].argsort():
            xy = positions[idx, :2]
            start = 0
            zorder = positions[idx][2]

            if idx < n_atoms:
                # Loop over all species on a site (usually just 1 for ordered sites)
                for species, occupancy in struct[idx].species.items():
                    # Strip oxidation state from element symbol (e.g. Ta5+ to Ta)
                    elem_symbol = species.symbol

                    radius = atomic_radii[elem_symbol] * scale  # type: ignore[index]
                    fallback_color = "gray"
                    face_color = elem_colors.get(elem_symbol, fallback_color)
                    if elem_symbol not in elem_colors:
                        warnings.warn(
                            f"{elem_symbol=} not in colors, using {fallback_color}",
                            UserWarning,
                            stacklevel=2,
                        )
                    wedge = Wedge(
                        xy,
                        radius,
                        360 * start,
                        360 * (start + occupancy),
                        facecolor=face_color,
                        edgecolor="black",
                        zorder=zorder,
                    )
                    # mostly here for testing purposes but has no none issues and might
                    # be useful to backtrace which wedge corresponds to which site
                    # see test_plot_structure_2d_color_schemes
                    wedge.elem_symbol, wedge.idx = elem_symbol, idx
                    ax.add_patch(wedge)

                    # Generate labels
                    if site_labels == "symbol":
                        txt = elem_symbol
                    elif site_labels in ("species", True):
                        txt = species
                    elif site_labels is False:
                        txt = ""
                    elif isinstance(site_labels, dict):
                        # Try element incl. oxidation state as dict key first (e.g.
                        # Na+), then just element as fallback
                        txt = site_labels.get(
                            repr(species), site_labels.get(elem_symbol, "")
                        )
                        if txt in special_site_labels:
                            txt = species if txt == "species" else elem_symbol
                    elif isinstance(site_labels, (list, tuple)):
                        txt = site_labels[idx]  # idx runs from 0 to n_atoms
                    else:
                        raise ValueError(
                            f"Invalid {site_labels=}. Must be one of (bool, "
                            f"{', '.join(special_site_labels)}, dict, list)"
                        )

                    # Add labels
                    if site_labels:
                        # Place element symbol half way along outer wedge edge for
                        # disordered sites
                        half_way = 2 * np.pi * (start + occupancy / 2)
                        direction = np.array([math.cos(half_way), math.sin(half_way)])
                        text_offset = (
                            (0.5 * radius) * direction if occupancy < 1 else (0, 0)
                        )

                        txt_kwds = dict(
                            ha="center",
                            va="center",
                            zorder=zorder,
                            **(label_kwargs or {}),
                        )
                        ax.text(*(xy + text_offset), txt, **txt_kwds)

                    start += occupancy

            # Plot unit cell
            else:
                cell_idx = idx - n_atoms
                # Only plot lines not obstructed by an atom
                if z_indices[cell_idx] != -1:
                    hxy = unit_cell_lines[z_indices[cell_idx]]
                    path = PathPatch(Path((xy + hxy, xy - hxy)), zorder=zorder)
                    ax.add_patch(path)

        if show_bonds:
            warnings.warn(
                "Warning: the show_bonds feature of plot_structure_2d() is "
                "experimental. Issues and PRs with improvements welcome.",
                category=ExperimentalWarning,
                stacklevel=2,
            )
            if show_bonds is True:
                neighbor_strategy_cls = CrystalNN
            elif issubclass(show_bonds, NearNeighbors):
                neighbor_strategy_cls = show_bonds
            else:
                raise ValueError(
                    f"Expected boolean or a NearNeighbors subclass for {show_bonds=}"
                )

            # If structure doesn't have any oxidation states yet, guess them from
            # chemical composition. Use CrystalNN and other strategies to better
            # estimate bond connectivity. Use getattr on site.specie since it's often a
            # pymatgen Element which has no oxi_state
            if not any(
                hasattr(getattr(site, "specie", None), "oxi_state") for site in struct
            ):
                try:
                    struct.add_oxidation_state_by_guess()
                except ValueError:  # fails for disordered structures
                    # Charge balance analysis requires integer values in Composition
                    pass

            structure_graph = neighbor_strategy_cls().get_bonded_structure(struct)

            bonds = structure_graph.graph.edges(data=True)
            for bond in bonds:
                from_idx, to_idx, data = bond
                if data["to_jimage"] != (0, 0, 0):
                    continue  # skip bonds across periodic boundaries
                from_xy = positions[from_idx, :2]
                to_xy = positions[to_idx, :2]

                bond_patch = PathPatch(Path((from_xy, to_xy)), **(bond_kwargs or {}))
                ax.add_patch(bond_patch)

        width, height, _ = scale * coord_ranges
        ax.set(xlim=[0, width], ylim=[0, height], aspect="equal")
        ax.axis(axis)

        return ax

    if isinstance(struct, pd.Series):
        struct = struct.to_dict()

    if isinstance(next(iter(struct), None), Structure) or (
        isinstance(struct, dict) and {*map(type, struct.values())} == {Structure}
    ):
        # Multiple structures, create grid of subplots
        n_structs = len(structures := struct)
        n_cols = min(n_cols, n_structs)
        n_rows = math.ceil(n_structs / n_cols)

        subplot_kwargs = dict(figsize=(3 * n_cols, 3 * n_rows), **subplot_kwargs or {})
        fig, axs = plt.subplots(n_rows, n_cols, **subplot_kwargs)

        for idx, (struct_or_key, ax) in enumerate(zip(structures, axs.flat), start=1):
            if isinstance(structures, dict):
                key = struct_or_key
                struct_i = structures[key]
            elif isinstance(struct_or_key, Structure):
                key = idx
                struct_i = struct_or_key
            else:
                raise TypeError(f"Expected pymatgen Structure or dict, got {struct=}")

            props = struct_i.properties
            if id_key := next(iter(set(props) & {Key.mat_id, "id", "ID"}), None):
                sub_title = props[id_key]
            elif isinstance(key, int):
                spg_num = struct_i.get_space_group_info()[1]
                sub_title = f"{struct_i.formula} ({spg_num})"
            else:
                sub_title = key
            if callable(subplot_title):
                sub_title = subplot_title(struct_i, key)

            plot_structure_2d(
                struct_i,
                ax=ax,
                rotation=rotation,
                atomic_radii=atomic_radii,
                elem_colors=elem_colors,
                scale=scale,
                show_unit_cell=show_unit_cell,
                show_bonds=show_bonds,
                site_labels=site_labels,
                label_kwargs=label_kwargs,
                bond_kwargs=bond_kwargs,
                standardize_struct=standardize_struct,
                axis=axis,
            )
            ax.set_title(f"{idx}. {sub_title}", fontsize=14)

        # Hide unused axes
        for ax in axs.flat[n_structs:]:
            ax.axis("off")

        return fig, axs

    raise TypeError(f"Expected pymatgen Structure or Sequence of them, got {struct=}")
