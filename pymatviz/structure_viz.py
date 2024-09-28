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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch, Wedge
from matplotlib.path import Path
from plotly.subplots import make_subplots
from pymatgen.analysis.local_env import CrystalNN, NearNeighbors
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SymmetryUndeterminedError

from pymatviz.colors import ELEM_COLORS_JMOL, ELEM_COLORS_VESTA
from pymatviz.enums import ElemColorScheme, Key
from pymatviz.utils import ExperimentalWarning, df_ptable, pick_bw_for_contrast


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any, Literal

    import plotly.graph_objects as go
    from matplotlib.typing import ColorType
    from numpy.typing import ArrayLike


# fallback value (in nanometers) for covalent radius of an element
# see https://wikipedia.org/wiki/Atomic_radii_of_the_elements
missing_covalent_radius = 0.2
covalent_radii: pd.Series = df_ptable[Key.covalent_radius].fillna(
    missing_covalent_radius
)
NO_SYM_MSG = "Symmetry could not be determined, skipping standardization"


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


def structure_2d(
    struct: Structure | Sequence[Structure],
    *,
    ax: plt.Axes | None = None,
    rotation: str = "10x,8y,3z",
    atomic_radii: float | dict[str, float] | None = None,
    elem_colors: ElemColorScheme | dict[str, str | ColorType] = ElemColorScheme.jmol,
    scale: float = 1,
    show_unit_cell: bool = True,
    show_bonds: bool | NearNeighbors = False,
    site_labels: Literal["symbol", "species", False]
    | dict[str, str]
    | Sequence[str] = "species",
    label_kwargs: dict[str, Any] | None = None,
    bond_kwargs: dict[str, Any] | None = None,
    standardize_struct: bool | None = None,
    axis: bool | str = "off",
    n_cols: int = 4,
    subplot_kwargs: dict[str, Any] | None = None,
    subplot_title: Callable[[Structure, str | int], str] | None = None,
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

    plot_atoms(AseAtomsAdaptor().get_atoms(mp_19017), rotation="10x,8y,3z", radii=0.5)

    # pymatviz
    from pymatviz import structure_2d

    structure_2d(mp_19017)
    ```

    Multiple structures in single figure example:

    ```py
    from pymatgen.ext.matproj import MPRester
    from pymatviz import structure_2d

    structures = {
        (mp_id := f"mp-{idx}"): MPRester().get_structure_by_material_id(mp_id)
        for idx in range(1, 5)
    }
    structure_2d(structures)
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
        site_labels ("symbol" | "species" | False | dict[str, str] | Sequence):
            How to annotate lattice sites.
            If True, labels are element species (symbol + oxidation
            state). If a dict, should map species strings (or element symbols but looks
            for species string first) to labels. If a list, must be same length as the
            number of sites in the crystal. If a string, must be "symbol" or
            "species". "symbol" hides the oxidation state, "species" shows it
            (equivalent to True). Defaults to "species".
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
            Keyword arguments for plt.subplots(). Defaults to None. Use this to specify
            figsize and other figure customization.
        subplot_title (Callable[[Structure, str | int], str], optional): Should return
            subplot titles. Receives the structure and its key or index when passed as
            a dict or pandas.Series. Defaults to None in which case the title is the
            structure's material id if available, otherwise its formula and space group.

    Raises:
        ValueError: On invalid site_labels.

    Returns:
        plt.Axes | tuple[plt.Figure, np.ndarray[plt.Axes]]: Axes instance with plotted
            structure. If multiple structures are passed, returns both the parent Figure
            and its Axes.
    """
    if isinstance(struct, Structure):
        ax = ax or plt.gca()

        if subplot_kwargs is not None:
            warnings.warn(
                "subplot_kwargs are ignored when plotting a single structure",
                UserWarning,
                stacklevel=2,
            )

        if isinstance(site_labels, list | tuple) and len(site_labels) != len(struct):
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
            try:
                spg_analyzer = SpacegroupAnalyzer(struct)
                struct = spg_analyzer.get_conventional_standard_structure()
            except SymmetryUndeterminedError:
                warnings.warn(NO_SYM_MSG, UserWarning, stacklevel=2)

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
                        elem_color_symbols = ", ".join(elem_colors)
                        warn_msg = (
                            f"{elem_symbol=} not in elem_colors, using "
                            f"{fallback_color=}\nelement color palette specifies the "
                            f"following elements: {elem_color_symbols}"
                        )
                        warnings.warn(warn_msg, UserWarning, stacklevel=2)
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
                    # see test_structure_2d_color_schemes
                    wedge.elem_symbol, wedge.idx = elem_symbol, idx
                    ax.add_patch(wedge)

                    # Generate labels
                    if site_labels == "symbol":
                        txt = elem_symbol
                    elif site_labels == "species":
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
                    elif isinstance(site_labels, list | tuple):
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

                        text_kwargs = dict(
                            ha="center",
                            va="center",
                            zorder=zorder,
                            **(label_kwargs or {}),
                        )
                        ax.text(*(xy + text_offset), txt, **text_kwargs)

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
                "Warning: the show_bonds feature of structure_2d() is "
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
            # estimate bond connectivity. Use hasattr("oxi_state") on site.specie since
            # it's often a pymatgen Element which has no oxi_state
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

        subplot_kwargs = dict(figsize=(3 * n_cols, 3 * n_rows)) | (subplot_kwargs or {})
        fig, axs = plt.subplots(n_rows, n_cols, **subplot_kwargs)

        for idx, (struct_or_key, ax) in enumerate(
            zip(structures, axs.flat, strict=False), start=1
        ):
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
            elif isinstance(key, int):  # key=int means it's an index, i.e. not to be
                # used as title. instead make title from formula and space group number
                spg_num = struct_i.get_space_group_info()[1]
                sub_title = f"{struct_i.formula} (spg={spg_num})"
            else:
                sub_title = key
            if callable(subplot_title):
                sub_title = subplot_title(struct_i, key)

            structure_2d(
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
    site_labels: Literal["symbol", "species", False]
    | dict[str, str]
    | Sequence[str] = "species",
    standardize_struct: bool | None = None,
    n_cols: int = 4,
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
    if isinstance(struct, Structure):
        structures = {0: struct}
    elif isinstance(struct, pd.Series):
        structures = struct.to_dict()
    elif isinstance(next(iter(struct), None), Structure):
        structures = dict(enumerate(struct))
    elif isinstance(struct, dict) and {*map(type, struct.values())} == {Structure}:
        structures = struct
    else:
        raise TypeError(
            f"Expected pymatgen Structure or Sequence of them, got {struct=}"
        )

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

        # Get colors
        if str(elem_colors) == str(ElemColorScheme.jmol):
            _elem_colors = ELEM_COLORS_JMOL
        elif str(elem_colors) == str(ElemColorScheme.vesta):
            _elem_colors = ELEM_COLORS_VESTA
        elif isinstance(elem_colors, dict):
            _elem_colors = elem_colors
        else:
            raise ValueError(
                f"colors must be a dict or one of ('{', '.join(ElemColorScheme)}')"
            )

        # Get atomic radii
        if atomic_radii is None or isinstance(atomic_radii, float):
            _atomic_radii = (
                0.7 * df_ptable[Key.covalent_radius].fillna(0.2) * (atomic_radii or 1)
            )
        else:
            _atomic_radii = atomic_radii

        # Apply rotation
        rotation_matrix = _angles_to_rotation_matrix(rotation)
        rotated_coords = np.dot(struct_i.cart_coords, rotation_matrix)

        # Plot atoms
        if show_sites:
            site_kwargs = dict(line=dict(width=0.3, color="gray"))
            if isinstance(show_sites, dict):
                site_kwargs.update(show_sites)

            special_site_labels = ("symbol", "species")
            for site_idx, (site, coords) in enumerate(
                zip(struct_i, rotated_coords, strict=False)
            ):
                # if site is disordered, site.species will be Composition. use majority
                # species for now to determine site radius. TODO: display disordered
                # sites as circle wedges with multiple radii and species labels
                species = getattr(site, "specie", site.species)
                majority_species = (
                    max(species, key=species.get)
                    if isinstance(species, Composition)
                    else species
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

                # Generate labels
                if site_labels == "symbol":
                    txt = str(major_elem_symbol)
                elif site_labels == "species":
                    txt = str(majority_species)
                elif site_labels is False:
                    txt = ""
                elif isinstance(site_labels, dict):
                    # Try element incl. oxidation state as dict key first (e.g.
                    # Na+), then just element as fallback
                    txt = site_labels.get(
                        repr(major_elem_symbol), site_labels.get(major_elem_symbol, "")
                    )
                elif isinstance(site_labels, list | tuple):
                    txt = site_labels[site_idx]
                else:
                    raise ValueError(
                        f"Invalid {site_labels=}. Must be one of (bool, "
                        f"{', '.join(special_site_labels)}, dict, list)"
                    )

                fig.add_scatter(
                    x=[coords[0]],
                    y=[coords[1]],
                    mode="markers+text" if txt else "markers",
                    marker=(
                        dict(size=site_radius * atom_size, color=color) | site_kwargs
                    ),
                    text=txt,
                    textposition="middle center",
                    textfont=dict(  # Determine text color based on marker color
                        color=pick_bw_for_contrast(color, text_color_threshold=0.5),
                        size=np.clip(atom_size * site_radius, 12, 18),
                    ),
                    hovertext=hover_text,
                    hoverinfo="text",
                    hoverlabel=dict(namelength=-1),
                    name=str(majority_species),
                    showlegend=False,
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
                unit_cell_kwargs.update(show_unit_cell)

            for start, end in [
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
            ]:
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
        if callable(subplot_title):
            sub_title = subplot_title(struct_i, idx)
            anno = dict(text=sub_title) if isinstance(sub_title, str) else sub_title
        elif isinstance(struct_key, int):  # key=int means it's an index, i.e. not to be
            # used as title. instead make title from formula and space group number
            spg_num = struct_i.get_space_group_info()[1]
            sub_title = f"{struct_i.formula} (spg={spg_num})"
            anno = dict(text=f"{idx}. {sub_title}")
        else:
            anno = dict(text=struct_key)

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
