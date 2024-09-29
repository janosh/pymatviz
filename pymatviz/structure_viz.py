"""2D plots of pymatgen structures with matplotlib.

structure_2d() and its helpers get_rot_matrix() and unit_cell_to_lines() were
inspired by ASE https://wiki.fysik.dtu.dk/ase/ase/visualize/visualize.html#matplotlib.
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Callable, Sequence
from itertools import product
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.patches import PathPatch, Wedge
from matplotlib.path import Path
from plotly.subplots import make_subplots
from pymatgen.analysis.local_env import CrystalNN, NearNeighbors
from pymatgen.core import Composition, Lattice, PeriodicSite, Structure
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

        special_site_labels = ("symbol", "species", False)
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
                            f"Invalid {site_labels=}. Must be one of "
                            f"({', '.join(map(str, special_site_labels))}, dict, list)"
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
                        size=_atomic_radii[site.specie.symbol]
                        * scale
                        * atom_size
                        * 0.8,
                        color=_elem_colors.get(site.specie.symbol, "gray"),
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
                        size=_atomic_radii[site.specie.symbol]
                        * scale
                        * atom_size
                        * 0.8,
                        color=_elem_colors.get(site.specie.symbol, "gray"),
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
