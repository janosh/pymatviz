"""Helper functions for 2D and 3D plots of pymatgen structures with plotly."""

from __future__ import annotations

import functools
import itertools
import math
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import plotly.graph_objects as go
from plotly import colors as pcolors
from pymatgen.core import Composition, Lattice, PeriodicSite, Species, Structure
from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pymatviz import colors
from pymatviz.enums import ElemColorScheme, Key, SiteCoords
from pymatviz.typing import Xyz
from pymatviz.utils import df_ptable, pick_max_contrast_color


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import Any, Literal

    import pandas as pd
    from numpy.typing import ArrayLike
    from pymatgen.analysis.local_env import NearNeighbors

    from pymatviz.typing import AnyStructure, ColorType, Xyz


def get_struct_prop(
    struct: AnyStructure, struct_key: str | int, prop_name: str, func_param: Any
) -> Any:
    """Get a structure related value with standardized precedence handling.

    Precedence order:
    1. structure.properties[prop_name] or atoms.info[prop_name]
    2. func_param (if dict, use struct_key; otherwise use directly)

    Args:
        struct (AnyStructure): The pymatgen Structure or ASE Atoms object.
        struct_key (str | int): Key identifying this structure in a collection.
        prop_name (str): Name of the property to look for in structure.properties or
            atoms.info.
        func_param (Any): Function parameter value (can be dict for per-structure
            values).

    Returns:
        Any: Resolved property value following precedence.
    """
    from pymatviz.process_data import is_ase_atoms

    # Check structure/atoms properties first (highest precedence)
    prop_value = None
    if is_ase_atoms(struct):
        prop_value = struct.info.get(prop_name)
    elif hasattr(struct, "properties"):
        prop_value = struct.properties.get(prop_name)

    if prop_value is not None:
        return prop_value

    # Fall back to function param
    if isinstance(func_param, dict):
        # For dict params, use struct_key to get per-structure value
        return func_param.get(struct_key, None)

    # For non-dict params, use directly
    return func_param


# fallback value (in nanometers) for covalent radius of an element
# see https://wikipedia.org/wiki/Atomic_radii_of_the_elements
missing_covalent_radius = 0.2
covalent_radii: pd.Series = df_ptable[Key.covalent_radius].fillna(
    missing_covalent_radius
)
NO_SYM_MSG = "Symmetry could not be determined, skipping standardization"
CELL_EDGES = (
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


def get_site_symbol(site: PeriodicSite) -> str:
    """Get a single element symbol for a site.

    Handles disordered sites by picking the element with the highest fraction.
    Handles Specie objects (e.g. Fe2+) by getting the element symbol.
    """
    if hasattr(site.species, "elements"):  # Likely Composition for disordered site
        # Element amounts in a disordered site (Composition)
        el_amt_dict = site.species.get_el_amt_dict()
        if el_amt_dict:
            # Get element with max amount
            return max(el_amt_dict, key=el_amt_dict.get)
        # Fallback for empty Composition or if get_el_amt_dict is not definitive
        if site.species.elements:
            return site.species.elements[0].symbol
        return "X"  # Should not happen for valid Compositions
    if hasattr(site.species, "symbol"):  # Element object
        return site.species.symbol
    if hasattr(site.species, "element"):  # Specie object (e.g. Fe2+)
        # Assuming site.species is a Pymatgen Specie object
        specie_obj = site.species
        if hasattr(specie_obj, "element") and isinstance(specie_obj.element, Element):
            return specie_obj.element.symbol
    # Fallback if it's just a string or other unknown type
    try:
        return site.species_string  # Last resort, might include oxidation state
    except AttributeError:
        return "X"  # Placeholder for unknown species type


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
    site: PeriodicSite,
    lattice: Lattice,
    cell_boundary_tol: float = 0.0,
    min_dist_dedup: float = 0.1,
) -> np.ndarray:
    """Get images for a given site in a lattice.

    Images are sites that are integer translations of the given site that are within
    or near the cell boundaries.

    Args:
        site (PeriodicSite): The site to get images for.
        lattice (Lattice): The lattice to get images for.
        cell_boundary_tol (float): Distance (in fractional coordinates) beyond the unit
            cell boundaries to include image atoms. Defaults to 0 for strict cell
            boundaries (only atoms with 0 <= coord <= 1). Higher values include atoms
            further outside the cell.
        min_dist_dedup (float): The min distance in Angstroms to any other site to avoid
            finding image atoms that are duplicates of original basis sites. Defaults to
            0.1.

    Returns:
        np.ndarray: Coordinates of all image sites.
    """
    coords_image_atoms: list[np.ndarray] = []

    # Generate all possible combinations of lattice vector offsets (except zero offset)
    offsets = set(itertools.product([-1, 0, 1], repeat=3)) - {(0, 0, 0)}

    for offset in offsets:
        new_frac = site.frac_coords + offset
        new_cart = lattice.get_cartesian_coords(new_frac)

        # Check if the new fractional coordinates are within cell bounds
        # Use cell_boundary_tol to control how far outside the cell atoms can be
        is_within_extended_cell = all(
            0 - cell_boundary_tol <= coord <= 1 + cell_boundary_tol
            for coord in new_frac
        )

        # filter sites that are too close to the original to avoid duplicates
        if is_within_extended_cell:
            distance_from_original = np.linalg.norm(
                new_cart - lattice.get_cartesian_coords(site.frac_coords)
            )

            if distance_from_original > min_dist_dedup:
                coords_image_atoms += [new_cart]

    return np.array(coords_image_atoms)


def cell_to_lines(cell: ArrayLike) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Convert lattice vectors to plot lines.

    Args:
        cell (np.array): Lattice vectors.

    Returns:
        tuple[np.array, np.array, np.array]:
        - Lines
        - z-indices that sort plot elements into out-of-plane layers
        - lines used to plot the cell
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
    if color_dict := getattr(colors, f"ELEM_COLORS_{str(elem_colors).upper()}", None):
        return color_dict
    raise ValueError(
        f"colors must be a dict or one of ('{', '.join(ElemColorScheme)}')"
    )


def get_atomic_radii(atomic_radii: float | dict[str, float] | None) -> dict[str, float]:
    """Get atomic radii based on the provided input."""
    if isinstance(atomic_radii, dict):
        return atomic_radii
    scale: float = 1.0 if atomic_radii is None else float(atomic_radii)
    return {elem: radius * scale for elem, radius in covalent_radii.items()}


def generate_site_label(
    site_labels: Literal["symbol", "species", "legend", False]
    | dict[str, str]
    | Sequence[str],
    site_idx: int,
    site: PeriodicSite,
) -> str | None:
    """Generate a label for a given site based on the site_labels strategy.

    Args:
        site_labels ("symbol" | "species" | "legend" | False] | dict[str, str]): The
            labeling strategy. If "legend", returns None. Can be "symbol", "species",
            "legend", False, a dict mapping symbols to custom labels, or a sequence of
            labels indexed by site position.
        site_idx (int): The index of the site.
        site: The site object.

    Returns:
        str | None: The generated label or None if no label should be shown.
    """
    if site_labels in (False, "legend"):
        return None

    if site_labels == "symbol":
        return get_site_symbol(site)
    if site_labels == "species":
        return site.species_string  # Use full species string for disordered

    label_text = ""
    symbol = get_site_symbol(site)  # Majority element symbol of site

    if isinstance(site_labels, dict):
        # Use provided label for symbol, else symbol itself, or empty if not found &
        # not True-like
        label_text = site_labels.get(symbol, symbol if site_labels else "")
    elif isinstance(site_labels, list):
        label_text = site_labels[site_idx] if site_idx < len(site_labels) else symbol

    return label_text


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
            from moyopy import MoyoDataset
            from moyopy.interface import MoyoAdapter

            spg_num = MoyoDataset(MoyoAdapter.from_py_obj(struct_i)).number
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
    float_fmt: str | Callable[[float], str] = ".4",
) -> str:
    """Generate hover text for a site based on the hover template.

    Args:
        site (PeriodicSite): The periodic site.
        hover_text (SiteCoords | Callable[[PeriodicSite], str]): The hover text template
            or a custom callable.
        majority_species (Species): The majority species at the site.
        float_fmt (str | Callable[[float], str]): Float formatting for coordinates. Can
            be an f-string format like ".4" (default) or a callable that takes a float
            and returns a string.

    Returns:
        str: The formatted hover text string.
    """
    if callable(hover_text):
        return hover_text(site)

    def format_coord(coord_val: float) -> str:
        """Format a coordinate value using the specified formatter."""
        if callable(float_fmt):
            return float_fmt(coord_val)
        # Convert to float to handle int coordinates properly
        formatted = f"{float(coord_val):{float_fmt}}"
        # For ints, remove unnecessary decimal places
        if coord_val == int(coord_val):
            return str(int(coord_val))
        return formatted

    cart_text = f"({', '.join(format_coord(c) for c in site.coords)})"
    frac_text = f"[{', '.join(format_coord(c) for c in site.frac_coords)}]"
    if hover_text == SiteCoords.cartesian:
        coords_text = cart_text
    elif hover_text == SiteCoords.fractional:
        coords_text = frac_text
    elif hover_text == SiteCoords.cartesian_fractional:
        coords_text = f"{cart_text} {frac_text}"
    else:
        raise ValueError(f"Invalid {hover_text=}")

    prefix = (
        "<b>Image of "
        if site.properties and site.properties.get("is_image")
        else "<b>Site: "
    )
    out_text = f"{prefix}{majority_species}</b><br>Coordinates {coords_text}"

    # Append other properties, excluding "is_image" if it was added
    other_props = {k: v for k, v in (site.properties or {}).items() if k != "is_image"}
    if other_props:
        out_text += "<br>Properties: " + ", ".join(
            f"{k}: {v}" for k, v in other_props.items()
        )
    return out_text


def normalize_elem_color(raw_color_from_map: ColorType) -> str:
    """Process a color from the element color map into a consistent RGB string format.

    Args:
        raw_color_from_map (ColorType): Color value from the element color map (tuple
            or string).

    Returns:
        str: Color in RGB format like "rgb(128,128,128)"
    """
    if (
        isinstance(raw_color_from_map, tuple)
        and len(raw_color_from_map) == 3
        and all(isinstance(c, (float, int)) for c in raw_color_from_map)
    ):
        r, g, b = (
            int(c * 255) if isinstance(c, float) and 0 <= c <= 1 else int(c)
            for c in raw_color_from_map
        )
        return f"rgb({r},{g},{b})"
    if isinstance(raw_color_from_map, str):
        return raw_color_from_map
    return "rgb(128,128,128)"  # Fallback gray for unexpected color types


def draw_site(
    fig: go.Figure,
    site: PeriodicSite,
    coords: np.ndarray,
    site_idx: int,
    site_labels: Any,
    elem_colors: dict[str, ColorType],
    atomic_radii: dict[str, float],
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
    float_fmt: str | Callable[[float], str] = ".4",
    legendgroup: str | None = None,
    showlegend: bool = False,
    legend: str = "legend",
    **kwargs: Any,
) -> None:
    """Add a site (regular or image) to the plot.

    Args:
        fig (go.Figure): The plotly figure to add the site to.
        site (PeriodicSite): The periodic site to draw.
        coords (np.ndarray): The coordinates of the site.
        site_idx (int): The index of the site.
        site_labels (str | dict | list): How to label the site.
        elem_colors (dict[str, ColorType]): Element color mapping.
        atomic_radii (dict[str, float]): Atomic radii mapping.
        atom_size (float): Scaling factor for atom sizes.
        scale (float): Overall scaling factor.
        site_kwargs (dict[str, Any]): Additional keyword arguments for site styling.
        is_image (bool): Whether this is an image site.
        is_3d (bool): Whether this is a 3D plot.
        row (int | None): Row for subplot.
        col (int | None): Column for subplot.
        scene (str | None): Scene name for 3D plots.
        hover_text (SiteCoords | Callable[[PeriodicSite], str]): Hover text template.
        float_fmt (str | Callable[[float], str]): Float formatting for hover
            coordinates.
        legendgroup (str | None): For interactive legend. If None (default), will be
            set to the site's species symbol.
        showlegend (bool): Whether to show this trace in the legend.
        legend (str): The legend group for the site. If None (default), will be set to
            the site's species symbol.
        **kwargs: Additional keyword arguments.
    """
    species = getattr(site, "specie", site.species)

    # Check if this is a disordered site (multiple species)
    if isinstance(species, Composition) and len(species) > 1:
        draw_disordered_site(
            fig=fig,
            site=site,
            coords=coords,
            site_idx=site_idx,
            site_labels=site_labels,
            elem_colors=elem_colors,
            atomic_radii=atomic_radii,
            atom_size=atom_size,
            scale=scale,
            site_kwargs=site_kwargs,
            is_image=is_image,
            is_3d=is_3d,
            row=row,
            col=col,
            scene=scene,
            hover_text=hover_text,
            float_fmt=float_fmt,
            legendgroup=legendgroup,
            showlegend=showlegend,
            legend=legend,
            **kwargs,
        )
        return

    # Handle ordered sites (single species)
    majority_species = (
        max(species, key=species.get) if isinstance(species, Composition) else species
    )
    if not isinstance(majority_species, Species):
        majority_species = Species(str(majority_species))
        # could add Species(get_site_symbol(site)) fallback for
        # unexpected/placeholder Species(symbol)

    site_radius = atomic_radii[majority_species.symbol] * scale
    raw_color_from_map = elem_colors.get(majority_species.symbol, "gray")

    # Process the color from the map into a string format
    atom_color = normalize_elem_color(raw_color_from_map)

    site_hover_text = get_site_hover_text(site, hover_text, majority_species, float_fmt)

    txt = generate_site_label(site_labels, site_idx, site)

    marker_kwargs = dict(
        size=site_radius * atom_size,
        color=atom_color,
        opacity=0.8 if is_image else 1,
        line=dict(width=1, color="rgba(0,0,0,0.4)"),  # Dark border
    )
    marker_kwargs.update(site_kwargs)

    # Calculate text color based on background color for max contrast
    text_color = pick_max_contrast_color(atom_color)
    scatter_kwargs = dict(
        x=[coords[0]],
        y=[coords[1]],
        mode="markers+text" if txt else "markers",
        marker=marker_kwargs,
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
        showlegend=showlegend,
        legendgroup=legendgroup,
        legend=legend,
    )
    scatter_kwargs |= kwargs

    if is_3d:
        scatter_kwargs["z"] = [coords[2]]
        fig.add_scatter3d(**scatter_kwargs, scene=scene)
    else:
        fig.add_scatter(**scatter_kwargs, row=row, col=col)


def get_disordered_site_legend_name(
    sorted_species: list[tuple[Species, float]], *, is_image: bool = False
) -> str:
    """Create a legend name for a disordered site showing all elements with occupancies.

    Args:
        sorted_species (list[tuple[Species, float]]): List of (Species, occupancy)
            tuples sorted by occupancy
        is_image (bool): Whether this is an image site

    Returns:
        str: Combined legend name like "Fe₀.₇₅Ni₀.₂₅" or "0.75Fe,0.25Ni"
    """
    # Format each species with its occupancy
    species_parts = []
    for element_species, occupancy in sorted_species:
        elem_symbol = element_species.symbol
        if occupancy == 1.0:
            species_parts.append(elem_symbol)
        else:
            # Use subscript numbers for fractional occupancies
            occupancy_str = f"{occupancy:.2f}".rstrip("0").rstrip(".")
            # Convert to subscript if possible (for common fractions)
            subscript_map = {
                "0": "₀",
                "1": "₁",
                "2": "₂",
                "3": "₃",
                "4": "₄",
                "5": "₅",
                "6": "₆",
                "7": "₇",
                "8": "₈",
                "9": "₉",
                ".": ".",
            }
            try:
                subscript_occupancy = "".join(
                    subscript_map.get(c, c) for c in occupancy_str
                )
                species_parts.append(f"{elem_symbol}{subscript_occupancy}")
            except KeyError:
                # Fallback to prefix notation if subscript fails
                species_parts.append(f"{occupancy_str}{elem_symbol}")

    legend_name = "".join(species_parts)

    if is_image:
        legend_name = f"Image of {legend_name}"

    return legend_name


def draw_disordered_site(
    fig: go.Figure,
    site: PeriodicSite,
    coords: np.ndarray,
    site_idx: int,
    site_labels: Any,
    elem_colors: dict[str, ColorType],
    atomic_radii: dict[str, float],
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
    float_fmt: str | Callable[[float], str] = ".4",
    legendgroup: str | None = None,
    showlegend: bool = False,
    legend: str = "legend",
    **kwargs: Any,  # noqa: ARG001
) -> None:
    """Draw a disordered site as pie slices (2D) or multiple spheres (3D).

    For 2D plots, creates pie slices with different colors and radii.
    For 3D plots, creates multiple spheres at the same position with different sizes.

    Args:
        fig (go.Figure): The plotly figure to add the site to.
        site (PeriodicSite): The periodic site to draw (must be disordered).
        coords (np.ndarray): The coordinates of the site.
        site_idx (int): The index of the site.
        site_labels (str | dict | list): How to label the site.
        elem_colors (dict[str, ColorType]): Element color mapping.
        atomic_radii (dict[str, float]): Atomic radii mapping.
        atom_size (float): Scaling factor for atom sizes.
        scale (float): Overall scaling factor.
        site_kwargs (dict[str, Any]): Additional keyword arguments for site styling.
        is_image (bool): Whether this is an image site.
        is_3d (bool): Whether this is a 3D plot.
        row (int | None): Row for subplot.
        col (int | None): Column for subplot.
        scene (str | None): Scene name for 3D plots.
        hover_text (SiteCoords | Callable[[PeriodicSite], str]): Hover text template.
        float_fmt (str | Callable[[float], str]): Float format for hover coordinates.
        legendgroup (str | None): For interactive legend. If None, will be set to
            site_idx for grouping all parts of this disordered site.
        showlegend (bool): Whether to show this trace in the legend.
        legend (str): The legend group for the site.
        **kwargs: Unused extra keyword arguments.
    """
    species = getattr(site, "specie", site.species)

    if not isinstance(species, Composition) or len(species) <= 1:
        # Not a disordered site, should use regular draw_site
        return

    # Sort species by occupancy for consistent ordering
    sorted_species = sorted(species.items(), key=lambda x: x[1], reverse=True)

    # Create a combined legend name showing all elements with occupancies
    legend_name = get_disordered_site_legend_name(sorted_species, is_image=is_image)

    # Set up legendgroup - use site_idx if not provided to group all parts together
    if legendgroup is None:
        legendgroup = f"disordered_site_{site_idx}"

    # Determine if we should show labels (fixed redundant condition)
    should_show_labels = site_labels not in ("legend", False)

    if is_3d:
        # For 3D: Create spherical wedges (3D pie slices) using meshes
        # Calculate the total base radius to use for wedge sizing
        base_radii = [
            atomic_radii.get(elem_spec.symbol, missing_covalent_radius) * scale
            for elem_spec, _ in sorted_species
        ]
        max_base_radius = max(base_radii)

        # Track the current angular position
        current_angle = 0.0

        for species_idx, (element_species, occupancy) in enumerate(sorted_species):
            elem_symbol = element_species.symbol
            raw_color_from_map = elem_colors.get(elem_symbol, "gray")

            atom_color = normalize_elem_color(raw_color_from_map)

            # Calculate the angular span for this species based on occupancy
            angle_span = 2 * np.pi * occupancy
            end_angle = current_angle + angle_span

            # Calculate radius for this wedge (proportional to occupancy)
            wedge_radius = max_base_radius * np.sqrt(occupancy)

            # Generate the spherical wedge mesh
            x_coords, y_coords, z_coords, i_indices, j_indices, k_indices = (
                get_spherical_wedge_mesh(
                    center=coords,
                    radius=wedge_radius,
                    start_angle=current_angle,
                    end_angle=end_angle,
                    n_theta=max(
                        MIN_3D_WEDGE_RESOLUTION_THETA,
                        int(MAX_3D_WEDGE_RESOLUTION_THETA * occupancy),
                    ),
                    n_phi=max(
                        MIN_3D_WEDGE_RESOLUTION_PHI,
                        int(MAX_3D_WEDGE_RESOLUTION_PHI * occupancy),
                    ),
                )
            )

            # Generate hover text for this species using the hover_text template
            site_hover_text = get_site_hover_text(
                site, hover_text, element_species, float_fmt
            )
            # Append species-specific information
            site_hover_text += f"<br>Species: {elem_symbol} ({occupancy:.2f})"

            # Add the spherical wedge
            fig.add_mesh3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                i=i_indices,
                j=j_indices,
                k=k_indices,
                color=atom_color,
                name=legend_name,  # Use combined legend name
                opacity=0.8 if is_image else 1,
                showlegend=showlegend
                and species_idx == 0,  # Only first species shows in legend
                legendgroup=legendgroup,  # Group all parts together
                scene=scene,
                showscale=False,
                hoverinfo="text",
                hovertext=site_hover_text,
                legend=legend,
            )

            # Add text label if needed
            if should_show_labels:
                # Calculate the label position at the center of the wedge
                label_angle = current_angle + angle_span / 2  # Middle of the wedge
                label_offset = max_base_radius * LABEL_OFFSET_3D_FACTOR
                label_radius = wedge_radius + label_offset

                # Calculate label position in 3D space
                label_x = coords[0] + label_radius * np.cos(label_angle)
                label_y = coords[1] + label_radius * np.sin(label_angle)
                label_z = coords[2]  # Keep at same Z level

                # Get the text for this element
                if isinstance(site_labels, dict):
                    txt = site_labels.get(elem_symbol, elem_symbol)
                elif site_labels == "species":
                    txt = str(element_species)
                else:  # site_labels == "symbol" or other modes
                    txt = elem_symbol

                text_color = pick_max_contrast_color(atom_color)

                # Apply site_kwargs to text styling if relevant
                text_kwargs = dict(
                    x=[label_x],
                    y=[label_y],
                    z=[label_z],
                    mode="text",
                    text=txt,
                    textposition="middle center",
                    textfont=dict(
                        color=text_color,
                        size=np.clip(
                            atom_size * max_base_radius * (0.8 if is_image else 1),
                            8,
                            16,
                        ),
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=legendgroup,  # Group with the main traces
                    scene=scene,
                )
                # Apply any text-specific styling from site_kwargs
                if isinstance(text_kwargs["textfont"], dict) and isinstance(
                    site_kwargs.get("textfont"), dict
                ):
                    text_kwargs["textfont"] |= site_kwargs["textfont"]

                fig.add_scatter3d(**text_kwargs)

            # Update the angle for the next species
            current_angle = end_angle

    else:
        # For 2D: Create pie slices using scatter traces instead of shapes
        # so they can be part of legendgroup for proper legend interaction
        # Calculate the total base radius to use for pie sizing
        base_radii = [
            atomic_radii.get(elem_spec.symbol, missing_covalent_radius) * scale
            for elem_spec, _ in sorted_species
        ]
        max_base_radius = max(base_radii)

        # Track the current angular position for pie slices
        current_angle = 0.0

        for species_idx, (element_species, occupancy) in enumerate(sorted_species):
            elem_symbol = element_species.symbol
            raw_color_from_map = elem_colors.get(elem_symbol, "gray")
            atom_color = normalize_elem_color(raw_color_from_map)

            # Calculate angular width for this species based on occupancy
            angular_width = 2 * math.pi * occupancy
            end_angle = current_angle + angular_width

            # Calculate radius for this pie slice
            slice_radius = max_base_radius * atom_size * 0.01  # Use scaling factor

            # Generate points for the pie slice
            n_points = max(8, int(16 * occupancy))  # More points for larger slices
            angles = np.linspace(current_angle, end_angle, n_points)

            # Create pie slice coordinates (including center point)
            slice_x = (
                [coords[0]]
                + [coords[0] + slice_radius * math.cos(angle) for angle in angles]
                + [coords[0]]
            )
            slice_y = (
                [coords[1]]
                + [coords[1] + slice_radius * math.sin(angle) for angle in angles]
                + [coords[1]]
            )

            # Generate hover text for this species using the hover_text template
            site_hover_text = get_site_hover_text(
                site, hover_text, element_species, float_fmt
            )
            # Append species-specific information
            site_hover_text += f"<br>Species: {elem_symbol} ({occupancy:.2f})"

            # Create a filled scatter trace for the pie slice
            fig.add_scatter(
                x=slice_x,
                y=slice_y,
                mode="lines",
                fill="toself",
                fillcolor=atom_color,
                line=dict(color=atom_color, width=1),
                opacity=0.8 if not is_image else 0.6,
                hoverinfo="skip",  # Hover handled by separate invisible trace below
                showlegend=False,  # Don't show in legend directly
                # same legend item as other parts of this disordered site
                legendgroup=legendgroup,
                name=legend_name,  # Use combined legend name
                row=row,
                col=col,
            )

            # Add invisible scatter point for hover and legend control
            # This point controls the entire disordered site in the legend
            fig.add_scatter(
                x=[coords[0]],
                y=[coords[1]],
                mode="markers",
                marker=dict(
                    size=0.1,  # Nearly invisible
                    color=atom_color,
                    opacity=0.01,  # Nearly transparent
                ),
                hoverinfo="text",
                hovertext=site_hover_text,
                showlegend=showlegend
                and species_idx == 0,  # Only first species shows in legend
                name=legend_name,  # Use combined legend name
                # same legend item as other parts of this disordered site
                legendgroup=legendgroup,
                # add to correct sublegend when plotting multiple structures
                legend=legend,
                row=row,
                col=col,
            )

            # Add text label if needed
            if should_show_labels:
                # Calculate the label position at the center of the slice
                label_angle = current_angle + angular_width / 2  # Middle of the slice
                label_offset = slice_radius * 0.3  # LABEL_OFFSET_2D_FACTOR
                label_radius = slice_radius + label_offset

                # Calculate label position
                label_x = coords[0] + label_radius * math.cos(label_angle)
                label_y = coords[1] + label_radius * math.sin(label_angle)

                # Get the text for this element
                if isinstance(site_labels, dict):
                    txt = site_labels.get(elem_symbol, elem_symbol)
                elif site_labels == "species":
                    txt = str(element_species)
                else:  # site_labels == "symbol" or other modes
                    txt = elem_symbol

                text_color = pick_max_contrast_color(atom_color)

                # Apply site_kwargs to text styling if relevant
                text_kwargs = dict(
                    x=[label_x],
                    y=[label_y],
                    mode="text",
                    text=txt,
                    textposition="middle center",
                    textfont=dict(
                        color=text_color,
                        size=np.clip(
                            atom_size * max_base_radius * (0.8 if is_image else 1),
                            8,
                            16,
                        ),
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=legendgroup,  # Group with the main traces
                    row=row,
                    col=col,
                )
                # Apply any text-specific styling from site_kwargs
                if isinstance(text_kwargs["textfont"], dict) and isinstance(
                    site_kwargs.get("textfont"), dict
                ):
                    text_kwargs["textfont"] |= site_kwargs["textfont"]

                fig.add_scatter(**text_kwargs)

            current_angle = end_angle


# Constants for disordered site rendering
LABEL_OFFSET_3D_FACTOR = 0.3  # Factor for 3D label offset from sphere surface
LABEL_OFFSET_2D_FACTOR = 0.3  # Factor for 2D label offset from pie slice
PIE_SLICE_COORD_SCALE = 0.01  # Scaling factor for pie slice coordinate units
MIN_3D_WEDGE_RESOLUTION_THETA = 8  # Minimum theta resolution for 3D wedges
MIN_3D_WEDGE_RESOLUTION_PHI = 6  # Minimum phi resolution for 3D wedges
MAX_3D_WEDGE_RESOLUTION_THETA = 16  # Base theta resolution for 3D wedges
MAX_3D_WEDGE_RESOLUTION_PHI = 24  # Base phi resolution for 3D wedges
MIN_PIE_SLICE_POINTS = 3  # Minimum points for 2D pie slices
MAX_PIE_SLICE_POINTS = 20  # Base points for 2D pie slices


def get_spherical_wedge_mesh(
    center: np.ndarray,
    radius: float,
    start_angle: float,
    end_angle: float,
    n_theta: int = 16,
    n_phi: int = 24,
) -> tuple[list[float], list[float], list[float], list[int], list[int], list[int]]:
    """Generate a spherical wedge (orange slice) mesh for 3D pie charts.

    Creates a wedge-shaped section of a sphere between two azimuthal angles,
    like a slice of an orange.

    Args:
        center (np.ndarray): Center coordinates (x, y, z) of the sphere
        radius (float): Radius of the sphere
        start_angle (float): Starting azimuthal angle in radians
        end_angle (float): Ending azimuthal angle in radians
        n_theta (int): Number of divisions in polar direction (from top to bottom)
        n_phi (int): Number of divisions in azimuthal direction (around the wedge)

    Returns:
        tuple: (x_coords, y_coords, z_coords, i_indices, j_indices, k_indices)
    """
    x_coords, y_coords, z_coords = [], [], []

    # Add center point
    center_idx = 0
    x_coords.append(center[0])
    y_coords.append(center[1])
    z_coords.append(center[2])

    # Generate points on sphere surface within the angular wedge
    vertex_map = {}  # Map (theta_idx, phi_idx) to vertex index

    # Create grid of points on sphere surface
    for theta_idx in range(n_theta + 1):  # Polar angle (0 to pi)
        theta = np.pi * theta_idx / n_theta  # From 0 (north pole) to pi (south pole)

        for phi_idx in range(n_phi + 1):  # Azimuthal angle within wedge
            phi = start_angle + (end_angle - start_angle) * phi_idx / n_phi

            # Spherical to cartesian coordinates
            x = center[0] + radius * np.sin(theta) * np.cos(phi)
            y = center[1] + radius * np.sin(theta) * np.sin(phi)
            z = center[2] + radius * np.cos(theta)

            vertex_idx = len(x_coords)
            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)

            vertex_map[(theta_idx, phi_idx)] = vertex_idx

    i_indices, j_indices, k_indices = [], [], []

    # Create triangular faces

    # 1. Curved surface triangles
    for theta_idx in range(n_theta):
        for phi_idx in range(n_phi):
            # Get four corners of this surface quad
            v00 = vertex_map[(theta_idx, phi_idx)]
            v01 = vertex_map[(theta_idx, phi_idx + 1)]
            v10 = vertex_map[(theta_idx + 1, phi_idx)]
            v11 = vertex_map[(theta_idx + 1, phi_idx + 1)]

            # Split quad into two triangles
            # Triangle 1: v00, v01, v10
            i_indices.append(v00)
            j_indices.append(v01)
            k_indices.append(v10)

            # Triangle 2: v01, v11, v10
            i_indices.append(v01)
            j_indices.append(v11)
            k_indices.append(v10)

    # 2. Side faces connecting center to edges (if not a full sphere)
    angle_span = end_angle - start_angle
    if angle_span < 2 * np.pi - 0.1:  # Not a complete sphere
        # Left side face (start_angle)
        for theta_idx in range(n_theta):
            v_top = vertex_map[(theta_idx, 0)]
            v_bottom = vertex_map[(theta_idx + 1, 0)]

            # Triangle: center, v_top, v_bottom
            i_indices.append(center_idx)
            j_indices.append(v_top)
            k_indices.append(v_bottom)

        # Right side face (end_angle)
        for theta_idx in range(n_theta):
            v_top = vertex_map[(theta_idx, n_phi)]
            v_bottom = vertex_map[(theta_idx + 1, n_phi)]

            # Triangle: center, v_bottom, v_top (opposite winding)
            i_indices.append(center_idx)
            j_indices.append(v_bottom)
            k_indices.append(v_top)

    return x_coords, y_coords, z_coords, i_indices, j_indices, k_indices


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


def draw_cell(
    fig: go.Figure,
    structure: Structure,
    cell_kwargs: dict[str, Any],
    *,
    is_3d: bool = True,
    row: int | None = None,
    col: int | None = None,
    scene: str | None = None,
    rotation_matrix: np.ndarray | None = None,
    show_faces: bool | dict[str, Any] = False,
) -> go.Figure:
    """Draw the cell of a structure in a 2D or 3D Plotly figure.

    Args:
        fig (go.Figure): The plotly figure to add the cell to.
        structure (Structure): The pymatgen structure.
        cell_kwargs (dict[str, Any]): Keyword arguments for cell styling.
        is_3d (bool): Whether this is a 3D plot. Defaults to True.
        row (int | None): Row for subplot. Defaults to None.
        col (int | None): Column for subplot. Defaults to None.
        scene (str | None): Scene name for 3D plots. Defaults to None.
        rotation_matrix (np.ndarray | None): Rotation matrix for 2D projections.
            Defaults to None.
        show_faces (bool | dict[str, Any]): Whether to show transparent cell
            surfaces. If a dict, will be used to customize surface appearance.
            Defaults to False.

    Returns:
        go.Figure: The updated plotly figure.
    """
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
    edge_kwargs = edge_defaults | cell_kwargs.get("edge", {})
    for idx, (start, end) in enumerate(CELL_EDGES):
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
            showlegend=False,
        )

    # Add corner spheres
    node_defaults = dict(size=3, color="black")
    node_kwargs = node_defaults | cell_kwargs.get("node", {})
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
            showlegend=False,
        )

    if show_faces:  # Add cell faces if requested
        surface_defaults = dict(color="rgba(255,255,255,0.1)", showscale=False)
        surface_kwargs = surface_defaults | ({} if show_faces is True else show_faces)

        if is_3d:
            # Define the 6 faces of the cell cube
            # Each face is defined by 4 corner indices
            faces = [
                [0, 1, 3, 2],  # bottom face (z=0)
                [4, 5, 7, 6],  # top face (z=1)
                [0, 1, 5, 4],  # front face (y=0)
                [2, 3, 7, 6],  # back face (y=1)
                [0, 2, 6, 4],  # left face (x=0)
                [1, 3, 7, 5],  # right face (x=1)
            ]

            for face_idx, face in enumerate(faces):
                face_corners = cart_corners[face]  # Get the 4 corners of this face

                # Split each rectangular face into 2 triangles
                triangles = [[0, 1, 2], [0, 2, 3]]

                for tri_idx, triangle in enumerate(triangles):
                    tri_corners = face_corners[triangle]
                    fig.add_mesh3d(
                        x=tri_corners[:, 0],
                        y=tri_corners[:, 1],
                        z=tri_corners[:, 2],
                        i=[0],
                        j=[1],
                        k=[2],
                        opacity=0.2,
                        color=surface_kwargs.get("color", "rgba(255,255,255,0.01)"),
                        showscale=surface_kwargs.get("showscale", False),
                        hoverinfo="skip",
                        name=f"surface-face{face_idx}-tri{tri_idx}",
                        showlegend=False,
                        scene=scene,
                    )
        else:  # For 2D, we can show projected outline of cell as filled polygon
            # Get convex hull of projected corners to create the outline
            from scipy.spatial import ConvexHull

            points_2d = cart_corners[:, :2]  # Use only x,y coords for 2D projection
            hull = ConvexHull(points_2d)
            hull_points = points_2d[hull.vertices]

            # Close the polygon by adding the first point at the end
            hull_x = np.append(hull_points[:, 0], hull_points[0, 0])
            hull_y = np.append(hull_points[:, 1], hull_points[0, 1])

            fig.add_scatter(
                x=hull_x,
                y=hull_y,
                mode="lines",
                fill="toself",
                fillcolor=surface_kwargs.get("color", "rgba(255,255,255,0.1)"),
                line=dict(width=0),  # No outline
                hoverinfo="skip",
                name="cell-face",
                showlegend=False,
                row=row,
                col=col,
            )

    return fig


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
    rotation_matrix: np.ndarray | None = None,
    elem_colors: dict[str, ColorType] | None = None,
    plotted_sites_coords: set[Xyz] | None = None,
) -> None:
    """Draw bonds between atoms in the structure.

    Args:
        fig (go.Figure): Plotly figure to add bonds to
        structure (Structure): This structure can be augmented to include primary sites
            and all relevant image sites to ensure comprehensive bond drawing.
        nn (NearNeighbors): NearNeighbors object to determine bonds
        is_3d: Whether the plot is 3D
        bond_kwargs (dict[str, Any] | None): Customization options for bonds. If
            bond_kwargs["color"] is a tuple/list of colors, gradient coloring will be
            used. If color=True and elem_colors is provided, the gradient will use the
            colors of the connected atoms.
        row (int | None): Row index for 2D subplots
        col (int | None): Column index for 2D subplots
        scene (str | None): Scene name for 3D plots
        rotation_matrix (np.ndarray | None): Rotation matrix for 2D plots
        elem_colors (dict[str, ColorType] | None): Element color map, used for gradient
            coloring if color=True
        plotted_sites_coords (set[Xyz] | None): Optional set of (x, y, z) tuples for
            sites that are actually plotted. If provided, bonds will only be drawn if
            both end points are in this set. Coordinates are expected to be rounded.
    """
    default_bond_color: ColorType | tuple[ColorType, ColorType] | bool = True
    if bond_kwargs and bond_kwargs.get("color") is False:
        default_bond_color = "darkgray"

    default_bond_kwargs = dict(color=default_bond_color, width=4)
    # User-provided bond_kwargs override defaults
    effective_bond_kwargs = default_bond_kwargs.copy()
    if bond_kwargs:
        effective_bond_kwargs.update(bond_kwargs)

    _elem_colors = elem_colors or get_elem_colors(ElemColorScheme.jmol)

    def parse_color(color_val: Any) -> str:
        """Convert various color formats to RGB string."""
        try:  # validate_colors returns a list of colors. We request 'rgb' type to get a
            # string 'rgb(r,g,b)'
            return pcolors.validate_colors([color_val], colortype="rgb")[0]
        except (ValueError, IndexError):
            return "rgb(128,128,128)"  # Fallback to gray if parsing fails

    for site_idx, site1 in enumerate(structure):
        try:
            connections = nn.get_nn_info(structure, n=site_idx)
        except ValueError as exc:
            if "No Voronoi neighbors found" in str(exc):
                warnings.warn(
                    f"Skipping bond drawing for {site_idx=} ({site1.species_string})"
                    f" due to CrystalNN error: {exc}",
                    UserWarning,
                    stacklevel=2,
                )
                continue  # Skip to the next site
            raise  # Re-raise other ValueErrors

        for con_dict in connections:
            site2 = structure[con_dict["site_index"]]
            jimage = con_dict["image"]

            coords_from = site1.coords
            actual_bonded_site_frac_coords = site2.frac_coords + jimage
            cart_coords_to = structure.lattice.get_cartesian_coords(
                actual_bonded_site_frac_coords
            )

            # If plotted_sites_coords is provided, check if both bond ends are plotted
            if plotted_sites_coords is not None:
                coords_from_rounded = tuple(np.round(coords_from, 5))
                cart_coords_to_rounded = tuple(np.round(cart_coords_to, 5))

                if (
                    coords_from_rounded not in plotted_sites_coords
                    or cart_coords_to_rounded not in plotted_sites_coords
                ):
                    continue  # Skip this bond if either end is not a plotted site

            current_bond_color_setting = effective_bond_kwargs["color"]
            color_for_segment_calc: tuple[str, str] | str

            if current_bond_color_setting is True:
                # Default gradient: use element colors of bonded sites
                elem1_symbol = get_site_symbol(site1)
                elem2_symbol = get_site_symbol(site2)
                color1_rgb_str = parse_color(_elem_colors.get(elem1_symbol, "gray"))
                color2_rgb_str = parse_color(_elem_colors.get(elem2_symbol, "gray"))
                color_for_segment_calc = (color1_rgb_str, color2_rgb_str)
            elif (
                isinstance(current_bond_color_setting, (list, tuple))
                and len(current_bond_color_setting) == 2
            ):
                # User-defined gradient: tuple of two colors
                color1_rgb_str = parse_color(current_bond_color_setting[0])
                color2_rgb_str = parse_color(current_bond_color_setting[1])
                color_for_segment_calc = (color1_rgb_str, color2_rgb_str)
            else:  # Solid color
                color_for_segment_calc = parse_color(current_bond_color_setting)

            n_segments = 1
            if isinstance(color_for_segment_calc, tuple):
                n_segments = 10  # Use more segments for gradients

            for segment_idx in range(n_segments):
                frac_start, frac_end = (
                    segment_idx / n_segments,
                    (segment_idx + 1) / n_segments,
                )
                current_segment_start = coords_from + frac_start * (
                    cart_coords_to - coords_from
                )
                current_segment_end = coords_from + frac_end * (
                    cart_coords_to - coords_from
                )

                segment_color_str: str
                if isinstance(color_for_segment_calc, tuple):
                    # Gradient calculation
                    segment_color_str = pcolors.find_intermediate_color(
                        color_for_segment_calc[0],
                        color_for_segment_calc[1],
                        (frac_start + frac_end) / 2,
                        colortype="rgb",
                    )
                else:  # Solid color
                    segment_color_str = parse_color(color_for_segment_calc)

                name = f"bond {site_idx}-{con_dict['site_index']} segment {segment_idx}"
                trace_kwargs = dict(
                    mode="lines",
                    line=dict(
                        width=effective_bond_kwargs.get("width", 2),
                        color=segment_color_str,
                        dash=effective_bond_kwargs.get("dash"),
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                    name=name,
                )

                if is_3d:
                    fig.add_scatter3d(
                        x=[current_segment_start[0], current_segment_end[0]],
                        y=[current_segment_start[1], current_segment_end[1]],
                        z=[current_segment_start[2], current_segment_end[2]],
                        scene=scene,
                        **trace_kwargs,
                    )
                else:
                    plot_segment_start = current_segment_start
                    plot_segment_end = current_segment_end
                    if rotation_matrix is not None:
                        plot_segment_start = np.dot(
                            current_segment_start, rotation_matrix
                        )
                        plot_segment_end = np.dot(current_segment_end, rotation_matrix)

                    fig.add_scatter(
                        x=[plot_segment_start[0], plot_segment_end[0]],
                        y=[plot_segment_start[1], plot_segment_end[1]],
                        row=row,
                        col=col,
                        **trace_kwargs,
                    )


def _standardize_struct(
    struct_i: Structure, *, standardize_struct: bool | None
) -> Structure:
    """Standardize the structure if needed."""
    if standardize_struct is None:
        standardize_struct = any(any(site.frac_coords < 0) for site in struct_i)
    if standardize_struct:
        try:
            spg_analyzer = SpacegroupAnalyzer(struct_i)
            return spg_analyzer.get_conventional_standard_structure()
        except ValueError:
            warnings.warn(NO_SYM_MSG, UserWarning, stacklevel=2)
    return struct_i


def _prep_augmented_structure_for_bonding(
    struct_i: Structure,
    *,
    show_image_sites: bool | dict[str, Any],
    cell_boundary_tol: float = 0,
) -> Structure:
    """Prepare an augmented structure including primary and optionally image sites for
    bonding.

    Args:
        struct_i (Structure): System to prepare.
        show_image_sites: Whether to include image sites.
        cell_boundary_tol (float): Distance beyond unit cell boundaries within which
            image atoms are included.
    """
    all_sites_for_bonding = [
        PeriodicSite(
            species=site_in_cell.species,
            coords=site_in_cell.frac_coords,
            lattice=struct_i.lattice,
            properties=site_in_cell.properties.copy() | dict(is_image=False),
            coords_are_cartesian=False,
        )
        for site_in_cell in struct_i
    ]

    if show_image_sites:  # True or a dict implies true for this purpose
        processed_image_coords: set[Xyz] = set()
        for site_in_cell in struct_i:
            image_cart_coords_arrays = get_image_sites(
                site_in_cell,
                struct_i.lattice,
                cell_boundary_tol=cell_boundary_tol,
            )
            for image_cart_coords_arr in image_cart_coords_arrays:
                coord_tuple_key = tuple(np.round(image_cart_coords_arr, 5))
                if coord_tuple_key not in processed_image_coords:
                    image_frac_coords = struct_i.lattice.get_fractional_coords(
                        image_cart_coords_arr
                    )
                    image_periodic_site = PeriodicSite(
                        site_in_cell.species,
                        image_frac_coords,
                        struct_i.lattice,
                        properties=site_in_cell.properties.copy() | dict(is_image=True),
                        coords_are_cartesian=False,
                    )
                    all_sites_for_bonding.append(image_periodic_site)
                    processed_image_coords.add(coord_tuple_key)

    return Structure.from_sites(
        all_sites_for_bonding, validate_proximity=False, to_unit_cell=False
    )


def configure_subplot_legends(
    fig: go.Figure,
    site_labels: Literal["symbol", "species", "legend", False]
    | dict[str, str]
    | Sequence[str],
    n_structs: int,
    n_cols: int,
    n_rows: int,
) -> None:
    """Configure legends for each subplot if site_labels is 'legend'."""
    if site_labels == "legend":
        for idx in range(1, n_structs + 1):
            row = (idx - 1) // n_cols + 1
            col = (idx - 1) % n_cols + 1

            # Calculate position within each subplot (bottom right)
            x_start = (col - 1) / n_cols
            x_end = col / n_cols
            y_start = 1 - row / n_rows
            y_end = 1 - (row - 1) / n_rows

            # Position legend much closer to bottom right of subplot
            legend_x = x_start + 0.98 * (x_end - x_start)
            legend_y = y_start + 0.02 * (y_end - y_start)

            # Position legend much closer to bottom right of subplot
            legend_x = x_start + 0.98 * (x_end - x_start)
            legend_y = y_start + 0.02 * (y_end - y_start)

            legend_key = "legend" if idx == 1 else f"legend{idx}"
            legend_config = dict(
                x=legend_x,
                y=legend_y,
                xref="paper",
                yref="paper",
                xanchor="right",
                yanchor="bottom",
                bgcolor="rgba(0,0,0,0)",  # Transparent background
                borderwidth=0,  # Remove border
                font=dict(size=12, weight="bold"),  # Larger and bold font
                itemsizing="constant",  # Keep legend symbols same size
                itemwidth=30,  # Min allowed
                tracegroupgap=2,  # Reduce vertical space between legend items
            )
            fig.layout[legend_key] = legend_config


def add_vacuum_if_needed(struct: Any) -> Any:
    """Add vacuum to ASE Atoms if they lack a proper cell."""
    from pymatviz.process_data import is_ase_atoms

    if is_ase_atoms(struct) and (
        not hasattr(struct, "cell")
        or struct.cell is None
        or (hasattr(struct.cell, "volume") and struct.cell.volume < 1e-6)
    ):
        # No proper cell - add vacuum for molecular systems
        struct = struct.copy()
        struct.center(vacuum=10.0)
    return struct
