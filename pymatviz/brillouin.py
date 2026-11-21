"""Plot the first Brillouin zone of crystal structures."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from pymatgen.core import Structure

from pymatviz.widgets.brillouin_zone import BrillouinZoneWidget


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import ipywidgets as ipw
    from pymatgen.core import IStructure

    from pymatviz.typing import AnyStructure


def seekpath_primitive(structure: Structure | IStructure) -> Structure:
    """Seekpath's standardized primitive cell, whose first Brillouin zone carries the
    standard high-symmetry k-path labels. Each distinct element occupancy is its own
    symmetry type, so disordered sites with different occupancies stay inequivalent
    while oxidation states (e.g. Fe2+ vs Fe3+) don't lower the symmetry. Primitive
    sites keep their input species if all input sites of their type share it, else
    (mixed oxidation states) the element occupancy.
    """
    import seekpath

    type_keys = list(
        dict.fromkeys(site.species.element_composition for site in structure)
    )
    type_nums = [
        type_keys.index(site.species.element_composition) for site in structure
    ]
    type_species: list[set[Any]] = [set() for _ in type_keys]
    for site, type_num in zip(structure, type_nums, strict=True):
        type_species[type_num].add(site.species)
    species_of_type = [
        next(iter(species)) if len(species) == 1 else type_key
        for type_key, species in zip(type_keys, type_species, strict=True)
    ]
    spglib_cell = (
        structure.lattice.matrix,
        structure.frac_coords,
        [type_num + 1 for type_num in type_nums],
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="dict interface .* is deprecated", module="spglib"
        )
        seekpath_dict = seekpath.get_path(spglib_cell)
    return Structure(
        seekpath_dict["primitive_lattice"],
        [
            species_of_type[type_num - 1]
            for type_num in seekpath_dict["primitive_types"]
        ],
        seekpath_dict["primitive_positions"],
    )


def brillouin_zone_3d(
    struct: AnyStructure | Sequence[AnyStructure] | Mapping[Any, AnyStructure],
    *,
    n_cols: int = 3,
    **kwargs: Any,
) -> BrillouinZoneWidget | ipw.GridBox:
    """Plot the first Brillouin zone of seekpath's standardized primitive cell of each
    structure as an interactive 3D scene.

    Args:
        struct (AnyStructure | Sequence | Mapping): Structure(s) to plot.
        n_cols (int): Number of grid columns when plotting several structures.
            Defaults to 3.
        **kwargs: Passed to every BrillouinZoneWidget, e.g. ``surface_color``,
            ``surface_opacity``, ``edge_color``, ``show_vectors``, ``show_ibz`` or
            ``style``.

    Returns:
        BrillouinZoneWidget | ipw.GridBox: The widget for a single structure. For
            several, an ipywidgets grid with one titled cell per structure, each cell
            a VBox whose ``children[1]`` is its BrillouinZoneWidget.

    Raises:
        TypeError: If an input is not a periodic structure.
        ValueError: If no structures are given or n_cols is not positive.
    """
    from pymatviz.process_data import normalize_periodic_structures
    from pymatviz.structure.helpers import widget_grid

    structures = normalize_periodic_structures(struct)
    widgets = {
        key: BrillouinZoneWidget(structure=seekpath_primitive(structure), **kwargs)
        for key, structure in structures.items()
    }
    if len(widgets) == 1:
        return next(iter(widgets.values()))
    return widget_grid(widgets, n_cols)
