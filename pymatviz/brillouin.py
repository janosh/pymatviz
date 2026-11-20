"""Plot the first Brillouin zone of crystal structures."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from pymatgen.core import Element, Structure

from pymatviz.widgets.brillouin_zone import BrillouinZoneWidget


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import ipywidgets as ipw
    from pymatgen.core import IStructure

    from pymatviz.typing import AnyStructure


def seekpath_primitive(structure: Structure | IStructure) -> Structure:
    """Seekpath's standardized primitive cell, whose first Brillouin zone carries the
    standard high-symmetry k-path labels.
    """
    import seekpath

    from pymatviz.structure.helpers import get_site_symbol

    spglib_cell = (
        structure.lattice.matrix,
        structure.frac_coords,
        [Element(get_site_symbol(site)).number for site in structure],
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="dict interface .* is deprecated", module="spglib"
        )
        seekpath_dict = seekpath.get_path(spglib_cell)
    return Structure(
        seekpath_dict["primitive_lattice"],
        seekpath_dict["primitive_types"],
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
