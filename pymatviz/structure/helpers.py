"""Site, standardization and layout helpers for structure plots."""

from __future__ import annotations

import html
import warnings
from typing import TYPE_CHECKING, Any, cast

from pymatgen.core import Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


if TYPE_CHECKING:
    from collections.abc import Mapping

    import ipywidgets as ipw
    from pymatgen.core import IStructure, Site, Species, Structure
    from pymatgen.core.periodic_table import Element


NO_SYM_MSG = "Symmetry could not be determined, skipping standardization"
MIN_ASE_CELL_VOLUME = 1e-6  # Å³


def get_site_species(site: Site) -> Composition | Species | Element:
    """Get species/composition from a site (handles ordered and disordered sites)."""
    return cast(
        "Composition | Species | Element", getattr(site, "specie", site.species)
    )


def get_site_elements(site: Site) -> set[str]:
    """Get element symbols from a site (handles ordered and disordered sites)."""
    species = get_site_species(site)
    if isinstance(species, Composition):
        return {getattr(sp, "symbol", str(sp)) for sp in species}
    return {getattr(species, "symbol", str(species))}


def get_site_symbol(site: Site) -> str:
    """Get single element symbol for a site (picks highest occupancy for disordered)."""
    species = get_site_species(site)

    if isinstance(species, Composition):
        el_amt_dict = species.get_el_amt_dict()
        return max(el_amt_dict, key=el_amt_dict.__getitem__) if el_amt_dict else "X"
    return species.symbol


def standardize_struct(struct: Structure | IStructure) -> Structure | IStructure:
    """Convert to the conventional standard cell, carrying over site properties
    (forces, magmoms, ...). Warns with NO_SYM_MSG and returns the input unchanged if
    spglib can't determine the symmetry.
    """
    try:
        return SpacegroupAnalyzer(struct).get_conventional_standard_structure(
            keep_site_properties=True
        )
    except ValueError as exc:
        warnings.warn(
            f"{NO_SYM_MSG} for {struct.formula}: {exc}", UserWarning, stacklevel=2
        )
    return struct


def widget_grid(widgets: Mapping[Any, ipw.DOMWidget], n_cols: int) -> ipw.GridBox:
    """Lay out labeled widgets in a grid with n_cols columns. Each cell is a VBox of
    an HTML title and the widget, so ``grid.children[idx].children[1]`` is the
    idx-th widget.

    Raises:
        ValueError: If n_cols is not positive.
    """
    import ipywidgets as ipw

    if n_cols < 1:
        raise ValueError(f"{n_cols=} must be positive")
    cells = [
        ipw.VBox([ipw.HTML(f"<b>{html.escape(str(label))}</b>"), widget])
        for label, widget in widgets.items()
    ]
    n_cols = min(n_cols, len(cells))
    return ipw.GridBox(
        cells,
        layout=ipw.Layout(grid_template_columns=f"repeat({n_cols}, minmax(0, 1fr))"),
    )


def add_vacuum_if_needed(struct: Any) -> Any:
    """Add vacuum to ASE Atoms if they lack a proper cell."""
    from pymatviz.process_data import is_ase_atoms

    if is_ase_atoms(struct) and (
        not hasattr(struct, "cell")
        or struct.cell is None
        or (hasattr(struct.cell, "volume") and struct.cell.volume < MIN_ASE_CELL_VOLUME)
    ):
        # No proper cell - add vacuum for molecular systems
        struct = struct.copy()
        struct.center(vacuum=10.0)
    return struct
