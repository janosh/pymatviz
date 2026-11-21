"""Site and layout helpers for structure plots."""

from __future__ import annotations

import html
import os
from typing import TYPE_CHECKING, Any

import ipywidgets as ipw


if TYPE_CHECKING:
    from collections.abc import Mapping

    from pymatgen.core import Composition, Site, Species
    from pymatgen.core.periodic_table import DummySpecies, Element


MIN_ASE_CELL_VOLUME = 1e-6  # Å³


def get_site_species(site: Site) -> Composition | Species | Element | DummySpecies:
    """Species of an ordered site, composition of a disordered one."""
    return site.specie if site.is_ordered else site.species


def get_site_elements(site: Site) -> set[str]:
    """Element symbols on a site (several for disordered sites)."""
    return {species.symbol for species in site.species}


def get_site_symbol(site: Site) -> str:
    """Element symbol of a site, the highest-occupancy one for disordered sites."""
    el_amt_dict = site.species.get_el_amt_dict()
    return max(el_amt_dict, key=el_amt_dict.__getitem__)


class WidgetGrid(ipw.GridBox):
    """GridBox of titled widget cells that, with the ``PYMATVIZ_STATIC`` env var set,
    displays as an HTML grid of the cells' PNGs (like single widgets do) instead of
    live widgets.
    """

    def _repr_mimebundle_(self, **kwargs: Any) -> dict[str, Any] | None:
        """Live grid, or a static HTML grid of PNGs under ``PYMATVIZ_STATIC``."""
        if not os.environ.get("PYMATVIZ_STATIC"):
            return super()._repr_mimebundle_(**kwargs)
        from pymatviz.widgets.matterviz import static_png_bundle

        # cells share the notebook width, so smaller renders keep the output light
        img_kwargs = {"width": 360, "dpi": 96}

        figures = ""
        for title, widget in (cell.children for cell in self.children):
            png = static_png_bundle(widget, **img_kwargs)["image/png"]
            figures += (
                f"<figure style='margin:0'><figcaption>{title.value}</figcaption>"
                f"<img src='data:image/png;base64,{png}' style='width:100%'></figure>"
            )
        cols = self.layout.grid_template_columns
        style = f"display:grid;gap:8px;grid-template-columns:{cols}"
        return {
            "text/html": f"<div style='{style}'>{figures}</div>",
            "text/plain": "<WidgetGrid>",
        }


def widget_grid(widgets: Mapping[Any, ipw.DOMWidget], n_cols: int) -> WidgetGrid:
    """Lay out labeled widgets in a grid with n_cols columns. Each cell is a VBox of
    an HTML title and the widget, so ``grid.children[idx].children[1]`` is the
    idx-th widget.

    Raises:
        ValueError: If n_cols is not positive.
    """
    if n_cols < 1:
        raise ValueError(f"{n_cols=} must be positive")
    cells = [
        ipw.VBox([ipw.HTML(f"<b>{html.escape(str(label))}</b>"), widget])
        for label, widget in widgets.items()
    ]
    n_cols = min(n_cols, len(cells))
    return WidgetGrid(
        cells,
        layout=ipw.Layout(grid_template_columns=f"repeat({n_cols}, minmax(0, 1fr))"),
    )


def add_vacuum_if_needed(struct: Any) -> Any:
    """Center ASE Atoms without a proper cell (e.g. molecules) in 10 Å of vacuum."""
    from pymatviz.process_data import is_ase_atoms

    if is_ase_atoms(struct) and struct.cell.volume < MIN_ASE_CELL_VOLUME:
        struct = struct.copy()
        struct.center(vacuum=10.0)
    return struct
