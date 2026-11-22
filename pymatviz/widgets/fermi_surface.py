"""Fermi surface visualization widget for Jupyter notebooks."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from pymatviz.widgets._normalize import normalize_plot_json
from pymatviz.widgets._schema_traits import FermiSurfaceTraits
from pymatviz.widgets._traits import (
    color_bar_trait,
    color_scale_trait,
    normalized_trait,
    validator,
)
from pymatviz.widgets.matterviz import MatterVizWidget


if TYPE_CHECKING:
    import traitlets as tl


class FermiSurfaceWidget(FermiSurfaceTraits, MatterVizWidget):
    """MatterViz widget for visualizing Fermi surfaces.

    Accepts pre-computed Fermi surface data (isosurfaces + reciprocal lattice) or
    raw band grid data (energies on a k-grid for marching cubes extraction).
    ``fermi_data``: ``{isosurfaces: [{vertices, faces, normals?, band_index, spin}],
    k_lattice, fermi_energy}`` or an IFermi ``FermiSurface.as_dict()``; ``band_data``:
    ``{energies: [spin][band][kx][ky][kz], k_grid, k_lattice, fermi_energy}``.

    Examples:
        From raw band grid data (one free-electron band, cut into a sphere):
        >>> import numpy as np
        >>> import pymatviz as pmv
        >>> k_axis = np.linspace(-0.5, 0.5, 8)
        >>> kx, ky, kz = np.meshgrid(k_axis, k_axis, k_axis, indexing="ij")
        >>> band_grid = {
        ...     "energies": [[kx**2 + ky**2 + kz**2]],  # [spin][band][kx][ky][kz]
        ...     "k_grid": [8, 8, 8],
        ...     "k_lattice": np.eye(3),
        ...     "fermi_energy": 0.15,
        ... }
        >>> widget = pmv.fermi_surface(band_data=band_grid)
        >>> widget

        Pre-computed isosurfaces go in ``fermi_data``, e.g.
        ``pmv.fermi_surface(fermi_data=ifermi_surface.as_dict())``.

        With custom options:
        >>> widget = pmv.fermi_surface(
        ...     band_data=band_grid,
        ...     surface_opacity=0.8,
        ...     show_bz=True,
        ...     color_property="property",
        ...     color_bar={"title": "Fermi velocity"},  # None hides it
        ...     style="height: 600px;",
        ... )
    """

    # JSON-normalized (numpy arrays to lists) on every assignment
    fermi_data = normalized_trait(
        functools.partial(normalize_plot_json, label="FermiSurface.fermi_data")
    )
    band_data = normalized_trait(
        functools.partial(normalize_plot_json, label="FermiSurface.band_data")
    )
    color_scale = color_scale_trait()
    # Shown only when color_property is "property"
    color_bar = color_bar_trait()

    def __init__(
        self,
        fermi_data: dict[str, Any] | None = None,
        band_data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the FermiSurfaceWidget.

        Args:
            fermi_data: Pre-computed Fermi surface data with isosurfaces and
                reciprocal lattice info.
            band_data: Raw band grid data with energies on a k-grid.
            **kwargs: Additional widget properties.

        Raises:
            TraitError: Unless exactly one of fermi_data and band_data is given.
        """
        super().__init__(fermi_data=fermi_data, band_data=band_data, **kwargs)

    @validator("fermi_data", "band_data")
    def _validate_one_source(self, proposal: tl.Bunch) -> Any:
        """Require exactly one of fermi_data and band_data, on construction and
        assignment, so the state (and its saved JSON) stays loadable. Inside
        ``hold_trait_notifications()`` this checks the final state, so swapping
        sources there works.
        """
        other = "band_data" if proposal["trait"].name == "fermi_data" else "fermi_data"
        if (proposal["value"] is None) == (getattr(self, other) is None):
            raise ValueError("Provide exactly one of fermi_data or band_data.")
        return proposal["value"]
