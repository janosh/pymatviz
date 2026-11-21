"""Fermi surface visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

from pymatviz.widgets._normalize import normalize_json_kwargs, normalize_plot_json
from pymatviz.widgets._schema_traits import FermiSurfaceTraits
from pymatviz.widgets._traits import color_bar_trait, color_scale_trait
from pymatviz.widgets.matterviz import MatterVizWidget


class FermiSurfaceWidget(FermiSurfaceTraits, MatterVizWidget):
    """MatterViz widget for visualizing Fermi surfaces.

    Accepts pre-computed Fermi surface data (isosurfaces + reciprocal lattice) or
    raw band grid data (energies on a k-grid for marching cubes extraction).
    ``fermi_data``: ``{isosurfaces: [{vertices, faces, normals?, band_index, spin}],
    k_lattice, fermi_energy}`` or an IFermi ``FermiSurface.as_dict()``; ``band_data``:
    ``{energies: [spin][band][kx][ky][kz], k_grid, k_lattice, fermi_energy}``.

    Examples:
        From pre-computed Fermi surface data:
        >>> import pymatviz as pmv
        >>> widget = pmv.fermi_surface(fermi_data=fermi_dict)
        >>> widget

        From raw band grid data:
        >>> widget = pmv.fermi_surface(band_data=band_grid_dict)

        With custom options:
        >>> widget = pmv.fermi_surface(
        ...     fermi_data=fermi_dict,
        ...     surface_opacity=0.8,
        ...     show_bz=True,
        ...     color_property="property",
        ...     color_bar={"title": "Fermi velocity"},  # None hides it
        ...     style="height: 600px;",
        ... )
    """

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
        """
        if (fermi_data is None) == (band_data is None):
            raise ValueError("Provide exactly one of fermi_data or band_data.")
        normalize_json_kwargs(kwargs, ("color_scale", "color_bar"), "FermiSurface")

        super().__init__(
            fermi_data=normalize_plot_json(fermi_data, "fermi_data"),
            band_data=normalize_plot_json(band_data, "band_data"),
            **kwargs,
        )
