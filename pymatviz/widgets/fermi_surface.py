"""Fermi surface visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets.matterviz import MatterVizWidget


class FermiSurfaceWidget(MatterVizWidget):
    """MatterViz widget for visualizing Fermi surfaces.

    Accepts pre-computed Fermi surface data (isosurfaces + reciprocal lattice) or
    raw band grid data (energies on a k-grid for marching cubes extraction). Both
    are plain JSON; the frontend packs them into typed arrays.

    ``fermi_data`` is a JSON mesh dict::

        {
            "isosurfaces": [
                {
                    "vertices": [[x, y, z], ...],  # Cartesian k-space rows
                    "faces": [[i, j, k], ...],  # vertex index rows (any polygon size)
                    "normals": [[nx, ny, nz], ...],  # optional, one row per vertex
                    "properties": [...],  # optional per-vertex scalar (e.g. velocity)
                    "band_index": 0,
                    "spin": "up" | "down" | None,
                },
            ],
            "k_lattice": [[...], [...], [...]],  # reciprocal lattice rows
            "fermi_energy": 5.0,  # eV
            "reciprocal_cell": "wigner_seitz" | "parallelepiped",
            "metadata": {"n_bands": 1, "n_surfaces": 1},
        }

    or an IFermi ``FermiSurface.as_dict()`` (``@class: "FermiSurface"`` with
    ``isosurfaces`` keyed by signed band index, negative = spin down, and
    ``reciprocal_space.reciprocal_lattice``), which the frontend converts itself.

    ``band_data`` is a band grid dict::

        {
            "energies": [[band_grid, ...]],  # [spin][band][kx][ky][kz] nested lists
            "k_grid": [nx, ny, nz],  # must match every band grid's shape
            "k_lattice": [[...], [...], [...]],
            "fermi_energy": 5.0,
            "n_bands": 4,
            "n_spins": 1,
            "origin": [0, 0, 0],  # optional
            "periodic": False,  # optional: True for k = i/n grids without endpoint
        }

    The frontend flattens the nested grids to z-fastest arrays (index
    ``(ix * ny + iy) * nz + iz``) once at the prop boundary.

    Examples:
        From pre-computed Fermi surface data:
        >>> from pymatviz import FermiSurfaceWidget
        >>> widget = FermiSurfaceWidget(fermi_data=fermi_dict)
        >>> widget

        From IFermi:
        >>> widget = FermiSurfaceWidget(fermi_data=ifermi_fermi_surface.as_dict())

        From raw band grid data:
        >>> widget = FermiSurfaceWidget(band_data=band_grid_dict)

        With custom options:
        >>> widget = FermiSurfaceWidget(
        ...     fermi_data=fermi_dict,
        ...     surface_opacity=0.8,
        ...     show_bz=True,
        ...     style="height: 600px;",
        ... )
    """

    fermi_data = tl.Dict(allow_none=True).tag(sync=True)
    band_data = tl.Dict(allow_none=True).tag(sync=True)

    # Display options
    mu = tl.Float(allow_none=True, default_value=None).tag(sync=True)
    representation = tl.Unicode(allow_none=True, default_value=None).tag(sync=True)
    surface_opacity = tl.Float(allow_none=True, default_value=None).tag(sync=True)
    show_bz = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    bz_opacity = tl.Float(allow_none=True, default_value=None).tag(sync=True)
    show_vectors = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    camera_projection = tl.Unicode(allow_none=True, default_value=None).tag(sync=True)

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

        super().__init__(
            widget_type="fermi_surface",
            fermi_data=fermi_data,
            band_data=band_data,
            **kwargs,
        )
