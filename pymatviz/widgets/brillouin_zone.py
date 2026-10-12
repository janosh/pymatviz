"""Brillouin zone visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._traits import optional_trait
from pymatviz.widgets.matterviz import MatterVizWidget


class BrillouinZoneWidget(MatterVizWidget):
    """MatterViz widget for visualizing Brillouin zones.

    Accepts a pymatgen Structure (BZ is computed by the matterviz frontend) or
    pre-computed BZ data with vertices, faces, and edges.

    Examples:
        From a pymatgen Structure:
        >>> from pymatviz import BrillouinZoneWidget
        >>> widget = BrillouinZoneWidget(structure=struct)
        >>> widget

        With pre-computed BZ data:
        >>> widget = BrillouinZoneWidget(bz_data=bz_dict)

        With custom options:
        >>> widget = BrillouinZoneWidget(
        ...     structure=struct,
        ...     show_ibz=True,
        ...     show_vectors=True,
        ...     style="height: 600px;",
        ... )
    """

    structure = tl.Dict(allow_none=True).tag(sync=True)
    bz_data = tl.Dict(allow_none=True).tag(sync=True)

    # Display options
    surface_color = optional_trait(tl.Unicode)
    surface_opacity = optional_trait(tl.Float)
    edge_color = optional_trait(tl.Unicode)
    edge_width = optional_trait(tl.Float)
    show_vectors = optional_trait(tl.Bool)
    show_ibz = optional_trait(tl.Bool)
    ibz_color = optional_trait(tl.Unicode)
    ibz_opacity = optional_trait(tl.Float)
    camera_projection = optional_trait(tl.Unicode)

    def __init__(
        self,
        structure: Any | None = None,
        bz_data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the BrillouinZoneWidget.

        Args:
            structure: A pymatgen Structure, ASE Atoms, PhonopyAtoms or structure
                dict (see ``structure_to_dict``). The BZ is computed by the matterviz
                frontend from the lattice.
            bz_data: Pre-computed BZ data with vertices, faces, and edges.
            **kwargs: Additional widget properties.
        """
        from pymatviz.widgets._normalize import normalize_plot_json, structure_to_dict

        super().__init__(
            widget_type="brillouin_zone",
            structure=structure_to_dict(structure),
            bz_data=normalize_plot_json(bz_data, "bz_data"),
            **kwargs,
        )
