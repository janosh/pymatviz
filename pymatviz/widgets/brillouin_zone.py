"""Brillouin zone visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

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
    surface_color = tl.Unicode(allow_none=True, default_value=None).tag(sync=True)
    surface_opacity = tl.Float(allow_none=True, default_value=None).tag(sync=True)
    edge_color = tl.Unicode(allow_none=True, default_value=None).tag(sync=True)
    edge_width = tl.Float(allow_none=True, default_value=None).tag(sync=True)
    show_vectors = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    show_ibz = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    ibz_color = tl.Unicode(allow_none=True, default_value=None).tag(sync=True)
    ibz_opacity = tl.Float(allow_none=True, default_value=None).tag(sync=True)
    camera_projection = tl.Unicode(allow_none=True, default_value=None).tag(sync=True)

    def __init__(
        self,
        structure: Any | None = None,
        bz_data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the BrillouinZoneWidget.

        Args:
            structure: A pymatgen Structure, ASE Atoms, or structure dict. The BZ
                is computed by the matterviz frontend from the lattice.
            bz_data: Pre-computed BZ data with vertices, faces, and edges.
            **kwargs: Additional widget properties.
        """
        from pymatviz.widgets._normalize import normalize_structure_for_bz

        super().__init__(
            widget_type="brillouin_zone",
            structure=normalize_structure_for_bz(structure),
            bz_data=bz_data,
            **kwargs,
        )
