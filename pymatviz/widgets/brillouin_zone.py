"""Brillouin zone visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._schema_traits import BrillouinZoneTraits
from pymatviz.widgets.matterviz import MatterVizWidget


class BrillouinZoneWidget(BrillouinZoneTraits, MatterVizWidget):
    """MatterViz widget for visualizing Brillouin zones.

    Accepts a pymatgen Structure (BZ is computed by the matterviz frontend) or
    pre-computed BZ data with vertices, faces, and edges.

    Examples:
        From a pymatgen Structure:
        >>> import pymatviz as pmv
        >>> widget = pmv.brillouin_zone(structure=struct)
        >>> widget

        With pre-computed BZ data:
        >>> widget = pmv.brillouin_zone(bz_data=bz_dict)

        With custom options:
        >>> widget = pmv.brillouin_zone(
        ...     structure=struct,
        ...     show_ibz=True,
        ...     show_vectors=True,
        ...     style="height: 600px;",
        ... )
    """

    default_k_path = tl.Bool(default_value=True).tag(sync=True)

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
            structure=structure_to_dict(structure, periodic=True),
            bz_data=normalize_plot_json(bz_data, "bz_data"),
            **kwargs,
        )
