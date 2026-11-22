"""Brillouin zone visualization widget for Jupyter notebooks."""

from __future__ import annotations

import functools
from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import normalize_plot_json, structure_to_dict
from pymatviz.widgets._schema_traits import BrillouinZoneTraits
from pymatviz.widgets._traits import normalized_trait
from pymatviz.widgets.matterviz import MatterVizWidget


class BrillouinZoneWidget(BrillouinZoneTraits, MatterVizWidget):
    """MatterViz widget for visualizing Brillouin zones.

    Accepts a kemio or pymatgen Structure, ASE Atoms or PhonopyAtoms (the BZ is
    computed by the matterviz frontend), or pre-computed ``bz_data`` with vertices,
    faces, and edges.

    Examples:
        From a kemio Structure:
        >>> import numpy as np
        >>> import kemio
        >>> import pymatviz as pmv
        >>> struct = kemio.Structure(np.eye(3) * 4, ["Fe"], [[0, 0, 0]])
        >>> widget = pmv.brillouin_zone(structure=struct)
        >>> widget

        With custom options:
        >>> widget = pmv.brillouin_zone(
        ...     structure=struct,
        ...     show_ibz=True,
        ...     show_vectors=True,
        ...     style="height: 600px;",
        ... )
    """

    # normalized on every assignment: structure objects to dicts, numpy to lists
    structure = normalized_trait(functools.partial(structure_to_dict, periodic=True))
    bz_data = normalized_trait(
        functools.partial(normalize_plot_json, label="BrillouinZone.bz_data")
    )
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
        super().__init__(structure=structure, bz_data=bz_data, **kwargs)
