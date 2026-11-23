"""Brillouin zone visualization widget for Jupyter notebooks."""

from __future__ import annotations

import functools

from pymatviz.widgets._normalize import normalize_plot_json, structure_to_dict
from pymatviz.widgets._schema_traits import BrillouinZoneTraits
from pymatviz.widgets._traits import JsonBool, normalized_trait
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
    default_k_path = JsonBool(default_value=True).tag(sync=True)

    _positional_traits = ("structure", "bz_data")
