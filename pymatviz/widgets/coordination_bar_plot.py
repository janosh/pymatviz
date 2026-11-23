"""Coordination number bar plot widget computing CNs in the frontend."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.utils import str_labels
from pymatviz.widgets._normalize import structure_to_dict
from pymatviz.widgets._schema_traits import CoordinationBarPlotTraits
from pymatviz.widgets._traits import PlotControlsTraits, normalized_trait
from pymatviz.widgets.matterviz import MatterVizWidget


def _labeled_structures(structures: Any) -> dict[str, Any] | None:
    """The bare structure dict or ``{label: dict}`` form MatterViz takes (sequences
    labeled ``Structure 1``, ...). Keys that stringify alike (e.g. 1 and "1") raise.
    """
    is_struct_dict = isinstance(structures, dict) and "sites" in structures
    if hasattr(structures, "items") and not is_struct_dict:  # dict, Series
        labeled = str_labels(dict(structures.items()))
        return {key: structure_to_dict(val) for key, val in labeled.items()}
    if isinstance(structures, (list, tuple)):
        return {
            f"Structure {idx}": structure_to_dict(struct)
            for idx, struct in enumerate(structures, start=1)
        }
    return structure_to_dict(structures)


class CoordinationBarPlotWidget(
    PlotControlsTraits, CoordinationBarPlotTraits, MatterVizWidget
):
    """Histogram of site coordination numbers, bonded in the frontend.

    MatterViz bonds each structure across its periodic boundaries with a bonding
    ``strategy`` (``"electroneg_ratio"`` or ``"explicit_only"``) and counts each
    site's neighbors. ``split_mode`` makes one series per element (summed over
    structures), per structure, or one for all sites (``"none"``).

    ``structures`` is one structure, a sequence of them or a ``{label:
    structure}`` mapping of pymatgen, ASE or phonopy structures or structure
    dicts.

    Examples:
        >>> import numpy as np
        >>> import kemio
        >>> nacl = kemio.Structure(
        ...     np.eye(3) * 5.6, ["Na", "Cl"], [[0] * 3, [0.5] * 3], fractional=True
        ... )
        >>> import pymatviz as pmv
        >>> pmv.coordination_bar_plot({"NaCl": nacl}, split_mode="by_element")
    """

    # a structure dict or {label: structure dict}, converted on every assignment
    structures = normalized_trait(_labeled_structures)
    # two-way: bar arrangement, also switchable from the controls pane
    mode = tl.CaselessStrEnum(
        values=["overlay", "stacked", "grouped"], default_value="grouped"
    ).tag(sync=True)

    _positional_traits = ("structures",)
