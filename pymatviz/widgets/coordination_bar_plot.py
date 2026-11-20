"""Coordination number bar plot widget computing CNs in the frontend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import traitlets as tl

from pymatviz.widgets._normalize import normalize_json_kwargs, structure_to_dict
from pymatviz.widgets._traits import (
    BONDING_STRATEGIES,
    PlotControlsTraits,
    optional_trait,
)
from pymatviz.widgets.matterviz import MatterVizWidget


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class CoordinationBarPlotWidget(PlotControlsTraits, MatterVizWidget):
    """Histogram of site coordination numbers, bonded in the frontend.

    MatterViz bonds each structure across its periodic boundaries with a bonding
    ``strategy`` (``"electroneg_ratio"`` or ``"explicit_only"``) and counts each
    site's neighbors. ``split_mode`` makes one series per element (summed over
    structures), per structure, or one for all sites (``"none"``).

    Examples:
        >>> from pymatgen.core import Lattice, Structure
        >>> from pymatviz.widgets import CoordinationBarPlotWidget
        >>> nacl = Structure(Lattice.cubic(5.6), ["Na", "Cl"], [[0] * 3, [0.5] * 3])
        >>> CoordinationBarPlotWidget({"NaCl": nacl}, split_mode="by_element")
    """

    # {label: structure dict} (or a list of structure dicts)
    structures = tl.Any(allow_none=True).tag(sync=True)
    strategy = optional_trait(tl.CaselessStrEnum, values=BONDING_STRATEGIES)
    split_mode = optional_trait(
        tl.CaselessStrEnum, values=["by_element", "by_structure", "none"]
    )
    # two-way: bar arrangement, also switchable from the controls pane
    mode = tl.CaselessStrEnum(
        values=["overlay", "stacked", "grouped"], default_value="grouped"
    ).tag(sync=True)
    orientation = optional_trait(tl.CaselessStrEnum, values=["vertical", "horizontal"])
    show_legend = optional_trait(tl.Bool)
    display = optional_trait(tl.Dict)
    legend = optional_trait(tl.Dict)
    padding = optional_trait(tl.Dict)
    x_axis = optional_trait(tl.Dict)
    y_axis = optional_trait(tl.Dict)

    @tl.validate("structures")
    def _validate_structures(
        self, proposal: dict[str, Any]
    ) -> dict[str, dict[str, Any] | None] | list[dict[str, Any] | None] | None:
        """Convert structures to the ``{label: dict}`` or list form MatterViz takes,
        on construction and on later assignment.
        """
        structures = proposal["value"]
        if structures is None:
            return None
        try:
            is_struct_dict = isinstance(structures, dict) and "sites" in structures
            if hasattr(structures, "items") and not is_struct_dict:  # dict, Series
                return {
                    str(key): structure_to_dict(val) for key, val in structures.items()
                }
            if isinstance(structures, (list, tuple)):
                return [structure_to_dict(struct) for struct in structures]
            return [structure_to_dict(structures)]
        except TypeError as exc:
            raise tl.TraitError(str(exc)) from exc

    def __init__(
        self,
        structures: Any | Mapping[str, Any] | Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a coordination bar plot widget.

        Args:
            structures: One structure, a sequence of them or a ``{label: structure}``
                mapping. pymatgen, ASE and phonopy structures and structure dicts
                (``Structure.as_dict()``) are accepted.
            **kwargs: Widget traits, e.g. ``strategy``, ``split_mode``, ``mode``,
                ``orientation``, ``x_axis``, ``y_axis``.
        """
        json_keys = ("display", "legend", "padding", "x_axis", "y_axis")
        normalize_json_kwargs(kwargs, json_keys, "CoordinationBarPlot")
        super().__init__(
            widget_type="coordination_bar_plot",
            structures=structures,
            **kwargs,
        )
