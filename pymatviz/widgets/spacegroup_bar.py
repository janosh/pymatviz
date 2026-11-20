"""Space group bar plot widget for Jupyter notebooks."""

from __future__ import annotations

from collections.abc import Mapping
from operator import index
from typing import Any

import numpy as np
import traitlets as tl

from pymatviz.widgets._traits import optional_trait
from pymatviz.widgets.matterviz import MatterVizWidget


def spacegroup_counts(data: Any) -> dict[str, int]:
    """Count space groups as ``{"<number>": count}`` (ascending numbers), so the
    widget syncs at most 230 entries however many structures went in.

    Args:
        data: Space group numbers, Hermann-Mauguin symbols or structures, or a
            ``{spacegroup: count}`` mapping (number/symbol keys may repeat a group).

    Raises:
        TypeError: If mapping counts aren't integers.
        ValueError: For negative counts or invalid space groups.
    """
    from pymatviz.process_data import normalize_spacegroups

    if data is None or len(data) == 0:
        return {}
    if isinstance(data, Mapping):
        try:
            counts = np.array([index(count) for count in data.values()], dtype=int)
        except TypeError as exc:
            raise TypeError(f"Space group counts must be integers, got {data}") from exc
        if (counts < 0).any():
            raise ValueError(f"Space group counts must be non-negative, got {data}")
        # keys one by one: they may mix numbers, symbols and (as synced) digit strings
        numbers = np.array(
            [
                normalize_spacegroups([int(key) if str(key).isdigit() else key]).iloc[0]
                for key in data
            ],
            dtype=int,
        )
    else:
        # object-dtype inputs (e.g. structure-derived columns) stay object after
        # normalizing, which bincount rejects
        numbers = normalize_spacegroups(data).to_numpy(dtype=int)
        counts = np.ones(len(numbers), dtype=int)
    totals = np.bincount(numbers, weights=counts, minlength=231)
    return {str(num): int(totals[num]) for num in np.flatnonzero(totals)}


class SpacegroupBarPlotWidget(MatterVizWidget):
    """MatterViz widget for space group frequency bar plots.

    Accepts space group numbers/symbols/structures or ``{spacegroup: count}``
    mappings and syncs them as per-space-group counts.

    Examples:
        >>> from pymatviz import SpacegroupBarPlotWidget
        >>> widget = SpacegroupBarPlotWidget(data=[225, 225, 166, 62, 62, 62])
        >>> widget.data
        {'62': 3, '166': 1, '225': 2}
    """

    data = tl.Any(allow_none=True).tag(sync=True)  # validated into {"<number>": count}
    show_counts = tl.Bool(default_value=True).tag(sync=True)
    show_legend = tl.Bool(default_value=False).tag(sync=True)
    orientation = tl.CaselessStrEnum(
        values=["vertical", "horizontal"], default_value="vertical"
    ).tag(sync=True)
    x_axis = optional_trait(tl.Dict)
    y_axis = optional_trait(tl.Dict)

    def __init__(self, data: Any = None, **kwargs: Any) -> None:
        """Initialize a space group bar plot widget.

        Args:
            data: Space group numbers/symbols/structures or a count mapping.
            **kwargs: Additional widget properties.
        """
        super().__init__(widget_type="spacegroup_bar", data=data, **kwargs)

    @tl.validate("data")
    def _validate_data(self, proposal: dict[str, Any]) -> dict[str, int]:
        """Count space groups on construction and reassignment."""
        try:
            return spacegroup_counts(proposal["value"])
        except (TypeError, ValueError) as exc:
            raise tl.TraitError(str(exc)) from exc
