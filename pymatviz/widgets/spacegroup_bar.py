"""Space group bar plot widget for Jupyter notebooks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sized
from numbers import Integral
from typing import Any

import numpy as np
import traitlets as tl

from pymatviz.widgets._normalize import materialize_iterator
from pymatviz.widgets._schema_traits import SpacegroupBarTraits
from pymatviz.widgets._traits import JsonBool, normalized_trait
from pymatviz.widgets.matterviz import MatterVizWidget


def _count_int(count: Any, data: Mapping[Any, Any]) -> int:
    """Int of a space group count: ints and integral floats (e.g. ``3.0`` from a
    float column) count, bools and fractional or non-finite floats don't.

    Raises:
        TypeError: For any other count, naming the whole mapping.
    """
    is_int = isinstance(count, Integral) and not isinstance(count, bool)
    if is_int or (isinstance(count, float | np.floating) and float(count).is_integer()):
        return int(count)
    raise TypeError(
        f"Space group counts must be integers (or integral floats), got {count!r} "
        f"in {data}"
    )


def spacegroup_counts(data: Any) -> dict[str, int]:
    """Count space groups as ``{"<number>": count}`` (ascending numbers), so the
    widget syncs at most 230 entries however many structures went in.

    Args:
        data: Space group numbers, Hermann-Mauguin symbols or structures (a
            sequence, array, Series or iterator such as a generator), or a
            ``{spacegroup: count}`` mapping (number/symbol keys may repeat a group).

    Raises:
        TypeError: For non-iterable data or mapping counts that aren't integers or
            integral floats (bools are rejected).
        ValueError: For negative counts or invalid space groups.
    """
    from pymatviz.structures import normalize_spacegroups

    data = materialize_iterator(data)  # e.g. generators, which have no len()
    if not isinstance(data, Sized):
        raise TypeError(
            "Space group data must be a sequence, iterator or mapping, got "
            f"{type(data).__name__}: {data!r}"
        )
    if len(data) == 0:
        return {}
    if isinstance(data, Mapping):
        counts = np.array([_count_int(count, data) for count in data.values()])
        if (counts < 0).any():
            raise ValueError(f"Space group counts must be non-negative, got {data}")
        # keys may mix numbers, symbols and (as synced) digit strings, so normalize
        # them in one batch per key type
        keys = [int(key) if str(key).isdigit() else key for key in data]
        positions_by_type: defaultdict[type, list[int]] = defaultdict(list)
        for pos, key in enumerate(keys):
            positions_by_type[type(key)].append(pos)
        numbers = np.empty(len(keys), dtype=int)
        for positions in positions_by_type.values():
            numbers[positions] = normalize_spacegroups(
                [keys[pos] for pos in positions]
            ).to_numpy(dtype=int)
    else:
        # object-dtype inputs (e.g. structure-derived columns) stay object after
        # normalizing, which bincount rejects
        numbers = normalize_spacegroups(data).to_numpy(dtype=int)
        counts = np.ones(len(numbers), dtype=int)
    totals = np.bincount(numbers, weights=counts, minlength=231)
    return {str(num): int(totals[num]) for num in np.flatnonzero(totals)}


class SpacegroupBarPlotWidget(SpacegroupBarTraits, MatterVizWidget):
    """MatterViz widget for space group frequency bar plots.

    Accepts space group numbers/symbols/structures or ``{spacegroup: count}``
    mappings and syncs them as per-space-group counts.

    Examples:
        >>> from pymatviz.widgets import SpacegroupBarPlotWidget
        >>> widget = SpacegroupBarPlotWidget(data=[225, 225, 166, 62, 62, 62])
        >>> widget.data
        {'62': 3, '166': 1, '225': 2}
    """

    data = normalized_trait(spacegroup_counts)  # into {"<number>": count}
    show_counts = JsonBool(default_value=True).tag(sync=True)
    show_legend = JsonBool(default_value=False).tag(sync=True)
    orientation = tl.CaselessStrEnum(
        values=["vertical", "horizontal"], default_value="vertical"
    ).tag(sync=True)

    _positional_traits = ("data",)
