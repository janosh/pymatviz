"""Sunburst of space groups grouped by crystal system."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, Literal

import matscikit as mk
import pandas as pd

from pymatviz.hierarchy import _limit_nodes, _validate_overflow
from pymatviz.structures import normalize_spacegroups
from pymatviz.widgets.sunburst import SunburstWidget


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatviz.typing import SpacegroupLike


def spacegroup_sunburst(
    data: Sequence[SpacegroupLike] | pd.Series,
    *,
    max_spacegroups: int | None = None,
    overflow: Literal["other", "drop"] = "other",
    **kwargs: Any,
) -> SunburstWidget:
    """Show space group counts with crystal systems as the inner ring.

    Args:
        data (Sequence[SpacegroupLike] | pd.Series): Space group numbers (1 -
            230), international symbols or structures (matscikit, pymatgen, ASE,
            phonopy).
        max_spacegroups (int | None): Maximum number of space groups to show for each
            crystal system. If None (default), all space groups are shown. If a
            positive integer, only the top N space groups by count are shown (ties
            keep the first seen).
        overflow ("other" | "drop"): How to handle space groups beyond
            max_spacegroups:
            - "other": Combine remaining space groups into an "Other" slice (default)
            - "drop": Discard remaining space groups entirely
        **kwargs: Passed to SunburstWidget, e.g. ``label_text`` (default
            ``"label+value"``), ``max_depth`` or ``style``.

    Returns:
        SunburstWidget: The MatterViz sunburst.

    Raises:
        ValueError: If data is empty, has missing values or invalid space groups, or
            overflow is invalid.
        TypeError: If the space group of a structure can't be determined.
    """
    _validate_overflow(overflow)
    if len(data) == 0:
        raise ValueError("spacegroup_sunburst requires non-empty data")

    numbers = normalize_spacegroups(data)  # merges spellings, settings, int/float
    # symbol inputs are labeled with their canonical short symbol, others by number
    as_symbols = isinstance(pd.Series(data).iloc[0], str)

    by_system: dict[str, list[dict[str, Any]]] = {}
    for number, count in Counter(numbers.tolist()).most_common():
        crystal_sys = mk.crystal_system(number)
        label = mk.space_group_symbol(number) if as_symbols else str(number)
        by_system.setdefault(crystal_sys, []).append(
            {"id": f"{crystal_sys}/{label}", "label": label, "value": count}
        )
    nodes = [
        {
            "id": crystal_sys,
            "label": crystal_sys,
            "children": _limit_nodes(
                children, max_spacegroups, overflow, parent_id=crystal_sys
            ),
        }
        for crystal_sys, children in by_system.items()
    ]
    return SunburstWidget(nodes, **{"label_text": "label+value"} | kwargs)
