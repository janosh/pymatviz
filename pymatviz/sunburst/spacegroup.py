"""Sunburst of space groups grouped by crystal system."""

from collections import Counter
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from pymatviz.hierarchy import _limit_nodes, _validate_overflow
from pymatviz.structures import get_spacegroup_number, is_structure_like
from pymatviz.utils import spg_to_crystal_sys
from pymatviz.widgets.sunburst import SunburstWidget


def spacegroup_sunburst(
    data: Sequence[Any] | pd.Series,
    *,
    max_spacegroups: int | None = None,
    overflow: Literal["other", "drop"] = "other",
    **kwargs: Any,
) -> SunburstWidget:
    """Show space group counts with crystal systems as the inner ring.

    Args:
        data (Sequence[int | str | Structure] | pd.Series): Space group numbers (1 -
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

    # materialize as list[Any] for Series-safe positional access and untyped iteration
    values: list[Any] = data.tolist() if isinstance(data, pd.Series) else list(data)
    if is_structure_like(values[0]):  # if 1st item is structure-like, assume all are
        spg_nums: list[Any] = []
        for idx, struct in enumerate(values):
            try:
                spg_nums.append(get_spacegroup_number(struct))
            except (TypeError, ValueError, RuntimeError) as exc:
                raise TypeError(
                    "Could not determine space group for structure at index "
                    f"{idx} ({type(struct).__name__}). Pass ordered periodic "
                    "structures."
                ) from exc
        values = spg_nums
    if pd.isna(pd.Series(values, dtype=object)).any():
        raise ValueError("Space group data must not contain missing values")
    # NumPy scalars to Python ints/floats/strings, which spg_to_crystal_sys accepts
    values = [val.item() if isinstance(val, np.generic) else val for val in values]

    by_system: dict[str, list[dict[str, Any]]] = {}
    for spg, count in Counter(values).most_common():
        crystal_sys = spg_to_crystal_sys(spg)
        by_system.setdefault(crystal_sys, []).append(
            {"id": f"{crystal_sys}/{spg}", "label": str(spg), "value": count}
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
