"""Bar plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymatviz.process_data import normalize_spacegroups
from pymatviz.widgets.bar_plot import bar_axis_defaults
from pymatviz.widgets.spacegroup_bar import SpacegroupBarPlotWidget


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    import pandas as pd
    from pymatgen.core import Structure


def spacegroup_bar(
    data: Sequence[int | float | str | Structure] | pd.Series,
    *,
    show_counts: bool = True,
    log: bool = False,
    **kwargs: Any,
) -> SpacegroupBarPlotWidget:
    """Bar chart of space group frequencies shaded by crystal system.

    Inputs are validated and converted to space group numbers in Python, so
    structures and invalid symbols fail here instead of being dropped in the
    frontend. Bars span the full 1-230 range in numerical order.

    Args:
        data (Sequence[int | float | str | Structure] | pd.Series): Space group symbols,
            integral numbers (1-230), or pymatgen structures/ASE Atoms.
        show_counts (bool, optional): Whether to annotate each crystal system region
            with its total count. Defaults to True.
        log (bool, optional): Whether to log scale the count axis. Defaults to False.
        **kwargs: SpacegroupBarPlotWidget traits, e.g. ``orientation``,
            ``show_legend``, ``x_axis``, ``y_axis`` (merged over the log default).

    Returns:
        SpacegroupBarPlotWidget: The space group bar plot.

    Raises:
        ValueError: If data is empty or holds missing/invalid space groups.
    """
    spg_nums = normalize_spacegroups(data).tolist()

    log_axis = {"scale_type": "log"} if log else {}
    kwargs |= bar_axis_defaults(kwargs, category={}, value=log_axis)
    return SpacegroupBarPlotWidget(spg_nums, show_counts=show_counts, **kwargs)
