"""Bar plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymatviz.widgets._normalize import with_defaults
from pymatviz.widgets.bar_plot import bar_axis_defaults
from pymatviz.widgets.spacegroup_bar import SpacegroupBarPlotWidget, spacegroup_counts


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    import pandas as pd

    from pymatviz.typing import SpacegroupLike


def spacegroup_bar_plot(
    data: Sequence[SpacegroupLike] | pd.Series,
    *,
    show_counts: bool = True,
    log_y: bool = False,
    **kwargs: Any,
) -> SpacegroupBarPlotWidget:
    """Bar chart of space group frequencies shaded by crystal system.

    Inputs are validated and counted per space group in Python, so structures and
    invalid symbols fail here instead of being dropped in the frontend and the
    widget syncs at most 230 counts. Bars span the full 1-230 range in numerical
    order.

    Args:
        data (Sequence[SpacegroupLike] | pd.Series | Mapping): Space
            group symbols, integral numbers (1-230), structures (kemio, pymatgen,
            ASE, phonopy),
            or a ``{spacegroup: count}`` mapping.
        show_counts (bool, optional): Whether to annotate each crystal system region
            with its total count. Defaults to True.
        log_y (bool): Whether to log scale the count axis. Defaults to False.
        **kwargs: SpacegroupBarPlotWidget traits, e.g. ``orientation``,
            ``show_legend``, ``x_axis``, ``y_axis`` (merged over the log_y default).

    Returns:
        SpacegroupBarPlotWidget: The space group bar plot.

    Raises:
        ValueError: If data is empty or holds missing/invalid space groups.
    """
    if not (counts := spacegroup_counts(data)):
        raise ValueError("Cannot plot empty spacegroup data")
    log_axis = {"scale_type": "log"} if log_y else {}
    axes = bar_axis_defaults(kwargs, category={}, value=log_axis)
    return SpacegroupBarPlotWidget(
        counts, show_counts=show_counts, **with_defaults(axes, kwargs)
    )
