from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

from pymatviz.bar import spacegroup_bar_plot
from pymatviz.widgets.spacegroup_bar import SpacegroupBarPlotWidget


if TYPE_CHECKING:
    from collections.abc import Sequence


@pytest.mark.parametrize(
    ("data", "counts"),
    [
        (range(1, 231), dict.fromkeys(range(1, 231), 1)),
        ([225, 2, 1, 225, 62], {1: 1, 2: 1, 62: 1, 225: 2}),
        ([225.0, 2.0, 1.0, 225.0, 62.0], {1: 1, 2: 1, 62: 1, 225: 2}),
        (["Fm-3m", "P-1", "P1", "Fm-3m", "Pnma"], {1: 1, 2: 1, 62: 1, 225: 2}),
        (
            pd.Series([225, 2, 1, 225, 62], dtype="Float64", name="spacegroup"),
            {1: 1, 2: 1, 62: 1, 225: 2},
        ),
        (pd.Series([225, 2, 225], dtype=object), {2: 1, 225: 2}),
    ],
    ids=["all-groups", "integers", "floats", "symbols", "nullable-series", "object"],
)
@pytest.mark.parametrize("show_counts", [True, False])
def test_spacegroup_bar(
    data: Sequence[int | float | str] | pd.Series,
    counts: dict[int, int],
    show_counts: bool,
) -> None:
    """Inputs sync as per-space-group counts keyed by ascending number strings."""
    widget = spacegroup_bar_plot(data, show_counts=show_counts)
    assert isinstance(widget, SpacegroupBarPlotWidget)
    assert widget.data == {str(spg): count for spg, count in sorted(counts.items())}
    assert all(type(count) is int for count in widget.data.values())
    assert widget.show_counts is show_counts


@pytest.mark.parametrize(
    ("kwargs", "log_axis", "cat_axis"),
    [
        ({}, "y_axis", "x_axis"),
        ({"orientation": "horizontal"}, "x_axis", "y_axis"),
        ({"y_axis": {"label": "N"}}, "y_axis", "x_axis"),
    ],
)
@pytest.mark.parametrize("log_y", [True, False])
def test_spacegroup_bar_log_y(
    kwargs: dict[str, Any], log_axis: str, cat_axis: str, log_y: bool
) -> None:
    """log_y=True log-scales only the count axis, keeping user axis options."""
    widget = spacegroup_bar_plot([1, 225, 225], log_y=log_y, **kwargs)
    expected = ({"scale_type": "log"} if log_y else {}) | kwargs.get(log_axis, {})
    assert getattr(widget, log_axis) == expected
    assert getattr(widget, cat_axis) == {}


def test_spacegroup_bar_structures(structures: Sequence[Any]) -> None:
    """Structures are converted to space group numbers in Python."""
    widget = spacegroup_bar_plot(structures)
    assert sum(widget.data.values()) == len(structures)
    assert all(1 <= int(spg) <= 230 for spg in widget.data)


@pytest.mark.parametrize(
    ("data", "match"),
    [([], "empty"), ([1, 231], "must be in"), (["not-a-spg"], "Invalid")],
)
def test_spacegroup_bar_invalid(data: list[Any], match: str) -> None:
    """Empty and invalid inputs raise instead of being dropped by the frontend."""
    with pytest.raises(ValueError, match=match):
        spacegroup_bar_plot(data)
