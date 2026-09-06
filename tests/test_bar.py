from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from pymatviz.bar import spacegroup_bar


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal


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
        (pd.Series([225.0, 225.0], name="Counts"), {225: 2}),
    ],
    ids=[
        "all-groups",
        "integers",
        "floats",
        "symbols",
        "nullable-series",
        "counts-name",
    ],
)
@pytest.mark.parametrize(
    ("xticks", "show_counts", "show_empty_bins", "log"),
    [
        ("all", True, True, True),
        ("all", False, False, False),
        ("crys_sys_edges", False, False, False),
        ("crys_sys_edges", True, True, False),
        (1, True, False, True),
        (50, False, True, False),
    ],
)
def test_spacegroup_bar(
    data: Sequence[int | float | str] | pd.Series,
    counts: dict[int, int],
    xticks: Literal["all", "crys_sys_edges", 1, 50],
    show_counts: bool,
    show_empty_bins: bool,
    log: bool,
) -> None:
    """Normalize inputs and align counts, ticks, and crystal-system boundaries."""
    fig = spacegroup_bar(
        data,
        xticks=xticks,
        show_counts=show_counts,
        show_empty_bins=show_empty_bins,
        log=log,
    )
    assert isinstance(fig, go.Figure)
    y_min, y_max = fig.layout.yaxis.range
    assert y_min == 0
    expected_max = max(counts.values()) * 1.05
    assert y_max == (np.log10(expected_max) if log else expected_max)
    assert fig.layout.xaxis.title.text == "International Spacegroup Number"

    numbers = list(range(1, 231)) if show_empty_bins else sorted(counts)
    positions = numbers if show_empty_bins else list(range(len(numbers)))
    heights = [counts.get(number, 0) for number in numbers]
    assert [value for trace in fig.data for value in trace.x] == positions
    assert [value for trace in fig.data for value in trace.y] == heights
    assert fig.layout.xaxis.range == (positions[0] - 0.5, positions[-1] + 0.5)

    boundaries = [2, 15, 74, 142, 167, 194, 230]
    group_ends = sorted(
        {sum(number <= end for number in numbers) for end in boundaries} - {0}
    )
    if isinstance(xticks, int):
        indices = sorted(range(len(numbers)), key=lambda idx: -heights[idx])[:xticks]
    elif xticks == "crys_sys_edges":
        indices = [end - 1 for end in group_ends]
    else:
        indices = list(range(len(numbers)))
    assert list(fig.layout.xaxis.tickvals) == [positions[idx] for idx in indices]
    assert list(fig.layout.xaxis.ticktext) == [numbers[idx] for idx in indices]

    edges = [positions[0] - 0.5, *[positions[end - 1] + 0.5 for end in group_ends]]
    assert [(shape.x0, shape.x1) for shape in fig.layout.shapes] == list(
        pairwise(edges)
    )
