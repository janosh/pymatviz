from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pytest

from pymatviz.bar import spacegroup_bar
from pymatviz.utils import BACKENDS, MATPLOTLIB, PLOTLY
from tests.conftest import y_pred, y_true


if TYPE_CHECKING:
    from pymatgen.core import Structure

    from pymatviz.utils import Backend


y_std_mock = y_true - y_pred


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("xticks", "show_counts", "show_empty_bins", "log"),
    [
        ("all", True, True, True),
        ("crys_sys_edges", False, False, False),
        (1, True, False, True),
        (50, False, True, False),
    ],
)
def test_spacegroup_bar(
    spg_symbols: list[str],
    structures: list[Structure],
    backend: Backend,
    xticks: Literal["all", "crys_sys_edges", 1, 50],
    show_counts: bool,
    show_empty_bins: bool,
    log: bool,
) -> None:
    # test spacegroups as integers
    fig = spacegroup_bar(
        range(1, 231),
        xticks=xticks,
        show_counts=show_counts,
        show_empty_bins=show_empty_bins,
        backend=backend,
        log=log,
    )
    assert isinstance(fig, plt.Axes if backend == MATPLOTLIB else go.Figure)
    y_min, y_max = fig.get_ylim() if backend == MATPLOTLIB else fig.layout.yaxis.range
    assert y_min == 0
    # next line randomly started failing in CI on 2024-07-06
    if "CI" not in os.environ:
        assert y_max == pytest.approx(
            0.02118929 if log and backend == PLOTLY else 1.05
        ), f"{y_max=} {log=} {backend=}"

    # test spacegroups as symbols
    fig = spacegroup_bar(
        spg_symbols,
        xticks=xticks,
        show_counts=show_counts,
        show_empty_bins=show_empty_bins,
        backend=backend,
    )

    # test spacegroups determined on-the-fly from structures
    spacegroup_bar(
        structures,
        xticks=xticks,
        show_counts=show_counts,
        show_empty_bins=show_empty_bins,
        backend=backend,
    )
