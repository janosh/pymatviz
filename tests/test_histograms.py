from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import pytest
from pymatgen.core import Structure

from pymatviz import residual_hist, spacegroup_hist, true_pred_hist

from .conftest import y_pred, y_true


@pytest.mark.parametrize("bins", [None, 1, 100])
@pytest.mark.parametrize("xlabel", [None, "foo"])
def test_residual_hist(bins: int | None, xlabel: str | None) -> None:
    residual_hist(y_true, y_pred, bins=bins, xlabel=xlabel)


def test_true_pred_hist():
    true_pred_hist(y_true, y_pred, y_true - y_pred)


@pytest.mark.parametrize("xticks", ["all", "crys_sys_edges", 1, 50])
@pytest.mark.parametrize("show_counts", [True, False])
@pytest.mark.parametrize("include_missing", [True, False])
def test_spacegroup_hist_num(
    spg_symbols: list[str],
    structures: list[Structure],
    xticks: Literal["all", "crys_sys_edges", 1, 50],
    show_counts: bool,
    include_missing: bool,
) -> None:
    # spg numbers
    spacegroup_hist(
        range(1, 231),
        xticks=xticks,
        show_counts=show_counts,
        include_missing=include_missing,
    )

    # spg symbols
    ax = spacegroup_hist(
        spg_symbols,
        xticks=xticks,
        show_counts=show_counts,
        include_missing=include_missing,
    )
    assert isinstance(ax, plt.Axes)

    # pmg structures
    spacegroup_hist(
        structures,
        xticks=xticks,
        show_counts=show_counts,
        include_missing=include_missing,
    )
