from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import pytest

from pymatviz import residual_hist, spacegroup_hist, true_pred_hist
from tests.conftest import df, y_pred, y_true


if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import ArrayLike
    from pymatgen.core import Structure


y_std_mock = y_true - y_pred


@pytest.mark.parametrize("bins", [None, 1, 100])
@pytest.mark.parametrize("xlabel", [None, "foo"])
def test_residual_hist(bins: int | None, xlabel: str | None) -> None:
    ax = residual_hist(y_true - y_pred, bins=bins, xlabel=xlabel)

    assert isinstance(ax, plt.Axes)
    assert (
        ax.get_xlabel() == xlabel or r"Residual ($y_\mathrm{test} - y_\mathrm{pred}$)"
    )
    assert len(ax.lines) == 1
    legend = ax.get_legend()
    assert len(ax.patches) == bins or 10
    assert legend._get_loc() == 2  # 2 meaning 'upper left'


@pytest.mark.parametrize("bins", [None, 1, 100])
@pytest.mark.parametrize("cmap", ["hot", "Blues"])
@pytest.mark.parametrize(
    "df, y_true, y_pred, y_std",
    [(None, y_true, y_pred, y_std_mock), (df, *df.columns[:2], df.columns[0])],
)
def test_true_pred_hist(
    df: pd.DataFrame | None,
    y_true: ArrayLike | str,
    y_pred: ArrayLike | str,
    y_std: ArrayLike | dict[str, ArrayLike] | str | list[str],
    bins: int | None,
    cmap: str,
) -> None:
    ax = true_pred_hist(y_true, y_pred, y_std, df, bins=bins, cmap=cmap)
    assert isinstance(ax, plt.Axes)


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
