from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pytest

from pymatviz import elements_hist, spacegroup_hist, true_pred_hist
from pymatviz.utils import VALID_BACKENDS, mpl_key
from tests.conftest import df_regr, y_pred, y_true


if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import ArrayLike
    from pymatgen.core import Structure

    from pymatviz.utils import Backend


y_std_mock = y_true - y_pred


@pytest.mark.parametrize("bins", [None, 1, 100])
@pytest.mark.parametrize("cmap", ["hot", "Blues"])
@pytest.mark.parametrize(
    "df, y_true, y_pred, y_std",
    [
        (None, y_true, y_pred, y_std_mock),
        (df_regr, *df_regr.columns[:2], df_regr.columns[0]),
    ],
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
@pytest.mark.parametrize("show_empty_bins", [True, False])
@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_spacegroup_hist(
    spg_symbols: list[str],
    structures: list[Structure],
    backend: Backend,
    xticks: Literal["all", "crys_sys_edges", 1, 50],
    show_counts: bool,
    show_empty_bins: bool,
) -> None:
    # spacegroup numbers
    fig = spacegroup_hist(
        range(1, 231),
        xticks=xticks,
        show_counts=show_counts,
        show_empty_bins=show_empty_bins,
        backend=backend,
    )
    assert isinstance(fig, plt.Axes if backend == mpl_key else go.Figure)

    # spacegroup symbols
    fig = spacegroup_hist(
        spg_symbols,
        xticks=xticks,
        show_counts=show_counts,
        show_empty_bins=show_empty_bins,
        backend=backend,
    )

    # pmg structures
    spacegroup_hist(
        structures,
        xticks=xticks,
        show_counts=show_counts,
        show_empty_bins=show_empty_bins,
        backend=backend,
    )


def test_hist_elemental_prevalence(glass_formulas: list[str]) -> None:
    ax = elements_hist(glass_formulas)
    assert isinstance(ax, plt.Axes)
    plt.clf()

    ax = elements_hist(glass_formulas, log=True)
    plt.clf()

    ax = elements_hist(glass_formulas, keep_top=10)
    plt.clf()

    elements_hist(glass_formulas, keep_top=10, bar_values="count")
    plt.clf()
