from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pytest

from pymatviz import elements_hist, spacegroup_hist, true_pred_hist
from pymatviz.utils import MPL_BACKEND, PLOTLY_BACKEND, VALID_BACKENDS
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


@pytest.mark.parametrize("backend", VALID_BACKENDS)
@pytest.mark.parametrize(
    ("xticks", "show_counts", "show_empty_bins", "log"),
    [
        ("all", True, True, True),
        ("crys_sys_edges", False, False, False),
        (1, True, False, True),
        (50, False, True, False),
    ],
)
def test_spacegroup_hist(
    spg_symbols: list[str],
    structures: list[Structure],
    backend: Backend,
    xticks: Literal["all", "crys_sys_edges", 1, 50],
    show_counts: bool,
    show_empty_bins: bool,
    log: bool,
) -> None:
    # test spacegroups as integers
    fig = spacegroup_hist(
        range(1, 231),
        xticks=xticks,
        show_counts=show_counts,
        show_empty_bins=show_empty_bins,
        backend=backend,
        log=log,
    )
    assert isinstance(fig, plt.Axes if backend == MPL_BACKEND else go.Figure)
    y_min, y_max = fig.get_ylim() if backend == MPL_BACKEND else fig.layout.yaxis.range
    assert y_min == 0
    assert y_max == pytest.approx(
        0.02118929 if log and backend == PLOTLY_BACKEND else 1.05
    ), f"{y_max=} {log=} {backend=}"

    # test spacegroups as symbols
    fig = spacegroup_hist(
        spg_symbols,
        xticks=xticks,
        show_counts=show_counts,
        show_empty_bins=show_empty_bins,
        backend=backend,
    )

    # test spacegroups determined on-the-fly from structures
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
