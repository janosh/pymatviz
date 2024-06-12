from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pytest

from pymatviz import elements_hist, plot_histogram, spacegroup_hist, true_pred_hist
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


@pytest.mark.parametrize("log_y", [True, False])
@pytest.mark.parametrize("backend", VALID_BACKENDS)
@pytest.mark.parametrize("bins", [20, 100])
def test_plot_histogram(log_y: bool, backend: Backend, bins: int) -> None:
    fig = plot_histogram(y_true, backend=backend, log_y=log_y, bins=bins)
    if backend == MPL_BACKEND:
        assert isinstance(fig, plt.Figure)
        y_min, y_max = fig.axes[0].get_ylim()
        y_min_exp, y_max_exp = {
            (True, 20): (0.891250938, 11.2201845),
            (True, 100): (0.93303299, 4.28709385),
            (False, 20): (0, 10.5),
            (False, 100): (0, 4.2),
        }[(log_y, bins)]
        assert y_min == pytest.approx(y_min_exp)
        assert y_max == pytest.approx(y_max_exp)
    else:
        assert isinstance(fig, go.Figure)
        dev_fig = fig.full_figure_for_development()
        y_min, y_max = dev_fig.layout.yaxis.range
        y_min_exp, y_max_exp = {
            (True, 20): (-0.05555555, 01.05555555),
            (True, 100): (-0.03344777, 0.63550776),
            (False, 20): (0, 10.5263157),
            (False, 100): (0, 4.21052631),
        }[(log_y, bins)]
        assert y_min == pytest.approx(y_min_exp)
        assert y_max == pytest.approx(y_max_exp)
