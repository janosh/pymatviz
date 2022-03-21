from __future__ import annotations

from typing import Literal

import pandas as pd
import pytest

from pymatviz import residual_hist, spacegroup_hist, true_pred_hist

from .conftest import y_pred, y_true


def test_residual_hist():
    residual_hist(y_true, y_pred)


def test_true_pred_hist():
    true_pred_hist(y_true, y_pred, y_true - y_pred)


@pytest.mark.parametrize("xticks", ["all", "crys_sys_edges", 1, 50])
@pytest.mark.parametrize("show_counts", [True, False])
@pytest.mark.parametrize("include_missing", [True, False])
def test_spacegroup_hist_num(
    xticks: Literal["all", "crys_sys_edges", 1, 50],
    show_counts: bool,
    include_missing: bool,
) -> None:
    spacegroup_hist(
        range(1, 231),
        xticks=xticks,
        show_counts=show_counts,
        include_missing=include_missing,
    )


@pytest.mark.parametrize("xticks", ["all", "crys_sys_edges", 1, 50])
@pytest.mark.parametrize("show_counts", [True, False])
@pytest.mark.parametrize("include_missing", [True, False])
def test_spacegroup_hist_symbol(
    spg_symbols: list[str],
    xticks: Literal["all", "crys_sys_edges", 1, 50],
    show_counts: bool,
    include_missing: bool,
) -> None:
    df = pd.DataFrame(spg_symbols, columns=["spg_symbol"])

    spacegroup_hist(
        df,
        spg_col="spg_symbol",
        xticks=xticks,
        show_counts=show_counts,
        include_missing=include_missing,
    )
