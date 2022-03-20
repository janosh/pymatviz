from __future__ import annotations

from typing import Literal

import pandas as pd
import pytest

from pymatviz import residual_hist, spacegroup_hist, true_pred_hist

from ._helpers import y_pred, y_true


@pytest.fixture
def df():
    spg_symbols = [
        "C2/m",
        "C2/m",
        "Fm-3m",
        "C2/m",
        "Cmc2_1",
        "P4/nmm",
        "P-43m",
        "P-43m",
        "P6_3mc",
        "Cmcm",
        "P2_1/m",
        "I2_13",
        "P-6m2",
    ]
    df = pd.DataFrame(spg_symbols, columns=["spg_symbol"])
    return df


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
    df: pd.DataFrame,
    xticks: Literal["all", "crys_sys_edges", 1, 50],
    show_counts: bool,
    include_missing: bool,
) -> None:
    spacegroup_hist(
        df.spg_symbol,
        xticks=xticks,
        show_counts=show_counts,
        include_missing=include_missing,
    )
