from __future__ import annotations

from typing import Literal

import pandas as pd
import plotly.graph_objects as go
import pytest

import pymatviz as pmv
from tests.conftest import np_rng


@pytest.mark.parametrize("labels_with_counts", [True, False, "percent"])
def test_sankey_from_2_df_cols(labels_with_counts: bool | Literal["percent"]) -> None:
    col_names = ["col_a", "col_b"]

    df_rand = pd.DataFrame(np_rng.integers(0, 10, size=(100, 2)), columns=col_names)
    fig = pmv.sankey_from_2_df_cols(
        df_rand, col_names, labels_with_counts=labels_with_counts
    )
    assert isinstance(fig, go.Figure)
