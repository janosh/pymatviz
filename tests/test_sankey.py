from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest


if sys.version_info < (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

from pymatviz.sankey import sankey_from_2_df_cols


@pytest.mark.parametrize("labels_with_counts", [True, False, "percent"])
def test_sankey_from_2_df_cols(labels_with_counts: bool | Literal["percent"]) -> None:
    col_names = "col_a col_b".split()
    df = pd.DataFrame(np.random.randint(0, 10, size=(100, 2)), columns=col_names)
    fig = sankey_from_2_df_cols(df, col_names, labels_with_counts)
    assert isinstance(fig, go.Figure)
