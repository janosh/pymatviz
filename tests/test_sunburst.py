from __future__ import annotations

from typing import Literal

import pandas as pd
import pytest
from plotly.graph_objs._figure import Figure

from pymatviz import spacegroup_sunburst


@pytest.mark.parametrize("show_vals", ["value", "percent", False])
def test_spacegroup_sunburst(
    spg_symbols: list[str], show_vals: Literal["value", "percent", False]
) -> None:
    fig = spacegroup_sunburst(range(1, 231), show_values=show_vals)
    assert isinstance(fig, Figure)
    assert set(fig.data[0].parents) == {
        "",
        "cubic",
        "trigonal",
        "triclinic",
        "orthorhombic",
        "tetragonal",
        "hexagonal",
        "monoclinic",
    }
    assert fig.data[0].branchvalues == "total"

    df = pd.DataFrame(spg_symbols, columns=["spg_symbol"])

    # frame with col name
    spacegroup_sunburst(df, spg_col="spg_symbol", show_values=show_vals)
    # series
    spacegroup_sunburst(df.spg_symbol, show_values=show_vals)
