from __future__ import annotations

import sys

import plotly.graph_objects as go
import pytest
from pymatgen.core import Structure

from pymatviz import spacegroup_sunburst


if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


@pytest.mark.parametrize("show_counts", ["value", "percent", False])
def test_spacegroup_sunburst(
    spg_symbols: list[str],
    structures: list[Structure],
    show_counts: Literal["value", "percent", False],
) -> None:
    # spg numbers
    fig = spacegroup_sunburst(range(1, 231), show_counts=show_counts)

    assert isinstance(fig, go.Figure)
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

    # spg symbols
    spacegroup_sunburst(spg_symbols, show_counts=show_counts)

    # pmg structures
    spacegroup_sunburst(structures, show_counts=show_counts)
