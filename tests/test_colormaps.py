"""Unit tests for colormaps."""


from __future__ import annotations

import matplotlib
import pytest
from matplotlib.colors import Colormap

from pymatviz.colormaps import combine_two_colormaps, truncate


@pytest.mark.parametrize(
    "cmap1, cmap2, node, N, reverse",
    [
        (
            matplotlib.colormaps["Reds"],
            matplotlib.colormaps["Blues_r"],
            0.5,
            256,
            False,
        ),
        ("Reds", "Blues_r", 0.25, 128, True),
    ],
)
def test_combine_two_colormaps(cmap1, cmap2, node, N, reverse):
    result_cmap = combine_two_colormaps([cmap1, cmap2], node=node, N=N, reverse=reverse)

    assert isinstance(result_cmap, Colormap)


@pytest.mark.parametrize(
    "cmap1, cmap2, node, N, reverse",
    [
        (-1, "Blues_r", 0.5, 256, False),
    ],
)
def test_combine_two_colormaps_invalid(cmap1, cmap2, node, N, reverse):
    with pytest.raises(TypeError) as excinfo:
        combine_two_colormaps([cmap1, cmap2], node=node, N=N, reverse=reverse)

    assert (
        str(excinfo.value)
        == "Invalid datatype in the list. It must contain either all Colormaps or strings."
    )


@pytest.mark.parametrize(
    "cmap, start, end, N",
    [
        ("viridis", 0.2, 0.8, 128),
        (matplotlib.colormaps["Reds"], 0, 0.8, 256),
    ],
)
def test_truncate(cmap, start, end, N):
    truncated_cmap = truncate(cmap, start, end, N)

    assert isinstance(truncated_cmap, Colormap)
