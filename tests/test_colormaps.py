"""Unit tests for colormaps."""


from __future__ import annotations

import pytest
import matplotlib
from matplotlib.colors import Colormap

from pymatviz.colormaps import combine_two_colormaps


@pytest.mark.parametrize(
    "cmap1, cmap2, node, N, reverse",
    [
        (matplotlib.colormaps["Reds"], matplotlib.colormaps["Blues_r"], 0.5, 256, False),
        ("Reds", "Blues_r", 0.5, 256, False),
    ])
def test_combine_two_colormaps(cmap1, cmap2, node, N, reverse):
    result_cmap = combine_two_colormaps([cmap1, cmap2], node=node, N=N, reverse=reverse)

    assert isinstance(result_cmap, Colormap)


@pytest.mark.parametrize(
    "cmap1, cmap2, node, N, reverse",
    [
        (-1, "Blues_r", 0.5, 256, False),
    ])
def test_combine_two_colormaps_invalid(cmap1, cmap2, node, N, reverse):
    with pytest.raises(TypeError) as excinfo:
        combine_two_colormaps([cmap1, cmap2], node=node, N=N, reverse=reverse)

    assert str(excinfo.value) == "Invalid datatype in the list. It must contain either all Colormaps or strings."
