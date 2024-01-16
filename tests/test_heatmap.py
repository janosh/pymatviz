"""Unit tests for heatmap plotter."""


from __future__ import annotations

from io import StringIO

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from pymatviz.heatmap import heatmap


def test_heatmap() -> None:
    # Test data string
    data_string = """x_1,x_2,x_3,x_4,x_5,x_6
y_1,0,8,7,1,-8,-11
y_2,8,0,-2,-7,-8,-17
y_3,7,-2,0,-2,-1,-7
y_4,7,-7,-2,0,2,-1
y_5,-8,-8,-1,2,0,0
y_6,-11,-17,-7,-1,0,0
"""

    # Convert data string to pd.DataFrame
    data_df = pd.read_csv(StringIO(data_string), index_col=0)

    ax = heatmap(data_df, matplotlib.colormaps["viridis"], True)
    assert isinstance(ax, plt.Axes)
