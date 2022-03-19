import pandas as pd
import pytest
from matminer.datasets import load_dataset
from matplotlib.axes import Axes
from plotly.graph_objs._figure import Figure

from pymatviz import (
    ROOT,
    count_elements,
    hist_elemental_prevalence,
    ptable_heatmap,
    ptable_heatmap_plotly,
    ptable_heatmap_ratio,
)


glasses = load_dataset("matbench_glass").composition
steels = load_dataset("matbench_steels").composition
df_ptable = pd.read_csv(f"{ROOT}/pymatviz/elements.csv").set_index("symbol")


glass_elem_counts = count_elements(glasses)
steel_elem_counts = count_elements(steels)


@pytest.mark.parametrize(
    "idx, expected", [(1, glass_elem_counts), (2, steel_elem_counts)]
)
def test_count_elements(idx, expected):
    # ground truth for element counts
    # df.squeeze("columns") turns single-col df into series
    el_cnt = pd.read_csv(f"{ROOT}/data/elem_counts_{idx}.csv", index_col=0)
    el_cnt = el_cnt.squeeze("columns")

    pd.testing.assert_series_equal(expected, el_cnt)


def test_count_elements_atomic_nums():
    el_cts = count_elements({str(idx): idx for idx in range(1, 119)})
    expected = pd.Series(range(1, 119), index=df_ptable.index, name="count")

    pd.testing.assert_series_equal(expected, el_cts)


@pytest.mark.parametrize("rng", [(-1, 10), (100, 200)])
def test_count_elements_bad_atomic_nums(rng):
    with pytest.raises(ValueError, match="assumed to represent atomic numbers"):
        count_elements({str(idx): 0 for idx in list(range(*rng))})


def test_hist_elemental_prevalence():
    ax = hist_elemental_prevalence(glasses)
    assert isinstance(ax, Axes)

    hist_elemental_prevalence(glasses, log=True)

    hist_elemental_prevalence(glasses, keep_top=10)

    hist_elemental_prevalence(glasses, keep_top=10, bar_values="count")


def test_ptable_heatmap():
    ax = ptable_heatmap(glasses)
    assert isinstance(ax, Axes)

    ptable_heatmap(glasses, log=True)

    # custom color map
    ptable_heatmap(glasses, log=True, cmap="summer")

    # heat_labels normalized to total count
    ptable_heatmap(glasses, heat_labels="fraction")
    ptable_heatmap(glasses, heat_labels="percent")

    # without heatmap values
    ptable_heatmap(glasses, heat_labels=None)
    ptable_heatmap(glasses, log=True, heat_labels=None)

    # element properties as heatmap values
    ptable_heatmap(df_ptable.atomic_mass)

    # element properties as heatmap values
    ptable_heatmap(df_ptable.atomic_mass, text_color=("red", "blue"))

    # custom max color bar value
    ptable_heatmap(glasses, cbar_max=1e2)
    ptable_heatmap(glasses, log=True, cbar_max=1e2)

    # element counts
    ptable_heatmap(glass_elem_counts)

    with pytest.raises(ValueError) as exc_info:
        ptable_heatmap(glasses, log=True, heat_labels="percent")

    assert exc_info.type is ValueError
    assert "Combining log color scale" in exc_info.value.args[0]


def test_ptable_heatmap_ratio():
    # composition strings
    ax = ptable_heatmap_ratio(glasses, steels)
    assert isinstance(ax, Axes)

    # element counts
    ptable_heatmap_ratio(glass_elem_counts, steel_elem_counts, normalize=True)

    # mixed element counts and composition
    ptable_heatmap_ratio(glasses, steel_elem_counts)
    ptable_heatmap_ratio(glass_elem_counts, steels)


def test_ptable_heatmap_plotly():
    fig = ptable_heatmap_plotly(glasses)
    assert isinstance(fig, Figure)
    assert len(fig.layout.annotations) == 18 * 10  # n_cols * n_rows
    assert sum(anno.text != "" for anno in fig.layout.annotations) == 118  # n_elements

    ptable_heatmap_plotly(
        glasses, hover_cols=["atomic_mass", "atomic_number", "density"]
    )
    ptable_heatmap_plotly(
        glasses,
        hover_data="density = " + df_ptable.density.astype(str) + " g/cm^3",
    )
    ptable_heatmap_plotly(df_ptable.density, precision=".1f")

    ptable_heatmap_plotly(glasses, heat_labels="percent")

    ptable_heatmap_plotly(glasses, colorscale=[(0, "red"), (1, "blue")])
