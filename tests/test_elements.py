import pandas as pd
import pytest
from matplotlib.axes import Axes
from plotly.graph_objects import Figure

from ml_matrics import (
    ROOT,
    count_elements,
    hist_elemental_prevalence,
    ptable_heatmap,
    ptable_heatmap_plotly,
    ptable_heatmap_ratio,
)


formulas_1 = pd.read_csv(f"{ROOT}/data/mp-n_elements<2.csv").formula
formulas_2 = pd.read_csv(f"{ROOT}/data/ex-ensemble-roost.csv").composition
df_ptable = pd.read_csv(f"{ROOT}/ml_matrics/elements.csv").set_index("symbol")


elem_counts_1 = count_elements(formulas_1)
elem_counts_2 = count_elements(formulas_2)


def test_count_elements():
    # ground truth for element counts
    # df.squeeze("columns") turns single-col df into series
    el_cnt_1 = pd.read_csv(f"{ROOT}/data/elem_counts_1.csv", index_col=0)
    el_cnt_1 = el_cnt_1.squeeze("columns")

    el_cnt_2 = pd.read_csv(f"{ROOT}/data/elem_counts_2.csv", index_col=0)
    el_cnt_2 = el_cnt_2.squeeze("columns")

    pd.testing.assert_series_equal(elem_counts_1, el_cnt_1)
    pd.testing.assert_series_equal(elem_counts_2.round(4), el_cnt_2.round(4))


def test_hist_elemental_prevalence():
    ax = hist_elemental_prevalence(formulas_1)
    assert isinstance(ax, Axes)

    hist_elemental_prevalence(formulas_1, log=True)

    hist_elemental_prevalence(formulas_1, keep_top=10)

    hist_elemental_prevalence(formulas_1, keep_top=10, bar_values="count")


def test_ptable_heatmap():
    ax = ptable_heatmap(formulas_1)
    assert isinstance(ax, Axes)

    ptable_heatmap(formulas_1, log=True)

    # custom color map
    ptable_heatmap(formulas_1, log=True, cmap="summer")

    # heat_labels normalized to total count
    ptable_heatmap(formulas_1, heat_labels="fraction")
    ptable_heatmap(formulas_1, heat_labels="percent")

    # without heatmap values
    ptable_heatmap(formulas_1, heat_labels=None)
    ptable_heatmap(formulas_1, log=True, heat_labels=None)

    # element properties as heatmap values
    ptable_heatmap(df_ptable.atomic_mass)

    # custom max color bar value
    ptable_heatmap(formulas_1, cbar_max=1e2)
    ptable_heatmap(formulas_1, log=True, cbar_max=1e2)

    # element counts
    ptable_heatmap(elem_counts_1)

    with pytest.raises(ValueError) as exc_info:
        ptable_heatmap(formulas_1, log=True, heat_labels="percent")

    assert exc_info.type is ValueError
    assert "Combining log color scale" in exc_info.value.args[0]


def test_ptable_heatmap_ratio():
    # composition strings
    ax = ptable_heatmap_ratio(formulas_1, formulas_2)
    assert isinstance(ax, Axes)

    # element counts
    ptable_heatmap_ratio(elem_counts_1, elem_counts_2, normalize=True)

    # mixed element counts and composition
    ptable_heatmap_ratio(formulas_1, elem_counts_2)
    ptable_heatmap_ratio(elem_counts_1, formulas_2)


def test_ptable_heatmap_plotly():
    fig = ptable_heatmap_plotly(formulas_1)
    assert isinstance(fig, Figure)
    assert len(fig.layout.annotations) == 18 * 10  # n_cols * n_rows
    assert sum(anno.text != "" for anno in fig.layout.annotations) == 118  # n_elements

    ptable_heatmap_plotly(
        formulas_1, hover_cols=["atomic_mass", "atomic_number", "density"]
    )
    ptable_heatmap_plotly(
        formulas_1,
        hover_data="density = " + df_ptable.density.astype(str) + " g/cm^3",
    )
    ptable_heatmap_plotly(df_ptable.density, precision=".1f")

    ptable_heatmap_plotly(formulas_1, heat_labels="percent")

    ptable_heatmap_plotly(formulas_1, colorscale=[(0, "red"), (1, "blue")])
