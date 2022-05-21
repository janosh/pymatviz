from __future__ import annotations

import pandas as pd
import pytest
from matminer.datasets import load_dataset
from matplotlib.axes import Axes
from plotly.exceptions import PlotlyError
from plotly.graph_objs._figure import Figure
from pymatgen.core import Composition

from pymatviz import (
    ROOT,
    count_elements,
    hist_elemental_prevalence,
    ptable_heatmap,
    ptable_heatmap_plotly,
    ptable_heatmap_ratio,
)


@pytest.fixture
def glasses() -> pd.Series[Composition]:
    return load_dataset("matbench_glass").composition


@pytest.fixture
def glass_elem_counts(glasses: pd.Series[Composition]) -> pd.Series[int]:
    return count_elements(glasses)


@pytest.fixture
def steels() -> pd.Series[Composition]:
    # unusually fractional compositions, good for testing edge cases
    return load_dataset("matbench_steels").composition


@pytest.fixture
def steel_elem_counts(steels: pd.Series[Composition]) -> pd.Series[int]:
    return count_elements(steels)


@pytest.fixture
def df_ptable() -> pd.DataFrame:
    return pd.read_csv(f"{ROOT}/pymatviz/elements.csv").set_index("symbol")


@pytest.mark.parametrize(
    "mode, counts",
    [
        ("composition", {"Fe": 22, "O": 63, "P": 12}),
        ("fractional_composition", {"Fe": 2.5, "O": 5, "P": 0.5}),
        ("reduced_composition", {"Fe": 13, "O": 27, "P": 3}),
    ],
)
def test_count_elements(df_ptable, mode, counts):
    series = count_elements(["Fe2 O3"] * 5 + ["Fe4 P4 O16"] * 3, mode=mode)
    expected = pd.Series(counts, index=df_ptable.index, name="count").fillna(0)
    assert series.equals(expected)


def test_count_elements_by_atomic_nums(df_ptable):
    series_in = pd.Series(1, index=range(1, 119))
    el_cts = count_elements(series_in)
    expected = pd.Series(1, index=df_ptable.index, name="count")

    pd.testing.assert_series_equal(expected, el_cts)


@pytest.mark.parametrize("rng", [(-1, 10), (100, 200)])
def test_count_elements_bad_atomic_nums(rng):
    with pytest.raises(ValueError, match="assumed to represent atomic numbers"):
        count_elements({str(idx): 0 for idx in list(range(*rng))})


def test_hist_elemental_prevalence(glasses):
    ax = hist_elemental_prevalence(glasses)
    assert isinstance(ax, Axes)

    hist_elemental_prevalence(glasses, log=True)

    hist_elemental_prevalence(glasses, keep_top=10)

    hist_elemental_prevalence(glasses, keep_top=10, bar_values="count")


def test_ptable_heatmap(glasses, glass_elem_counts, df_ptable):
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


def test_ptable_heatmap_ratio(steels, glasses, steel_elem_counts, glass_elem_counts):
    # composition strings
    ax = ptable_heatmap_ratio(glasses, steels)
    assert isinstance(ax, Axes)

    # element counts
    ptable_heatmap_ratio(glass_elem_counts, steel_elem_counts, normalize=True)

    # mixed element counts and composition
    ptable_heatmap_ratio(glasses, steel_elem_counts)
    ptable_heatmap_ratio(glass_elem_counts, steels)


def test_ptable_heatmap_plotly(df_ptable, glasses):
    fig = ptable_heatmap_plotly(glasses)
    assert isinstance(fig, Figure)
    assert len(fig.layout.annotations) == 18 * 10  # n_cols * n_rows
    assert sum(anno.text != "" for anno in fig.layout.annotations) == 118  # n_elements

    ptable_heatmap_plotly(
        glasses, hover_props=["atomic_mass", "atomic_number", "density"]
    )
    ptable_heatmap_plotly(
        glasses,
        hover_data="density = " + df_ptable.density.astype(str) + " g/cm^3",
    )
    ptable_heatmap_plotly(df_ptable.density, precision=".1f")

    ptable_heatmap_plotly(glasses, heat_labels="percent")

    # test that bad colorscale raises ValueError
    with pytest.raises(ValueError, match="should be string, list of strings or list"):
        ptable_heatmap_plotly(glasses, colorscale=lambda: "foobar")  # type: ignore

    # test that unknown builtin colorscale raises ValueError
    with pytest.raises(PlotlyError, match="Colorscale foobar is not a built-in scale"):
        ptable_heatmap_plotly(glasses, colorscale="foobar")


@pytest.mark.parametrize(
    "clr_scl", ["YlGn", ["blue", "red"], [(0, "blue"), (1, "red")]]
)
def test_ptable_heatmap_plotly_colorscale(glasses, clr_scl):
    fig = ptable_heatmap_plotly(glasses, colorscale=clr_scl)
    assert isinstance(fig, Figure)
