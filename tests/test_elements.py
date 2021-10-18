import pandas as pd

from ml_matrics import (
    count_elements,
    hist_elemental_prevalence,
    ptable_heatmap,
    ptable_heatmap_ratio,
)


compositions_1 = pd.read_csv("data/mp-n_elements<2.csv").formula
compositions_2 = pd.read_csv("data/ex-ensemble-roost.csv").composition
df_ptable = pd.read_csv("ml_matrics/elements.csv")

elem_counts_1 = count_elements(compositions_1)
elem_counts_2 = count_elements(compositions_2)


def test_hist_elemental_prevalence():
    hist_elemental_prevalence(compositions_1)
    hist_elemental_prevalence(compositions_1, log=True)
    hist_elemental_prevalence(compositions_1, keep_top=10)
    hist_elemental_prevalence(compositions_1, keep_top=10, bar_values="count")


def test_ptable_heatmap():
    ptable_heatmap(compositions_1)
    ptable_heatmap(compositions_1, log=True)

    # custom color map
    ptable_heatmap(compositions_1, log=True, cmap="summer")

    # heat_labels normalized to total count
    ptable_heatmap(compositions_1, heat_labels="fraction")
    ptable_heatmap(compositions_1, heat_labels="percent")

    # without heatmap values
    ptable_heatmap(compositions_1, heat_labels=None)
    ptable_heatmap(compositions_1, log=True, heat_labels=None)

    # element properties as heatmap values
    ptable_heatmap(df_ptable.atomic_mass)

    # custom max color bar value
    ptable_heatmap(compositions_1, cbar_max=1e2)
    ptable_heatmap(compositions_1, log=True, cbar_max=1e2)

    # element counts
    ptable_heatmap(elem_counts_1)


def test_ptable_heatmap_ratio():
    # composition strings
    ptable_heatmap_ratio(compositions_1, compositions_2)

    # element counts
    ptable_heatmap_ratio(elem_counts_1, elem_counts_2)

    # mixed element counts and composition
    ptable_heatmap_ratio(compositions_1, elem_counts_2)
    ptable_heatmap_ratio(elem_counts_1, compositions_2)
