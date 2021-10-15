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


def test_hist_elemental_prevalence_log():
    hist_elemental_prevalence(compositions_1, log=True)


def test_hist_elemental_prevalence_with_keep_top():
    hist_elemental_prevalence(compositions_1, keep_top=10)


def test_hist_elemental_prevalence_with_2ar_values_count():
    hist_elemental_prevalence(compositions_1, keep_top=10, bar_values="count")


def test_ptable_heatmap():
    ptable_heatmap(compositions_1)
    ptable_heatmap(df_ptable.atomic_mass)


def test_ptable_heatmap_log():
    ptable_heatmap(compositions_1)


def test_ptable_heatmap_cbar_max():
    cbar_max = max(elem_counts_1.max(), elem_counts_2.max())
    ptable_heatmap(compositions_1, cbar_max=cbar_max)
    ptable_heatmap(compositions_2, cbar_max=cbar_max)


def test_ptable_heatmap_with_elem_counts():
    ptable_heatmap(elem_counts_1)


def test_ptable_heatmap_ratio():
    ptable_heatmap_ratio(compositions_1, compositions_2)


def test_ptable_heatmap_ratio_with_elem_counts():
    ptable_heatmap_ratio(elem_counts_1, elem_counts_2)


def test_ptable_heatmap_ratio_with_mixed():
    ptable_heatmap_ratio(compositions_1, elem_counts_2)
    ptable_heatmap_ratio(compositions_1, elem_counts_2)
