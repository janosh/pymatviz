import pandas as pd

from ml_matrics import (
    count_elements,
    hist_elemental_prevalence,
    ptable_elemental_prevalence,
    ptable_elemental_ratio,
)


compositions_a = pd.read_csv("data/mp-n_elements<2.csv").formula
compositions_b = pd.read_csv("data/ex-ensemble-roost.csv").composition

counts_a = count_elements(compositions_a)
counts_b = count_elements(compositions_b)


def test_hist_elemental_prevalence():
    hist_elemental_prevalence(compositions_a)


def test_hist_elemental_prevalence_log():
    hist_elemental_prevalence(compositions_a, log=True)


def test_hist_elemental_prevalence_with_keep_top():
    hist_elemental_prevalence(compositions_a, keep_top=10)


def test_hist_elemental_prevalence_with_bar_values_count():
    hist_elemental_prevalence(compositions_a, keep_top=10, bar_values="count")


def test_ptable_elemental_prevalence():
    ptable_elemental_prevalence(compositions_a)


def test_ptable_elemental_prevalence_log():
    ptable_elemental_prevalence(compositions_a)


def test_ptable_elemental_prevalence_cbar_max():
    cbar_max = max(counts_a.max(), counts_b.max())
    ptable_elemental_prevalence(compositions_a, cbar_max=cbar_max)
    ptable_elemental_prevalence(compositions_b, cbar_max=cbar_max)


def test_ptable_elemental_prevalence_with_elem_counts():
    elem_counts = count_elements(compositions_a)
    ptable_elemental_prevalence(elem_counts=elem_counts)


def test_ptable_elemental_ratio():
    ptable_elemental_ratio(compositions_a, compositions_b)


def test_ptable_elemental_ratio_with_elem_counts():
    elem_counts = count_elements(compositions_a)
    elem_counts_b = count_elements(compositions_b)
    ptable_elemental_ratio(elem_counts_a=elem_counts, elem_counts_b=elem_counts_b)


def test_ptable_elemental_ratio_with_mixed():
    elem_counts_b = count_elements(compositions_b)
    ptable_elemental_ratio(formulas_a=compositions_a, elem_counts_b=elem_counts_b)
    ptable_elemental_ratio(formulas_b=compositions_a, elem_counts_a=elem_counts_b)
