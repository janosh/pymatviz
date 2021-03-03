import pandas as pd

from mlmatrics import (
    count_elements,
    hist_elemental_prevalence,
    ptable_elemental_prevalence,
    ptable_elemental_ratio,
)

compositions = pd.read_csv("data/mp-n_elements<2.csv").formula
compositions_b = pd.read_csv("data/ex-ensemble-roost.csv").composition


def test_ptable_elemental_prevalence():
    ptable_elemental_prevalence(compositions)


def test_ptable_elemental_prevalence_log():
    ptable_elemental_prevalence(compositions)


def test_ptable_elemental_prevalence_with_elem_counts():
    elem_counts = count_elements(compositions)
    ptable_elemental_prevalence(elem_counts=elem_counts)


def test_hist_elemental_prevalence():
    hist_elemental_prevalence(compositions)


def test_hist_elemental_prevalence_log():
    hist_elemental_prevalence(compositions, log=True)


def test_hist_elemental_prevalence_with_keep_top():
    hist_elemental_prevalence(compositions, keep_top=10)


def test_hist_elemental_prevalence_with_bar_values_count():
    hist_elemental_prevalence(compositions, keep_top=10, bar_values="count")


def test_ptable_elemental_ratio():
    ptable_elemental_ratio(compositions, compositions_b)


def test_ptable_elemental_ratio_log():
    ptable_elemental_ratio(compositions, compositions_b, log=True)
