import pandas as pd

from mlmatrics import hist_elemental_prevalence, ptable_elemental_prevalence

df = pd.read_csv("data/mp-n_elements<2.csv")


def test_ptable_elemental_prevalence():
    ptable_elemental_prevalence(df.formula)


def test_hist_elemental_prevalence():
    hist_elemental_prevalence(df.formula)


def test_hist_elemental_prevalence_with_keep_top():
    hist_elemental_prevalence(df.formula, keep_top=10)
