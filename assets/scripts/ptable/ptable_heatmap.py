"""Periodic table heatmap examples."""

# /// script
# dependencies = [
#     "matminer>=0.10.1",
#     "pyyaml>=6.0.3",
# ]
# ///

import os
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd
import yaml
from matminer.datasets import load_dataset

import pymatviz as pmv
from pymatviz.enums import ElemCountMode


rng = np.random.default_rng(seed=0)
module_dir = os.path.dirname(__file__)


# %%
df_steels = load_dataset("matbench_steels")
df_expt_gap = load_dataset("matbench_expt_gap")


# %% Atomic masses (the Series name titles the color bar)
widget = pmv.ptable_heatmap(pmv.utils.df_ptable["mass"])
widget.show()
widget.to_img(
    f"{pmv.utils.ROOT}/assets/svg/ptable-heatmap-atomic-mass.svg",
    width=1000,
    transparent=True,
)


# %% Element counts as percentages of all atoms in the dataset
widget = pmv.ptable_heatmap(df_expt_gap["composition"], heat_mode="percent")
widget.show()


# %% Element counts on a log color scale
widget = pmv.ptable_heatmap(
    df_expt_gap["composition"],
    log=True,
    color_bar={"title": "Elements in Matbench experimental band gaps"},
)
widget.show()
widget.to_img(
    f"{pmv.utils.ROOT}/assets/svg/ptable-heatmap-log.svg", width=1000, transparent=True
)


# %% Specific heat on a log scale, excluding light elements
widget = pmv.ptable_heatmap(
    pmv.utils.df_ptable["specific_heat_capacity"].dropna(),
    log=True,
    exclude_elements=["H", "He", "C", "O"],
    color_scale="interpolateYlOrRd",
    color_bar={"title": "Specific heat"},
)
widget.show()


# %% Fractional element counts of 30 compositions
widget = pmv.ptable_heatmap(
    df_expt_gap["composition"][:30],
    count_mode=ElemCountMode.fractional_composition,
    heat_mode="fraction",
    color_scale="interpolatePlasma",
)
widget.show()


# %% Valence electrons in VASP PBE 64 pseudo-potentials (MP input set)
with open(f"{module_dir}/vasp-pbe-64-n-val-elecs.yml") as file:
    elem_to_n_val_elecs = yaml.safe_load(file)
# convert Potcar symbol to element symbol
elem_to_n_val_elecs = {
    key.split("_")[0]: val for key, val in elem_to_n_val_elecs.items()
}
widget = pmv.ptable_heatmap(
    elem_to_n_val_elecs, color_bar={"title": "Valence electrons in VASP PBE 64"}
)
widget.show()


# %% Ratio of element frequencies in 2 datasets (normalized by dataset size)
gap_counts = pmv.count_elements(df_expt_gap["composition"]) / len(df_expt_gap)
steel_counts = pmv.count_elements(df_steels["composition"]) / len(df_steels)
widget = pmv.ptable_heatmap(
    gap_counts / steel_counts,
    log=True,  # ratios are better viewed on log scale
    color_scale="interpolateRdBu",
    missing={"color": "#ddd"},  # elements not present in either dataset
    color_bar={"title": "Matbench (band gap / steel) element counts"},
)
widget.show()


# %% Element prices
try:
    df_price = pd.read_html(
        "https://en.wikipedia.org/wiki/Prices_of_chemical_elements"
    )[0]
    df_price.columns = df_price.columns.droplevel(0)
    df_price = df_price.set_index("Symbol")
    prices = pd.to_numeric(df_price["USD/kg"], errors="coerce")
    prices = prices[prices.index.isin(pmv.utils.df_ptable.index)]  # drop footnote rows
except (HTTPError, URLError):
    prices = {elem: rng.lognormal(2, 1) * 1000 for elem in pmv.utils.df_ptable.index}

widget = pmv.ptable_heatmap(
    prices, log=True, color_bar={"title": "Cost of element per kg"}
)
widget.show()
