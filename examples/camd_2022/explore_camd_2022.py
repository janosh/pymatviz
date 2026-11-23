"""Rhys received this dataset from Joseph Montoya at Toyota Research Institute (TRI) via
email on 2022-01-12.

Download link: https://data.matr.io/7
GitHub repo: https://github.com/TRI-AMDD/CAMD

DFT calculations are unfortunately OQMD based, i.e. not Materials Project compatible.

Description:
TRI's second active learning crystal discovery dataset from Computational Autonomy for
Materials Discovery (CAMD). The dataset has ~100k crystal structures, 25k of which are
within 20 meV of the hull and ~1k of which are on the hull. They organized all of the
campaigns by chemical system.
"""

# %%
import os

import pandas as pd
import requests
from kemio.interop import to_structure

import pymatviz as pmv


# %% Download data (if needed), cached as JSON since CSV would stringify the
# structure dicts
cache_path = "camd-2022-wo-features.json.gz"
if os.path.isfile(cache_path):
    print("Loading local data...")
    df_camd = pd.read_json(cache_path, orient="records")
else:
    print("Fetching data from AWS...")
    base_url = "https://s3.amazonaws.com/publications.matr.io/7/deployment/data/files"
    dataset_url = f"{base_url}/camd_data_to_release_wofeatures.json"
    df_camd = pd.DataFrame(requests.get(dataset_url, timeout=60).json())
    df_camd.to_json(cache_path, orient="records")


# %%
df_camd.hist(bins=50)


# %%
df_camd["structure"] = df_camd["structure"].map(to_structure)
df_camd["formula"] = [
    struct.composition.reduced_formula for struct in df_camd["structure"]
]
elem_counts = pmv.count_elements(df_camd["formula"])
title = "Elements in CAMD 2022 dataset"
pmv.ptable_heatmap(elem_counts, log=True, color_bar={"title": title}).show()


# %%
source_counts = df_camd.data_source.value_counts()
widget = pmv.bar_plot(
    series=[{"x": list(map(str, source_counts.index)), "y": source_counts.to_numpy()}],
    x_axis={"label": "Data source"},
    y_axis={"label": "Count"},
)
widget.show()


# %%
widget = pmv.spacegroup_sunburst(df_camd.space_group, label_text="label+percent")
widget.to_img("camd-2022-spacegroup-sunburst.pdf")
widget.show()
