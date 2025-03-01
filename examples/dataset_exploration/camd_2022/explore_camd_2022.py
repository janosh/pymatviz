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

import pymatviz as pmv


# %% Download data (if needed)
if os.path.isfile("camd-2022-wo-features.csv.bz2"):
    print("Loading local data...")
    df_camd = pd.read_csv("camd-2022-wo-features.csv.bz2")
else:
    print("Fetching data from AWS...")
    url = "https://s3.amazonaws.com/publications.matr.io/7/deployment/data/files"
    with_feat_str = "w" if (with_feat := False) else "wo"
    dataset_url = f"{url}/camd_data_to_release_{with_feat_str}features.json"
    data = requests.get(dataset_url, timeout=10).json()
    df_camd = pd.DataFrame(data)
    df_camd = pd.to_csv(f"camd-2022-{with_feat_str}-features.csv.bz2")


# %%
df_camd.hist(bins=50)


# %%
elem_counts = pmv.count_elements(df_camd.reduced_formula)
fig = pmv.ptable_heatmap_plotly(elem_counts, log=True)
fig.layout.title.update(text="<b>Elements in CAMD 2022 dataset</b>")
fig.show()
# pmv.save_fig(fig, "camd-2022-ptable-heatmap.pdf")


# %%
ax = df_camd.data_source.value_counts().plot.bar(fontsize=18, rot=0)
pmv.powerups.annotate_bars(ax, v_offset=3e3)


# %%
fig = pmv.spacegroup_sunburst(df_camd.space_group, show_counts="percent")
pmv.save_fig(fig, "camd-2022-spacegroup-sunburst.pdf")
fig.show()
