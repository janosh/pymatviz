# ruff: noqa: RUF001
"""Stats for the Electronic Transport Properties dataset.

Larger/complete version of BoltzTrap MP (data/boltztrap_mp.json.gz).

https://contribs.materialsproject.org/projects/carrier_transport

Unprocessed data available from
https://contribs.materialsproject.org/projects/carrier_transport.json.gz
(see https://git.io/JOMwY).

Reference:
Ricci, F. et al. An ab initio electronic transport database for inorganic materials.
https://www.nature.com/articles/sdata201785
Dryad Digital Repository. https://doi.org/10.5061/dryad.gn001

Extensive column descriptions and metadata at
https://hackingmaterials.lbl.gov/matminer/dataset_summary.html#ricci-boltztrap-mp-tabular.
"""

# %%
from matminer.datasets import load_dataset
from tqdm import tqdm

import pymatviz as pmv
from pymatviz.enums import Key


# %%
df_carrier = load_dataset("ricci_boltztrap_mp_tabular")
df_carrier = df_carrier.dropna(subset=Key.structure)
data_name = "Ricci carrier transport dataset"

# Getting space group symbols and numbers (take about 2 min)
df_carrier[[Key.spg_symbol, Key.spg_num]] = [
    struct.get_space_group_info() for struct in tqdm(df_carrier[Key.structure])
]


# %%
fig = pmv.ptable_heatmap_plotly(df_carrier.pretty_formula.dropna(), log=True)
title = f"{data_name} elemental prevalence"
fig.layout.title.update(text=title)
fig.show()
# pmv.save_fig(fig, "carrier-transport-ptable-heatmap.pdf")


# %%
ax, *_ = df_carrier.hist(bins=50, log=True, figsize=[30, 16]).flat
ax.figure.suptitle(data_name, y=1.05)
# pmv.save_fig(ax, "carrier-transport-hists.pdf")


# %%
ax, *_ = df_carrier.filter(like="S.").hist(bins=50, log=True, figsize=[18, 8]).flat
ax.figure.suptitle(f"{data_name} histograms for n- and p-type Seebeck coefficients")
pmv.save_fig(ax, "carrier-transport-seebeck-n+p.pdf")


# %%
dependent_vars = [
    "S.p [µV/K]",
    "S.n [µV/K]",
    "Sᵉ.p.v [µV/K]",
    "Sᵉ.n.v [µV/K]",
    "σ.p [1/Ω/m/s]",
    "σ.n [1/Ω/m/s]",
    "σᵉ.p.v [1/Ω/m/s]",
    "σᵉ.n.v [1/Ω/m/s]",
    "PF.p [µW/cm/K²/s]",
    "PF.n [µW/cm/K²/s]",
    "PFᵉ.p.v [µW/cm/K²/s]",
    "PFᵉ.n.v [µW/cm/K²/s]",
    "κₑ.p [W/K/m/s]",
    "κₑ.n [W/K/m/s]",
    "κₑᵉ.p.v [W/K/m/s]",
    "κₑᵉ.n.v [W/K/m/s]",
]

ax, *_ = df_carrier[dependent_vars].hist(bins=50, log=True, figsize=[30, 16]).flat
ax.figure.suptitle(f"{data_name} dependent variables", y=1.05)
# pmv.save_fig(ax.flat[0], "carrier-transport-hists-dependent-vars.pdf")


# %%
fig = pmv.spacegroup_bar(df_carrier[Key.spg_num])
fig.layout.title.update(text="Spacegroup distribution in the {data_name}")
fig.show()
# pmv.save_fig(fig, "carrier-transport-spacegroup-hist.pdf")
