"""Stats for the boltztrap_mp dataset.

Input: Pymatgen Structure of the material.
Columns:
  - mpid: Materials Project identifier
  - formula: Chemical formula of the entry
  - m_n: n-type/conduction band effective mass. Units: m_e (electron mass),
    i.e. m_n is a unitless ratio
  - m_p: p-type/valence band effective mass.
  - pf_n: n-type thermoelectric power factor in uW/cm2.K
    where uW is microwatts and a constant relaxation time of 1e-14 assumed.
  - pf_p: p-type power factor in uW/cm2.K
  - s_n: n-type Seebeck coefficient in micro Volts per Kelvin
  - s_p: p-type Seebeck coefficient in micro Volts per Kelvin
Entries: 8924

Effective mass and thermoelectric properties of 8924 MP compounds calculated
by the BoltzTraP software package run on the GGA-PBE or GGA+U DFT results.
The properties are reported at 300 Kelvin and carrier concentration of 1e18/cm3.

Reference:
Ricci, F. et al. An ab initio electronic transport database for inorganic materials.
https://www.nature.com/articles/sdata201785
Dryad Digital Repository. https://doi.org/10.5061/dryad.gn001

https://hackingmaterials.lbl.gov/matminer/dataset_summary.html
"""

# %%
import matplotlib.pyplot as plt
from matminer.datasets import load_dataset

from pymatviz import count_elements, ptable_heatmap
from pymatviz.enums import Key
from pymatviz.io import save_fig


# %%
df_boltz = load_dataset("boltztrap_mp")
df_boltz.describe()


# %%
fig = ptable_heatmap(count_elements(df_boltz[Key.formula]), log=True)
fig.suptitle("Elements in BoltzTraP MP dataset")
save_fig(fig, "boltztrap_mp-ptable-heatmap.pdf")


# %%
fig = ptable_heatmap(
    count_elements(df_boltz.sort_values("pf_n").tail(100)[Key.formula])
)
fig.suptitle("Elements of top 100 n-type powerfactors in BoltzTraP MP dataset")
save_fig(fig, "boltztrap_mp-ptable-heatmap-top-100-nPF.pdf")


# %%
ax = df_boltz.hist(bins=50, log=True, layout=[2, 3], figsize=[18, 8])
plt.suptitle("BoltzTraP MP")
save_fig(ax, "boltztrap_mp-hists.pdf")


# %%
df_boltz.sort_values("pf_n", ascending=False).head(1000).hist(
    bins=50, log=True, layout=[2, 3], figsize=[18, 8]
)
plt.suptitle("BoltzTraP MP")
