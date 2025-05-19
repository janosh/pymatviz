"""Explore the MatPES dataset.

- https://matpes.ai
- https://arxiv.org/abs/2503.04070
"""

# %%
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pymatgen.core import Composition, Structure
from tqdm import tqdm

import pymatviz as pmv
from pymatviz.enums import Files, Key


pmv.set_plotly_template("pymatviz_dark")
module_dir = os.path.dirname(__file__)
AWS_URL = "https://s3.us-east-1.amazonaws.com/materialsproject-contribs/MatPES_2025_1"
MATPES_DIR = f"{module_dir}/../../paper/data/matpes"
N_R2SCAN, N_PBE = 387_897, 434_712

if not os.path.isdir(MATPES_DIR):
    raise FileNotFoundError(f"{MATPES_DIR=}")


class DataFiles(Files, base_dir=MATPES_DIR, auto_download=True):
    """MatPES file names and URLs."""

    pbe = "MatPES-PBE-2025.1.json.gz", f"{AWS_URL}/MatPES-PBE-2025.1.json.gz"
    r2scan = "MatPES-R2SCAN-2025.1.json.gz", f"{AWS_URL}/MatPES-R2SCAN-2025.1.json.gz"


# %%
df_r2scan = pd.read_json(DataFiles.r2scan.file_path)
df_pbe = pd.read_json(DataFiles.pbe.file_path)

for df_data, expected_rows in ((df_r2scan, N_R2SCAN), (df_pbe, N_PBE)):
    if len(df_data) != expected_rows:
        raise ValueError(f"{expected_rows=}, {len(df_data)=}")
    df_data.index.name = Key.mat_id
    df_data.rename(columns={"magmom": Key.magmoms}, inplace=True)  # noqa: PD002
    df_data[Key.structure] = df_data[Key.structure].map(Structure.from_dict)
    df_data[Key.formula] = [struct.formula for struct in df_data[Key.structure]]
    df_data[Key.n_sites] = df_data[Key.structure].map(len)


# %% plot histogram of energies
fig = go.Figure()

fig.add_histogram(x=df_r2scan[Key.energy], name="r2scan", opacity=0.8)
fig.add_histogram(x=df_pbe[Key.energy], name="pbe", opacity=0.8)

fig.layout.xaxis.title = Key.energy.label
fig.layout.margin = dict(l=5, r=5, t=5, b=5)
fig.layout.legend.update(x=0, y=1)
fig.show()
# pmv.save_fig(fig, "energy-hist.pdf")

# @janosh 2024-05-15: initially surprised by the difference in r2SCAN and PBE energy
# distributions. how could energy differences between two similar chemistries always be
# similar across r2SCAN and PBE if the distribution for r2SCAN is much wider?
# Update: Seems fine actually. Parity plot reveals this nice collection of bands which
# looks like within each chemical system, you indeed get consistent energy differences.
# just across different systems, the zero-level energies differ wildly
fig = go.Figure()

fig.add_scatter(
    x=df_r2scan[Key.energy].head(1000),
    y=df_pbe.loc[df_r2scan.index][Key.energy].head(1000),
    mode="markers",
)
fig.layout.xaxis.title = f"r2SCAN {Key.energy.label}"
fig.layout.yaxis.title = f"PBE {Key.energy.label}"
pmv.powerups.add_identity_line(fig, retain_xy_limits=True)
fig.show()


# %% plot histograms of absolute sum of forces on each atom
total_force_col = "Σ|force<sub>i</sub>| (eV/Å)"
df_pbe[total_force_col] = df_pbe[Key.forces].map(lambda arr: np.abs(arr).sum(axis=1))
df_r2scan[total_force_col] = df_r2scan[Key.forces].map(
    lambda arr: np.abs(arr).sum(axis=1)
)

fig = go.Figure()
fig.add_histogram(x=df_r2scan[total_force_col].explode(), name="r2scan", opacity=0.8)
fig.add_histogram(x=df_pbe[total_force_col].explode(), name="pbe", opacity=0.8)

fig.update_layout(xaxis_title=total_force_col, margin=dict(l=5, r=5, t=5, b=5))
fig.layout.legend.update(x=0, y=1)
fig.update_yaxes(type="log")
fig.show()
# pmv.save_fig(fig, "forces-hist.pdf")


# %% plot element counts
r2scan_elem_counts = locals().get("r2scan_elem_counts")
if r2scan_elem_counts is None:
    r2scan_elem_counts = pmv.count_elements(df_r2scan[Key.formula])

fig = pmv.ptable_heatmap_plotly(r2scan_elem_counts)
fig.show()

# pmv.save_fig(fig, "r2scan-element-counts-ptable.pdf")


# %%
pbe_elem_counts = locals().get("pbe_elem_counts")
if pbe_elem_counts is None:
    pbe_elem_counts = pmv.count_elements(df_pbe[Key.formula])
fig = pmv.ptable_heatmap_plotly(pbe_elem_counts)
fig.show()


# %% calculate per element energies
frac_comp_col = "fractional composition"
for df_data in (df_r2scan, df_pbe):
    df_data[frac_comp_col] = [
        Composition(comp).fractional_composition for comp in tqdm(df_data[Key.formula])
    ]

df_r2scan_frac_comp = pd.DataFrame(
    comp.as_dict() for comp in df_r2scan[frac_comp_col]
).set_index(df_r2scan.index)
df_pbe_frac_comp = pd.DataFrame(
    comp.as_dict() for comp in df_pbe[frac_comp_col]
).set_index(df_pbe.index)

if any(df_r2scan_frac_comp.sum(axis="columns").round(6) != 1):
    raise ValueError("composition fractions don't sum to 1")

df_per_elem = pd.DataFrame()
r2scan_col = "r2SCAN energy"
col_name = Key.cohesive_energy_per_atom
df_per_elem[r2scan_col] = (
    df_r2scan_frac_comp * df_r2scan[col_name].to_numpy()[:, None]
).mean()
pbe_col = "PBE energy"
df_per_elem[pbe_col] = (df_pbe_frac_comp * df_pbe[col_name].to_numpy()[:, None]).mean()


# %% cohesive energies should (and do) look nearly identical between r2SCAN and PBE
per_elem_cohesive_energy = {
    key: list(dct.values()) for key, dct in df_per_elem.to_dict(orient="index").items()
}

fig = pmv.ptable_heatmap_splits_plotly(
    per_elem_cohesive_energy,
    colorbar=dict(title=col_name.label, orientation="h"),
)
fig.show()


# %% which elements have a higher share of missing r2SCAN data
fig = pmv.ptable_heatmap_plotly(
    (pbe_elem_counts - r2scan_elem_counts) / pbe_elem_counts,
    colorbar=dict(
        title="Fraction of missing PBE calcs missing r2SCAN", orientation="h"
    ),
    heat_mode="percent",
)
fig.show()

pmv.save_fig(fig, "ptable-has-pbe-but-no-r2scan.pdf")


# %% per-elem mean abs magmoms
df_per_elem_magmoms = pd.DataFrame(
    {site.specie.symbol: abs(site.properties["magmom"]) for site in struct}
    for struct in df_r2scan[Key.structure]
).mean()

fig = pmv.ptable_heatmap_plotly(df_per_elem_magmoms)
fig.layout.coloraxis.colorbar.title = r"Mean |magmom| ($\mu_B$)"
fig.show()
# pmv.save_fig(fig, "magmoms-ptable.pdf")


# %% spacegroup distribution
for label, df_data in (
    ("r2scan", df_r2scan),
    # ("pbe", df_pbe),
):
    df_data[Key.spg_num] = [
        struct.get_space_group_info()[1]
        for struct in tqdm(df_data[Key.structure], desc=f"{label} spacegroups")
    ]


# %% high-temperate MLMD frames are expected to have low symmetry (mostly triclinic)
fig = pmv.spacegroup_sunburst(df_r2scan[Key.spg_num], show_counts="percent")
fig.layout.title = dict(text=f"{N_R2SCAN:,} r2SCAN spacegroups", x=0.5, y=0.98)
fig.layout.margin = dict(l=0, r=0, b=0, t=30)
fig.show()
# pmv.save_fig(fig, "r2scan-spacegroup-sunburst.pdf")


# %% spacegroup histogram
fig = pmv.spacegroup_bar(
    df_r2scan[Key.spg_num], title="r2SCAN spacegroup histogram", log=True
)
fig.show()
# pmv.save_fig(fig, "r2scan-spacegroup-hist.pdf")


# %% most calcs missing r2SCAN results have 4 sites, almost all 2 or 3-site r2SCAN calcs
# completed
fig = go.Figure()

fig.add_histogram(x=df_r2scan[Key.n_sites], name="r2scan", opacity=0.8)
fig.add_histogram(x=df_pbe[Key.n_sites], name="pbe", opacity=0.8)

fig.layout.legend.update(x=0, y=1)
fig.layout.xaxis.title = Key.n_sites.label
fig.layout.yaxis.title = "count"
fig.show()


# %% plot absolute forces projected onto elements
df_r2scan[Key.forces] = df_r2scan[Key.forces].map(np.abs)
df_pbe[Key.forces] = df_pbe[Key.forces].map(np.abs)

df_r2scan_elem_forces = pd.DataFrame(
    {
        site.specie.symbol: np.linalg.norm(force)
        for site, force in zip(struct, forces, strict=True)
    }
    for struct, forces in zip(
        df_r2scan[Key.structure], df_r2scan[Key.forces], strict=True
    )
).mean()

df_pbe_elem_forces = pd.DataFrame(
    {
        site.specie.symbol: np.linalg.norm(force)
        for site, force in zip(struct, forces, strict=True)
    }
    for struct, forces in zip(df_pbe[Key.structure], df_pbe[Key.forces], strict=True)
).mean()


# %%
fig = pmv.ptable_heatmap_splits_plotly(
    {
        elem: [df_r2scan_elem_forces[elem], df_pbe_elem_forces[elem]]
        for elem in df_r2scan_elem_forces.index
    },
    colorbar=dict(title="Mean |force| (eV/Å)", orientation="h"),
)
fig.show()


# %%
df_e_atoms = pd.read_csv(f"{module_dir}/2024-10-30-MatPES-atomic-energies.csv")

fig = px.line(
    df_e_atoms.round(3),
    x=Key.element,
    y=[Key.pbe.label, Key.r2scan.label],
    markers=True,
)
fig.layout.yaxis.title = "Cohesive Energy (eV/atom)"
fig.layout.legend.update(x=0, y=0, title="")
fig.layout.xaxis.range = [-1, len(df_e_atoms)]
fig.show()
