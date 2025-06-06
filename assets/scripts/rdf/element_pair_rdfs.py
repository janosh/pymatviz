"""Element-pair RDF examples."""

# %%
import numpy as np
from matminer.datasets import load_dataset
from pymatgen.core import Lattice, Structure

import pymatviz as pmv
from pymatviz.enums import Key


pmv.set_plotly_template("pymatviz_white")


# %%
df_phonons = load_dataset("matbench_phonons")

# get the 2 largest structures
df_phonons[Key.n_sites] = df_phonons[Key.structure].map(len)

# plot element-pair RDFs for each structure
for struct in df_phonons.nlargest(2, Key.n_sites)[Key.structure]:
    fig = pmv.element_pair_rdfs(struct, n_bins=100, cutoff=10)
    formula = struct.formula
    fig.layout.title.update(text=f"Pairwise RDFs - {formula}", x=0.5, y=0.99)
    fig.layout.margin.t = 55

    fig.show()
    pmv.io.save_and_compress_svg(fig, f"element-pair-rdfs-{formula.replace(' ', '')}")


# %% compare RDFs for crystalline LiPO4 and randomly-placed "amorphous" LiPO4
lattice = Lattice.orthorhombic(4.7, 6.0, 4.7)
species = ["Li", "P", "O", "O", "O", "O"]
coords = [
    [0, 0, 0],  # Li
    [0.5, 0.5, 0.5],  # P
    [0.25, 0.5, 0.25],  # O
    [0.75, 0.5, 0.75],  # O
    [0.5, 0.25, 0.5],  # O
    [0.5, 0.75, 0.5],  # O
]
crystal_lipo4 = Structure(lattice, species, coords)

n_formula_units = 50
lattice = Lattice.cubic(20)
n_atoms = n_formula_units * 6
coords = np.random.default_rng(seed=0).uniform(size=(n_atoms, 3))
amorphous_lipo4 = Structure(lattice, species * n_formula_units, coords)

# Generate RDF multi-structure-comparison plot
fig_crys_vs_amorph = pmv.element_pair_rdfs(
    structures={"Crystal": crystal_lipo4, "Amorphous": amorphous_lipo4}
)

# Update layout
title = "Pairwise RDFs - LiPO4 Crystal vs Amorphous"
fig_crys_vs_amorph.layout.title = dict(text=title, x=0.5, y=0.99)
fig_crys_vs_amorph.layout.margin = dict(l=0, r=0, t=70, b=0)
fig_crys_vs_amorph.show()

# pmv.io.save_and_compress_svg(
#     fig_crys_vs_amorph, "element-pair-rdfs-crystal-vs-amorphous"
# )
