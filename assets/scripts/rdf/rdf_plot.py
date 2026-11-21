"""RDF examples: element-pair and full RDFs, computed in the browser or Python."""

# %%
import numpy as np
from matminer.datasets import load_dataset
from pymatgen.core import Lattice, Structure

import pymatviz as pmv


# %%
df_phonons = load_dataset("matbench_phonons")
df_phonons["n_sites"] = df_phonons["structure"].map(len)

# element-pair RDFs of the 2 largest structures, computed in the browser
for struct in df_phonons.nlargest(2, "n_sites")["structure"]:
    widget = pmv.rdf_plot(struct, bins=100, cutoff=10)
    widget.show()
    formula = struct.formula.replace(" ", "")
    if formula == "Na8Nb8O24":  # in readme
        widget.to_img(
            f"{pmv.utils.ROOT}/assets/svg/element-pair-rdfs-{formula}.svg",
            width=700,
            transparent=True,
        )


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
n_atoms = n_formula_units * 6
coords = np.random.default_rng(seed=0).uniform(size=(n_atoms, 3))
amorphous_lipo4 = Structure(Lattice.cubic(20), species * n_formula_units, coords)

# selected element pairs (computed in Python), each structure's curves in a legend group
widget = pmv.rdf_plot(
    {"Crystal": crystal_lipo4, "Amorphous": amorphous_lipo4},
    element_pairs=[("Li", "O"), ("O", "P"), ("O", "O")],
)
widget.show()
svg_path = f"{pmv.utils.ROOT}/assets/svg/element-pair-rdfs-crystal-vs-amorphous.svg"
widget.to_img(svg_path, width=700, transparent=True)


# %% one all-atom RDF per structure
pmv.rdf_plot(
    {"Crystal": crystal_lipo4, "Amorphous": amorphous_lipo4}, mode="full"
).show()
