"""Full RDF examples."""

# %%
import numpy as np
from pymatgen.core import Lattice, Structure

import pymatviz as pmv


# %% crystalline LiPO4 and randomly-placed "amorphous" LiPO4
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


# %% compare full RDFs for crystalline LiPO4 and randomly-placed "amorphous" LiPO4
pmv.full_rdf({"Crystal": crystal_lipo4, "Amorphous": amorphous_lipo4}).show()
