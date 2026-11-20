# /// script
# dependencies = [
#     "seekpath>=2.1",
# ]
# ///
"""Brillouin zone 3D examples."""

# %%
from glob import glob

from IPython.display import display
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

import pymatviz as pmv
from pymatviz.utils.testing import TEST_FILES


# keys are (material ID, formula, crystal system) from the file names
structures = {
    tuple(path.split("/")[-1].removesuffix(".json.gz").rsplit("-", maxsplit=2)): (
        Structure.from_file(path)
    )
    for path in glob(f"{TEST_FILES}/structures/mp-*-*-*.json.gz")
}


# %% render the Brillouin zone of one structure per crystal system
for struct in structures.values():
    display(pmv.brillouin_zone_3d(struct, show_vectors=True))


# %% pass ase.Atoms
display(
    pmv.brillouin_zone_3d(AseAtomsAdaptor.get_atoms(next(iter(structures.values()))))
)


# %% grid of all crystal systems with translucent zones and the irreducible wedge
labeled = {
    f"{formula} {system}": struct for (_, formula, system), struct in structures.items()
}
display(pmv.brillouin_zone_3d(labeled, n_cols=4, surface_opacity=0.4, show_ibz=True))
