"""Compare local coordination environments in one crystal and across a dataset."""

# %%
from glob import glob

from pymatgen.core import Structure

import pymatviz as pmv
from pymatviz.utils.testing import TEST_FILES


structure_dir = f"{TEST_FILES}/structures"
structure = Structure.from_file(f"{structure_dir}/mp-1183089-Ac4Mg2-monoclinic.json.gz")
widget = pmv.chem_env_treemap(
    structure, chem_env_settings="chemenv", style="height: 480px"
)
widget.show()
widget.to_img(
    f"{pmv.ROOT}/assets/svg/chem-env-treemap-basic.svg", width=700, transparent=True
)


# %% Retain the most common coordination numbers and environments within each group.
structures = [
    Structure.from_file(path) for path in sorted(glob(f"{structure_dir}/*.json.gz"))
]
widget = pmv.chem_env_treemap(
    structures, max_coordination_numbers=3, max_environments=4, style="height: 480px"
)
widget.show()
widget.to_img(
    f"{pmv.ROOT}/assets/svg/chem-env-treemap-large-dataset.svg",
    width=700,
    transparent=True,
)
