"""Compare local coordination environments in one crystal and across a dataset."""

# /// script
# dependencies = [
#     "pymatgen>=2026.9.24",
# ]
# ///

# %%
from glob import glob

from pymatgen.core import Structure

import pymatviz as pmv
from pymatviz.data import TEST_FILES


structure_dir = f"{TEST_FILES}/structures"
structure = Structure.from_file(f"{structure_dir}/mp-1183089-Ac4Mg2-monoclinic.json.gz")
widget = pmv.chem_env_hierarchy(structure, style="height: 480px", kind="treemap")
widget.show()
widget.to_img(
    f"{pmv.utils.ROOT}/assets/svg/chem-env-treemap-basic.svg",
    width=700,
    transparent=True,
)


# %% Retain the most common coordination numbers and environments within each group.
structures = [
    Structure.from_file(path) for path in sorted(glob(f"{structure_dir}/*.json.gz"))
]
widget = pmv.chem_env_hierarchy(
    structures,
    max_coordination_numbers=3,
    max_environments=4,
    style="height: 480px",
    kind="treemap",
)
widget.show()
widget.to_img(
    f"{pmv.utils.ROOT}/assets/svg/chem-env-treemap-large-dataset.svg",
    width=700,
    transparent=True,
)
