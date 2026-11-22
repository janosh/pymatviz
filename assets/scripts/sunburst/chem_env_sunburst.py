"""Compare coordination environments with site counts or equal weight per structure."""

# /// script
# dependencies = [
#     "pymatgen>=2026.9.24",
# ]
# ///

# %%
from glob import glob

from pymatgen.core import Structure

import pymatviz as pmv
from pymatviz.utils.testing import TEST_FILES


structure_dir = f"{TEST_FILES}/structures"
structure = Structure.from_file(
    f"{structure_dir}/mp-1207297-Ac2Br2O1-tetragonal.json.gz"
)
widget = pmv.chem_env_hierarchy(
    structure, label_text="label+value", style="height: 480px"
)
widget.show()
widget.to_img(
    f"{pmv.utils.ROOT}/assets/svg/chem-env-sunburst-basic.svg",
    width=700,
    transparent=True,
)


# %% Each structure contributes one unit, regardless of its number of sites.
structures = [
    Structure.from_file(path) for path in sorted(glob(f"{structure_dir}/*.json.gz"))
]
widget = pmv.chem_env_hierarchy(
    structures,
    normalize=True,
    max_coordination_numbers=4,
    label_text="label+percent",
    style="height: 480px",
)
widget.show()
widget.to_img(
    f"{pmv.utils.ROOT}/assets/svg/chem-env-sunburst-normalized.svg",
    width=700,
    transparent=True,
)
