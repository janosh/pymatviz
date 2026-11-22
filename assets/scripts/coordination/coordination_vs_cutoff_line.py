"""Coordination vs cutoff line examples."""

# /// script
# dependencies = [
#     "matminer>=0.10.1",
#     "pymatgen>=2026.9.24",
# ]
# ///

# %%
from glob import glob

from matminer.datasets import load_dataset
from pymatgen.core import Structure

import pymatviz as pmv
from pymatviz.data import TEST_FILES
from pymatviz.structures import formula_spg_str


df_phonon = load_dataset(data_name := "matbench_phonons")


# %%
structures = {
    formula_spg_str(struct := Structure.from_file(file)): struct
    for file in (
        glob(f"{TEST_FILES}/structures/*.json.gz") + glob(f"{TEST_FILES}/xrd/*.cif")
    )
}
key1, key2, key3, *_ = structures


# %% Coordination vs Cutoff example for a single structure
widget = pmv.coordination_vs_cutoff_line(structures[key1])
widget.show()
widget.to_img(
    f"{pmv.utils.ROOT}/assets/svg/coordination-vs-cutoff-single.svg",
    width=700,
    transparent=True,
)


# %% Multiple structures: legend group and line dash per structure
widget = pmv.coordination_vs_cutoff_line(
    {key1: structures[key1], key2: structures[key2], key3: structures[key3]},
    cutoff_range=(1.0, 6.0),
    num_points=100,
)
widget.show()
widget.to_img(
    f"{pmv.utils.ROOT}/assets/svg/coordination-vs-cutoff-multiple.svg",
    width=700,
    transparent=True,
)


# %% Custom element colors
pmv.coordination_vs_cutoff_line(
    structures[key1], element_colors={"Pd": "red", "Zr": "blue"}
).show()


# %% First 4 matbench_phonons structures
pmv.coordination_vs_cutoff_line(df_phonon["structure"][:4].tolist()).show()
