"""Brillouin zone widget examples. MatterViz standardizes each input cell, so zones
and k-path labels follow the Setyawan-Curtarolo convention whatever cell is passed.
"""

# /// script
# dependencies = [
#     "pillow>=11",
#     "pymatgen>=2026.9.24",
# ]
# ///

from glob import glob

from IPython.display import display
from PIL import Image
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

import pymatviz as pmv
from pymatviz.data import TEST_FILES


# keys are (material ID, formula, crystal system) from the file names
structures = {
    tuple(path.split("/")[-1].removesuffix(".json.gz").rsplit("-", maxsplit=2)): (
        Structure.from_file(path)
    )
    for path in glob(f"{TEST_FILES}/structures/mp-*-*-*.json.gz")
}


# %% render the Brillouin zone of one structure per crystal system
for (mp_id, _formula, system), struct in structures.items():
    widget = pmv.brillouin_zone(structure=struct, show_vectors=True)
    display(widget)
    if system in {"cubic", "hexagonal", "monoclinic", "orthorhombic"}:  # in readme
        img_path = f"{pmv.utils.ROOT}/assets/png/brillouin-{system}-{mp_id}.png"
        widget.to_img(img_path, width=500, transparent=True)
        # trimming leaves each zone's own aspect (tall monoclinic, flat orthorhombic):
        # center on a square canvas so the readme grid cells match
        img = Image.open(img_path)
        side = max(img.size)
        square = Image.new("RGBA", (side, side))
        square.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
        square.save(img_path, optimize=True)


# %% pass ase.Atoms
atoms = AseAtomsAdaptor.get_atoms(next(iter(structures.values())))
display(pmv.brillouin_zone(structure=atoms))


# %% grid of all crystal systems with translucent zones and the irreducible wedge
zones = {
    f"{formula} {system}": pmv.brillouin_zone(
        structure=struct, surface_opacity=0.4, show_ibz=True
    )
    for (_, formula, system), struct in structures.items()
}
display(pmv.widget_grid(zones, n_cols=4))
