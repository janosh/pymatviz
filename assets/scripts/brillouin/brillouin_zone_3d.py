# %%
from glob import glob

from pymatgen.core import Structure

import pymatviz as pmv
from pymatviz.utils.testing import TEST_FILES


# %% render example Brillouin zone for each crystal system
structures = {
    tuple(
        path.split("/")[-1].replace(".json.gz", "").rsplit("-", maxsplit=2)
    ): Structure.from_file(path)
    for path in glob(f"{TEST_FILES}/structures/*-*-*-*.json.gz")
}

for (mat_id, formula, system), struct in structures.items():
    fig = pmv.brillouin_zone_3d(struct)
    title = f"{formula} {system.title()} ({mat_id})"
    fig.layout.title = dict(text=title, x=0.5, y=0.97, font_size=20)
    fig.show()
    pmv.io.save_and_compress_svg(fig, f"brillouin-{system.lower()}-{mat_id}")
