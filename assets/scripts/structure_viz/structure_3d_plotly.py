# %%
from matminer.datasets import load_dataset
from pymatgen.core import Lattice, Structure

import pymatviz as pmv
from pymatviz.enums import ElemColorScheme, Key, SiteCoords


df_phonons = load_dataset("matbench_phonons")


# %% 3d example
fig = pmv.structure_3d_plotly(
    df_phonons[Key.structure].head(6).to_dict(),
    elem_colors=ElemColorScheme.jmol,
    # show_unit_cell={"edge": dict(color="white", width=1.5)},
    hover_text=SiteCoords.cartesian_fractional,
)
fig.show()
# pmv.io.save_and_compress_svg(fig, "matbench-phonons-structures-3d-plotly")


# %% BaTiO3 = https://materialsproject.org/materials/mp-5020
batio3 = Structure(
    lattice=Lattice.cubic(4.0338),
    species=["Ba", "Ti", "O", "O", "O"],
    coords=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
)

fig = pmv.structure_3d_plotly(
    batio3, show_unit_cell={"edge": dict(color="white", width=2)}
)
fig.show()
# pmv.io.save_and_compress_svg(fig, "bato3-structure-3d-plotly")
