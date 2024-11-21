# %%
from matminer.datasets import load_dataset

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
pmv.io.save_and_compress_svg(fig, "matbench-phonons-structures-3d-plotly")
