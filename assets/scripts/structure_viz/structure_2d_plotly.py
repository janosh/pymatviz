# %%
from matminer.datasets import load_dataset

import pymatviz as pmv
from pymatviz.enums import ElemColorScheme, Key


df_phonons = load_dataset("matbench_phonons")


# %% Plot Matbench phonon structures with plotly
fig = pmv.structure_2d_plotly(
    df_phonons[Key.structure].head(6).to_dict(),
    # show_unit_cell={"edge": dict(color="white", width=1.5)},
    # show_sites=dict(line=None),
    elem_colors=ElemColorScheme.jmol,
    n_cols=3,
    subplot_title=lambda _struct, _key: dict(font=dict(color="black")),
    hover_text=lambda site: f"<b>{site.frac_coords}</b>",
)
fig.layout.paper_bgcolor = "rgba(255,255,255,0.4)"
fig.show()
# pmv.io.save_and_compress_svg(fig, "matbench-phonons-structures-2d-plotly")
