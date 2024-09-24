from __future__ import annotations

from matminer.datasets import load_dataset

import pymatviz as pmv
from pymatviz.enums import Key


pmv.set_plotly_template("pymatviz_white")

df_phonons = load_dataset("matbench_phonons")


# get the 2 largest structures
df_phonons[Key.n_sites] = df_phonons[Key.structure].apply(len)

# plot element-pair RDFs for each structure
for struct in df_phonons.nlargest(2, Key.n_sites)[Key.structure]:
    fig = pmv.element_pair_rdfs(struct, n_bins=100, cutoff=10)
    formula = struct.formula
    fig.layout.title.update(text=f"Pairwise RDFs - {formula}", x=0.5, y=0.98)
    fig.layout.margin = dict(l=40, r=0, t=50, b=0)

    fig.show()
    pmv.io.save_and_compress_svg(fig, f"element-pair-rdfs-{formula.replace(' ', '')}")
