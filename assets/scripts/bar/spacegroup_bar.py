"""Spacegroup bar examples."""

# %%
import pandas as pd
from matminer.datasets import load_dataset

import pymatviz as pmv
from pymatviz.enums import Key


pmv.set_plotly_template("pymatviz_white")


# %%
df_phonons = load_dataset("matbench_phonons")

df_sym = pd.DataFrame(
    struct.get_symmetry_dataset(backend="moyopy", return_raw_dataset=True).as_dict()
    for struct in df_phonons[Key.structure]
)
df_phonons[Key.spg_num] = df_sym["number"]
df_phonons[Key.spg_symbol] = df_sym["number"].map(pmv.utils.spg_num_to_from_symbol)


# %% Spacegroup histograms
fig = pmv.spacegroup_bar(df_phonons[Key.spg_num])
fig.show()
# pmv.io.save_and_compress_svg(fig, f"spg-num-hist-{backend}")

fig = pmv.spacegroup_bar(df_phonons[Key.spg_symbol])
fig.show()
# pmv.io.save_and_compress_svg(fig, f"spg-symbol-hist-{backend}")
