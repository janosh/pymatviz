# %%
from matminer.datasets import load_dataset
from moyopy import SpaceGroupType
from tqdm import tqdm

import pymatviz as pmv
from pymatviz.enums import Key


pmv.set_plotly_template("pymatviz_white")


# %%
df_phonons = load_dataset("matbench_phonons")

df_phonons[Key.spg_num] = [
    struct.get_symmetry_dataset(backend="moyopy", return_raw_dataset=True).number
    for struct in tqdm(df_phonons[Key.structure], desc="Getting spacegroups")
]
df_phonons[Key.spg_symbol] = [
    SpaceGroupType(spg_num).hm_short for spg_num in df_phonons[Key.spg_num]
]


# %% Spacegroup histograms
fig = pmv.spacegroup_bar(df_phonons[Key.spg_num])
fig.show()
# pmv.io.save_and_compress_svg(fig, f"spg-num-hist-{backend}")

fig = pmv.spacegroup_bar(df_phonons[Key.spg_symbol])
fig.show()
# pmv.io.save_and_compress_svg(fig, f"spg-symbol-hist-{backend}")
