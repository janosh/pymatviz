# %%
from matminer.datasets import load_dataset
from tqdm import tqdm

import pymatviz as pmv
from pymatviz.enums import Key


pmv.set_plotly_template("pymatviz_white")


# %%
df_phonons = load_dataset("matbench_phonons")

df_phonons[[Key.spg_symbol, Key.spg_num]] = [
    struct.get_space_group_info()
    for struct in tqdm(df_phonons[Key.structure], desc="Getting spacegroups")
]


# %% Spacegroup histograms
for backend in pmv.typing.BACKENDS:
    fig = pmv.spacegroup_bar(df_phonons[Key.spg_num], backend=backend)
    pmv.io.save_and_compress_svg(fig, f"spg-num-hist-{backend}")

    fig = pmv.spacegroup_bar(df_phonons[Key.spg_symbol], backend=backend)
    pmv.io.save_and_compress_svg(fig, f"spg-symbol-hist-{backend}")
