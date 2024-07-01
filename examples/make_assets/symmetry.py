# %%
from matminer.datasets import load_dataset
from pymatgen.core import Structure
from tqdm import tqdm

from pymatviz.enums import Key
from pymatviz.io import save_and_compress_svg
from pymatviz.sunburst import spacegroup_sunburst


struct: Structure  # for type hinting


# %%
df_phonons = load_dataset("matbench_phonons")

df_phonons[[Key.spg_symbol, Key.spg_num]] = [
    struct.get_space_group_info() for struct in tqdm(df_phonons[Key.structure])
]


# %% Sunburst Plots
fig = spacegroup_sunburst(df_phonons[Key.spg_num], show_counts="percent")
title = "Matbench Phonons Spacegroup Sunburst"
fig.layout.title = dict(text=f"<b>{title}</b>", x=0.5, y=0.96, font_size=18)
save_and_compress_svg(fig, "spg-num-sunburst")

fig = spacegroup_sunburst(df_phonons[Key.spg_symbol], show_counts="percent")
title = "Matbench Phonons Spacegroup Symbols Sunburst"
fig.layout.title = dict(text=f"<b>{title}</b>", x=0.5, y=0.96, font_size=18)
save_and_compress_svg(fig, "spg-symbol-sunburst")
