# %%
from matminer.datasets import load_dataset
from moyopy import SpaceGroupType
from tqdm import tqdm

import pymatviz as pmv
from pymatviz.enums import Key


map_spg_num_to_symbol = {
    # SpaceGroupType(number).hm_short is separated by a space like "F m -3 m"
    number: SpaceGroupType(number).hm_short.replace(" ", "")
    for number in range(1, 230 + 1)
}


# %%
df_phonons = load_dataset("matbench_phonons")

df_phonons[Key.spg_num] = [
    struct.get_symmetry_dataset(backend="moyopy", return_raw_dataset=True).number
    for struct in tqdm(df_phonons[Key.structure])
]

df_phonons[Key.spg_symbol] = df_phonons[Key.spg_num].map(map_spg_num_to_symbol)


# %% Sunburst Plots
fig = pmv.spacegroup_sunburst(df_phonons[Key.spg_num], show_counts="percent")
title = "Matbench Phonons Spacegroup Sunburst"
fig.layout.title = dict(text=f"<b>{title}</b>", x=0.5, y=0.96, font_size=18)
fig.show()
# pmv.io.save_and_compress_svg(fig, "spg-num-sunburst")

fig = pmv.spacegroup_sunburst(df_phonons[Key.spg_symbol], show_counts="percent")
title = "Matbench Phonons Spacegroup Symbols Sunburst"
fig.layout.title = dict(text=f"<b>{title}</b>", x=0.5, y=0.96, font_size=18)
fig.show()
# pmv.io.save_and_compress_svg(fig, "spg-symbol-sunburst")
