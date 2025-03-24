# %%
from matminer.datasets import load_dataset
from tqdm import tqdm

import pymatviz as pmv
from pymatviz.enums import Key


# %%
df_phonons = load_dataset("matbench_phonons")

df_phonons[Key.spg_num] = [
    struct.get_symmetry_dataset(backend="moyopy", return_raw_dataset=True).number
    for struct in tqdm(df_phonons[Key.structure])
]

df_phonons[Key.spg_symbol] = df_phonons[Key.spg_num].map(
    pmv.utils.spg_num_to_from_symbol
)


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


# %% Demonstrate max_slices with "other" mode
# Show only top 3 space groups per crystal system, combine the rest into "Other"
max_slices = 6
fig = pmv.spacegroup_sunburst(
    df_phonons[Key.spg_num],
    show_counts="value+percent",
    max_slices=max_slices,
    max_slices_mode="other",
)
title = f"Matbench Phonons Space Groups - Top {max_slices} Crystal Systems (Other Mode)"
fig.layout.title = dict(text=f"<b>{title}</b>", x=0.5, y=0.96, font_size=18)
fig.layout.update(height=600)
fig.show()
# pmv.io.save_and_compress_svg(fig, "spg-num-sunburst-max-slices-other")


# %% Demonstrate max_slices with "drop" mode
# Show only top 2 space groups per crystal system, discard the rest
max_slices = 5
fig = pmv.spacegroup_sunburst(
    df_phonons[Key.spg_num],
    show_counts="value+percent",
    max_slices=max_slices,
    max_slices_mode="drop",
)
title = f"Matbench Phonons Space Groups - Top {max_slices} Crystal Systems (Drop Mode)"
fig.layout.title = dict(text=f"<b>{title}</b>", x=0.5, y=0.96, font_size=18)
fig.layout.update(height=600)
fig.show()
# pmv.io.save_and_compress_svg(fig, "spg-num-sunburst-max-slices-drop")
