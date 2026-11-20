"""Spacegroup sunburst examples."""

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
widget = pmv.spacegroup_sunburst(df_phonons[Key.spg_num], label_text="label+percent")
widget.show()
widget.to_img(f"{pmv.ROOT}/assets/svg/spg-num-sunburst.svg", width=600)

widget = pmv.spacegroup_sunburst(df_phonons[Key.spg_symbol], label_text="label+percent")
widget.show()
widget.to_img(f"{pmv.ROOT}/assets/svg/spg-symbol-sunburst.svg", width=600)


# %% Show the top 6 space groups per crystal system, combine the rest into "Other"
pmv.spacegroup_sunburst(
    df_phonons[Key.spg_num], max_slices=6, max_slices_mode="other"
).show()


# %% Show the top 5 space groups per crystal system, discard the rest
pmv.spacegroup_sunburst(
    df_phonons[Key.spg_num], max_slices=5, max_slices_mode="drop"
).show()
