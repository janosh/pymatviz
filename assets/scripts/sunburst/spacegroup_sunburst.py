"""Spacegroup sunburst examples."""

# /// script
# dependencies = [
#     "matminer>=0.10.1",
#     "tqdm>=4.67",
# ]
# ///

# %%
import matscikit as mk
from matminer.datasets import load_dataset
from tqdm import tqdm

import pymatviz as pmv


# %%
df_phonons = load_dataset("matbench_phonons")

df_phonons["spg_num"] = [
    struct.get_symmetry_dataset(backend="moyopy", return_raw_dataset=True).number
    for struct in tqdm(df_phonons["structure"])
]

df_phonons["spg_symbol"] = df_phonons["spg_num"].map(mk.space_group_symbol)


# %% Sunburst Plots
widget = pmv.spacegroup_sunburst(df_phonons["spg_num"], label_text="label+percent")
widget.show()
widget.to_img(
    f"{pmv.utils.ROOT}/assets/svg/spg-num-sunburst.svg", width=600, transparent=True
)

widget = pmv.spacegroup_sunburst(df_phonons["spg_symbol"], label_text="label+percent")
widget.show()


# %% Show the top 6 space groups per crystal system, combine the rest into "Other"
pmv.spacegroup_sunburst(
    df_phonons["spg_num"], max_spacegroups=6, overflow="other"
).show()


# %% Show the top 5 space groups per crystal system, discard the rest
pmv.spacegroup_sunburst(
    df_phonons["spg_num"], max_spacegroups=5, overflow="drop"
).show()
