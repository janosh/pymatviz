"""Sankey diagram of crystal systems and space groups."""

# /// script
# dependencies = [
#     "matminer>=0.10.1",
# ]
# ///

# %%
import pandas as pd
from matminer.datasets import load_dataset

import pymatviz as pmv


# %% Sankey diagram of crystal systems and space groups
df_phonons = load_dataset("matbench_phonons")

df_sym = pd.DataFrame(
    struct.get_symmetry_dataset(backend="moyopy", return_raw_dataset=True).as_dict()
    for struct in df_phonons["structure"]
).rename(columns={"number": "spg_num"})
df_sym["crystal_system"] = df_sym["spg_num"].map(pmv.utils.spg_to_crystal_sys)


# %%
frequent_symbols = df_sym["spg_num"].value_counts().nlargest(20).index

df_spg = df_sym.query("spg_num in @frequent_symbols")


# %% Common space groups in matbench_phonons, flowing from their crystal systems
widget = pmv.sankey(
    df_spg,
    "crystal_system",
    "spg_num",
    labels_with_counts="percent",
    link_color_mode="gradient",
    style="height: 500px",
)
widget.show()
widget.to_img(
    f"{pmv.utils.ROOT}/assets/svg/sankey-crystal-sys-to-spg-num.svg",
    width=700,
    transparent=True,
)
