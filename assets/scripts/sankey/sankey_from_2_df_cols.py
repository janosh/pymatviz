"""Sankey diagram of crystal systems and space groups."""

# %%
import pandas as pd
from matminer.datasets import load_dataset

import pymatviz as pmv
from pymatviz.enums import Key


# %% Sankey diagram of crystal systems and space groups
df_phonons = load_dataset("matbench_phonons")

df_sym = pd.DataFrame(
    struct.get_symmetry_dataset(backend="moyopy", return_raw_dataset=True).as_dict()
    for struct in df_phonons[Key.structure]
).rename(columns={"number": Key.spg_num})
df_sym[Key.crystal_system] = df_sym[Key.spg_num].map(pmv.utils.spg_to_crystal_sys)


# %%
frequent_symbols = df_sym[Key.spg_num].value_counts().nlargest(20).index

df_spg = df_sym.query(f"{Key.spg_num} in @frequent_symbols")


# %% Common space groups in matbench_phonons, flowing from their crystal systems
widget = pmv.sankey_from_2_df_cols(
    df_spg,
    [Key.crystal_system, Key.spg_num],
    labels_with_counts="percent",
    link_color_mode="gradient",
    style="height: 500px",
)
widget.show()
widget.to_img(
    f"{pmv.ROOT}/assets/svg/sankey-crystal-sys-to-spg-num.svg",
    width=700,
    transparent=True,
)
