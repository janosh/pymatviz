# %%
import pandas as pd
from matminer.datasets import load_dataset

import pymatviz as pmv
from pymatviz.enums import Key


pmv.set_plotly_template("pymatviz_white")


# %% Sankey diagram of crystal systems and space groups
data_name = "matbench_phonons"
df_phonons = load_dataset(data_name)

df_sym = pd.DataFrame(
    struct.get_symmetry_dataset(backend="moyopy", return_raw_dataset=True).as_dict()
    for struct in df_phonons[Key.structure]
).rename(columns={"number": Key.spg_num})
df_sym[Key.crystal_system] = df_sym[Key.spg_num].map(pmv.utils.spg_to_crystal_sys)


# %%
frequent_symbols = df_sym[Key.spg_num].value_counts().nlargest(20).index

df_spg = df_sym.query(f"{Key.spg_num} in @frequent_symbols")


# %%
fig = pmv.sankey_from_2_df_cols(
    df_spg, [Key.crystal_system, Key.spg_num], labels_with_counts="percent"
)
title = f"Common Space Groups in {data_name}"
fig.layout.title = dict(text=title, x=0.5, y=0.95)
fig.layout.margin.t = 50
fig.show()
pmv.io.save_and_compress_svg(fig, f"sankey-{data_name}")
