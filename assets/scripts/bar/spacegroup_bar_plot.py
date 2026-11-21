"""Spacegroup bar examples."""

# %%
import pandas as pd
from matminer.datasets import load_dataset

import pymatviz as pmv
from pymatviz.enums import Key


# %%
df_phonons = load_dataset("matbench_phonons")

df_sym = pd.DataFrame(
    struct.get_symmetry_dataset(backend="moyopy", return_raw_dataset=True).as_dict()
    for struct in df_phonons[Key.structure]
)
df_phonons[Key.spg_num] = df_sym["number"]
df_phonons[Key.spg_symbol] = df_sym["number"].map(pmv.utils.spg_num_to_from_symbol)


# %% Spacegroup histograms from numbers and symbols
for key, kwargs in ((Key.spg_num, {}), (Key.spg_symbol, {"log_y": True})):
    widget = pmv.spacegroup_bar_plot(df_phonons[key], **kwargs)
    widget.show()
    widget.to_img(
        f"{pmv.utils.ROOT}/assets/svg/{key.replace('_', '-')}-bar.svg",
        width=600,
        transparent=True,
    )
