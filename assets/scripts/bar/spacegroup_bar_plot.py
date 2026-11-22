"""Spacegroup bar examples."""

# /// script
# dependencies = [
#     "matminer>=0.10.1",
# ]
# ///

# %%
import kemio
from matminer.datasets import load_dataset

import pymatviz as pmv


# %% Space groups of all structures, searched by kemio in one parallel batch
df_phonons = load_dataset("matbench_phonons")
df_phonons["spg_num"] = pmv.structures.normalize_spacegroups(df_phonons["structure"])
df_phonons["spg_symbol"] = df_phonons["spg_num"].map(kemio.space_group_symbol)


# %% Spacegroup histograms from numbers and symbols
for key, kwargs in (("spg_num", {}), ("spg_symbol", {"log_y": True})):
    widget = pmv.spacegroup_bar_plot(df_phonons[key], **kwargs)
    widget.show()
    widget.to_img(
        f"{pmv.utils.ROOT}/assets/svg/{key.replace('_', '-')}-bar.svg",
        width=600,
        transparent=True,
    )
