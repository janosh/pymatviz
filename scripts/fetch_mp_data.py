# %%
import pandas as pd
from matminer.datasets import load_dataset
from pymatgen.ext.matproj import MPRester

from ml_matrics import ROOT


# %%
with MPRester(api_key="X2UaF2zkPMcFhpnMN") as mpr:
    formulas = mpr.query({"nelements": {"$lt": 2}}, ["pretty_formula"])


# %%
df = pd.DataFrame(formulas).rename(columns={"pretty_formula": "formula"})

df.to_csv(f"{ROOT}/data/mp-n_elements<2.csv", index=False)


# %%
phonons = load_dataset("matbench_phonons")

phonons[["sg_symbol", "sg_number"]] = phonons.apply(
    lambda row: row.structure.get_space_group_info(), axis=1, result_type="expand"
)

phonons.to_csv(f"{ROOT}/data/matbench-phonons.csv", index=False)
