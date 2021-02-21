# %%
import pandas as pd
from pymatgen import MPRester

from mlmatrics import ROOT

# %%
with MPRester(api_key="X2UaF2zkPMcFhpnMN") as mpr:
    formulas = mpr.query({"nelements": {"$lt": 2}}, ["pretty_formula"])


# %%
df = pd.DataFrame(formulas).rename(columns={"pretty_formula": "formula"})

df.to_csv(f"{ROOT}/data/mp-n_elements<2.csv", index=False)
