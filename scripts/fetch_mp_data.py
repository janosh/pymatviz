# %%
import pandas as pd
from matminer.datasets import load_dataset
from pymatgen.core import Structure
from pymatgen.ext.matproj import MPRester

from ml_matrics import ROOT


# %%
# needs MP API key in ~/.pmgrc.yml available at https://materialsproject.org/dashboard
# pmg config --add PMG_MAPI_KEY <your_key>
formulas = MPRester().query({"nelements": {"$lt": 2}}, ["pretty_formula"])


# %%
df = pd.DataFrame(formulas).rename(columns={"pretty_formula": "formula"})

df.to_csv(f"{ROOT}/data/mp-elements.csv", index=False)


# %%
phonons = load_dataset("matbench_phonons")

phonons[["sg_symbol", "sg_number"]] = phonons.apply(
    lambda row: row.structure.get_space_group_info(), axis=1, result_type="expand"
)

phonons.to_csv(f"{ROOT}/data/matbench-phonons.csv", index=False)


# %% write MP structures to disk
mp_ids = [
    *["mp-568662", "mp-2201", "mp-3834", "mp-2490", "mp-1367", "mp-2542", "mp-2624"],
    *["mp-1170", "mp-23259", "mp-2176", "mp-406", "mp-661", "mp-22875", "mp-4452"],
    *["mp-4979", "mp-9252", "mp-4763", "mp-27529", "mp-22896", "mp-19399"],
]
structs = [MPRester().get_structure_by_material_id(idx) for idx in mp_ids]

for struct, mp_id in zip(structs, mp_ids):
    struct.material_id = mp_id
struct.to(filename=f"data/structures/{mp_id}.yml")


# %% load MP structures from disk
structs = [Structure.from_file(f"../data/structures/{mp_id}.yml") for mp_id in mp_ids]
