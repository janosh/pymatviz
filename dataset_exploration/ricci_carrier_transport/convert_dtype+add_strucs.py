# %%
import pandas as pd
from matminer.datasets import load_dataset
from pymatgen.ext.matproj import MPRester


"""
Unprocessed data in data/carrier_transport.json.gz obtained from https://git.io/JOMwY.
"""


# %%
df_carrier = load_dataset("ricci_boltztrap_mp_tabular")


# %%
df_carrier = pd.concat([df_carrier, pd.json_normalize(df_carrier.data)], axis=1).drop(
    columns=["data", "is_public", "project"]
)

df_carrier = df_carrier.set_index("identifier")
df_carrier.index.name = "mp_id"


# %%
with MPRester() as mpr:
    strucs = mpr.query(
        {"task_ids": {"$in": df_carrier.index.to_list()}},
        ["task_ids", "structure"],
    )


# %%
# get a map from task ID to its structure
struc_df = pd.DataFrame(strucs).explode("task_ids").set_index("task_ids")

df_carrier[struc_df.columns] = struc_df

df_carrier["pretty_formula"] = [struct.formula for struct in df_carrier.structure]


# %% move all units from rows to header
col_map = {
    "type": "functional",
    "ΔE": "ΔE [eV]",
    "V": "V [Å³]",
    "S.p": "S.p [µV/K]",
    "S.n": "S.n [µV/K]",
    "Sᵉ.p.v": "Sᵉ.p.v [µV/K]",
    "Sᵉ.n.v": "Sᵉ.n.v [µV/K]",
    "Sᵉ.p.c": "Sᵉ.p.c [cm⁻³]",
    "Sᵉ.n.c": "Sᵉ.n.c [cm⁻³]",
    "Sᵉ.p.T": "Sᵉ.p.T [K]",
    "Sᵉ.n.T": "Sᵉ.n.T [K]",
    "σ.p": "σ.p [1/Ω/m/s]",
    "σ.n": "σ.n [1/Ω/m/s]",
    "σᵉ.p.v": "σᵉ.p.v [1/Ω/m/s]",
    "σᵉ.n.v": "σᵉ.n.v [1/Ω/m/s]",
    "σᵉ.p.c": "σᵉ.p.c [cm⁻³]",
    "σᵉ.n.c": "σᵉ.n.c [cm⁻³]",
    "σᵉ.n.T": "σᵉ.n.T [K]",
    "σᵉ.p.T": "σᵉ.p.T [K]",
    "PF.p": "PF.p [µW/cm/K²/s]",
    "PF.n": "PF.n [µW/cm/K²/s]",
    "PFᵉ.p.v": "PFᵉ.p.v [µW/cm/K²/s]",
    "PFᵉ.n.v": "PFᵉ.n.v [µW/cm/K²/s]",
    "PFᵉ.p.c": "PFᵉ.p.c [cm⁻³]",
    "PFᵉ.n.c": "PFᵉ.n.c [cm⁻³]",
    "PFᵉ.p.T": "PFᵉ.p.T [K]",
    "PFᵉ.n.T": "PFᵉ.n.T [K]",
    "κₑ.p": "κₑ.p [W/K/m/s]",
    "κₑ.n": "κₑ.n [W/K/m/s]",
    "κₑᵉ.p.v": "κₑᵉ.p.v [W/K/m/s]",
    "κₑᵉ.n.v": "κₑᵉ.n.v [W/K/m/s]",
    "κₑᵉ.p.c": "κₑᵉ.p.c [cm⁻³]",
    "κₑᵉ.n.c": "κₑᵉ.n.c [cm⁻³]",
    "κₑᵉ.p.T": "κₑᵉ.p.T [K]",
    "κₑᵉ.n.T": "κₑᵉ.n.T [K]",
    "mₑᶜ.p.ε̄": "mₑᶜ.p.ε̄ [mₑ]",
    "mₑᶜ.p.ε₁": "mₑᶜ.p.ε₁ [mₑ]",
    "mₑᶜ.p.ε₂": "mₑᶜ.p.ε₂ [mₑ]",
    "mₑᶜ.p.ε₃": "mₑᶜ.p.ε₃ [mₑ]",
    "mₑᶜ.n.ε̄": "mₑᶜ.n.ε̄ [mₑ]",
    "mₑᶜ.n.ε₁": "mₑᶜ.n.ε₁ [mₑ]",
    "mₑᶜ.n.ε₂": "mₑᶜ.n.ε₂ [mₑ]",
    "mₑᶜ.n.ε₃": "mₑᶜ.n.ε₃ [mₑ]",
}
df_carrier = df_carrier.rename(columns=col_map)


# %% convert all target columns to dtype float
units = ["Å³", "µV/K", "cm⁻³", "1/Ω/m/s", "K", "µW/cm/K²/s", "mₑ", "eV", "W/K/m/s"]


for col, vals in df_carrier[col_map.values()].items():
    df_carrier[col] = vals.str.replace("|".join(units), "", regex=True).astype(float)


# %%
df_carrier.to_json(
    "cleaned_ricci_boltztrap_mp_tabular.json.gz", default_handler=lambda x: x.as_dict()
)
