"""Example script comparing MACE vs MP elastic constants.

Adapted from the elastic tensor benchmark in the MACE-MP-0 paper by Matthew Kuner.
See Figure 52 in https://arxiv.org/pdf/2401.00096.

This script demonstrates how to:
1. Fetch elastic constant data from the Materials Project
2. Calculate elastic constants using MACE
3. Create density scatter plots comparing MACE vs MP predictions
"""

# %%
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from ase.units import GPa
from emmet.core.elasticity import ElasticityDoc
from mace.calculators import mace_mp
from matcalc.elasticity import ElasticityCalc
from tqdm import tqdm

import pymatviz as pmv


try:
    from mp_api.client import MPRester
except ImportError:
    raise SystemExit(0) from None

pmv.set_plotly_template("pymatviz_white")
checkpoint = "https://github.com/ACEsuit/mace-mp/releases/download/mace_matpes_0/MACE-matpes-pbe-omat-ft.model"


# %%
print("Fetching elastic constant data from MP...")
num_chunks, chunk_size = 1, 100
with MPRester() as mpr:
    mp_elastic_docs: list[ElasticityDoc] = mpr.materials.elasticity.search(
        num_chunks=num_chunks, chunk_size=chunk_size, all_fields=True
    )


print(f"Calculating elastic constants with {checkpoint=}...")
error_list: dict[str, Exception] = {}
ml_results: dict[str, Any] = {}
calculator = mace_mp(model=checkpoint)
elasticity_calc = ElasticityCalc(calculator, relax_structure=True)

for elastic_doc in (
    pbar := tqdm(mp_elastic_docs, desc="elastic constants with matcalc")
):
    pbar.set_postfix_str(elastic_doc.material_id)
    try:
        elast_dict = elasticity_calc.calc(elastic_doc.structure)
        # convert eV/A^3 to GPa
        for key in ("bulk", "shear"):
            elast_dict[f"{key}_modulus_vrh"] /= GPa
        ml_results[elastic_doc.material_id] = elast_dict

    except Exception as exc:  # noqa: BLE001
        print(f"Error processing structure {elastic_doc.material_id}: {exc}")
        error_list[elastic_doc.material_id] = exc

df_ml = pd.DataFrame(ml_results).T.convert_dtypes()
k_vrh_pred_col = "bulk_modulus_vrh_pred"
g_vrh_pred_col = "shear_modulus_vrh_pred"
col_map = {"bulk_modulus_vrh": k_vrh_pred_col, "shear_modulus_vrh": g_vrh_pred_col}
df_ml = df_ml.rename(columns=col_map)
k_vrh_ref_col = "bulk_modulus_vrh_mp"
df_ml[k_vrh_ref_col] = pd.Series(
    {doc.material_id: getattr(doc.bulk_modulus, "vrh", None) for doc in mp_elastic_docs}
)
g_vrh_ref_col = "shear_modulus_vrh_mp"
df_ml[g_vrh_ref_col] = pd.Series(
    {
        doc.material_id: getattr(doc.shear_modulus, "vrh", None)
        for doc in mp_elastic_docs
    }
)
# filter unrealistic values
df_ml = df_ml.query(f"0 <= {k_vrh_ref_col} < 1e4 and 0 <= {g_vrh_ref_col} < 1e4")

# Bulk modulus plot
fig = px.scatter(df_ml, x=k_vrh_ref_col, y=k_vrh_pred_col)
fig.layout.title.update(text="MACE vs MP Bulk Modulus VRH", x=0.5)

pmv.powerups.enhance_parity_plot(fig)
fig.layout.margin.t = 30
fig.show()

# Shear modulus plot
fig = px.scatter(df_ml, x=g_vrh_ref_col, y=g_vrh_pred_col)
fig.layout.title.update(text="MACE vs MP Shear Modulus VRH", x=0.5)
pmv.powerups.enhance_parity_plot(fig)
fig.layout.margin.t = 30
fig.show()

# df_ml.to_csv(f"elastic-constants-{checkpoint.split('/')[-1]}.csv", index=False)


# %% plot elastic tensor single material MP-149 Si2
print("\nPredicting elastic constants for MP-149 Si2...")
with MPRester() as mpr:
    structure = mpr.get_structure_by_material_id("mp-149")
    print(f"Fetched {structure.formula=}")
    # Get MP reference elastic data
    elastic_doc = mpr.materials.elasticity.get_data_by_id("mp-149")

calculator = mace_mp(model=checkpoint)
elasticity_calc = ElasticityCalc(calculator, relax_structure=True)
ml_results = elasticity_calc.calc(structure)

fig_ml = px.imshow(ml_results["elastic_tensor"].voigt / GPa, text_auto=".2f")
fig_ml.layout.coloraxis.colorbar.title.update(
    text="ML Elastic Tensor [GPa]", side="right"
)
fig_ml.show()

fig_pbe = px.imshow(np.asarray(elastic_doc.elastic_tensor.raw), text_auto=".2f")
fig_pbe.layout.coloraxis.colorbar.title.update(
    text="PBE Elastic Tensor [GPa]", side="right"
)
fig_pbe.show()
