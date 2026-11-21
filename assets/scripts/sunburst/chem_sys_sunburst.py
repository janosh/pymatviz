"""Explore formula grouping and the most common systems in Ward metallic glasses."""

# %%
import pandas as pd

import pymatviz as pmv
from pymatviz.enums import Key


# Fe2O3 and Fe4O6 share a reduced formula; FeO belongs to the same chemical system.
formulas = ["Fe2O3", "Fe4O6", "FeO", "Li2O", "LiFeO2"]
for group_by in ("formula", "reduced_formula", "chem_sys"):
    pmv.chem_sys_sunburst(formulas, group_by=group_by).show()


# %% Group rare systems into an Other sector within each arity.
ward = pd.read_csv(
    f"{pmv.ROOT}/examples/ward_metallic_glasses/ward-metallic-glasses.csv.xz",
    na_values=(),
).query("comment.isna()")
widget = pmv.chem_sys_sunburst(
    ward[Key.composition],
    max_systems=5,
    inner_radius=0.18,
    label_text="label+percent",
    style="height: 480px",
)
widget.show()
widget.to_img(
    f"{pmv.ROOT}/assets/svg/chem-sys-sunburst-ward-bmg.svg", width=700, transparent=True
)
