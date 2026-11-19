"""Compare formula-level and system-level treemaps of compositions."""

# %%
import pandas as pd

import pymatviz as pmv
from pymatviz.enums import Key


formulas = ["Fe2O3", "Fe4O6", "FeO", "Li2O", "LiFeO2"]
widget = pmv.chem_sys_treemap(
    formulas, group_by="formula", label_text="label+value", style="height: 480px"
)
widget.show()
widget.to_img(f"{pmv.ROOT}/assets/svg/chem-sys-treemap-formula.svg", width=700)


# %% Compare all observed systems with a view limited to five per arity.
ward = pd.read_csv(
    f"{pmv.ROOT}/examples/ward_metallic_glasses/ward-metallic-glasses.csv.xz",
    na_values=(),
).query("comment.isna()")
for max_systems, name in ((None, "ward-bmg"), (5, "top-5")):
    widget = pmv.chem_sys_treemap(
        ward[Key.composition], max_systems=max_systems, style="height: 480px"
    )
    widget.show()
    widget.to_img(f"{pmv.ROOT}/assets/svg/chem-sys-treemap-{name}.svg", width=700)
