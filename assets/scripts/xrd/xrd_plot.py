"""XRD pattern examples."""

# %%
from glob import glob

import matscikit as mk
import numpy as np
from matscikit.io import read_structure

import pymatviz as pmv
from pymatviz.utils.testing import TEST_FILES, formula_spg_str


# %%
structures = {
    formula_spg_str(struct := read_structure(file)): struct
    for file in glob(f"{TEST_FILES}/xrd/*.cif")
    + glob(f"{TEST_FILES}/structures/*.json.gz")
}
xrd_patterns = dict(
    zip(structures, mk.xrd_patterns(list(structures.values())), strict=True)
)
key1, key2, key3, *_ = xrd_patterns


# %% Single pattern: MatterViz labels the 5 strongest peaks with hkl and 2θ
widget = pmv.xrd_plot(xrd_patterns[key1], style="height: 400px")
widget.show()
widget.to_img(
    f"{pmv.utils.ROOT}/assets/svg/xrd-pattern.svg", width=700, transparent=True
)


# %% Overlay patterns of several structures (computed with a Mo K-alpha wavelength)
widget = pmv.xrd_plot(
    {key: structures[key] for key in (key1, key2, key3)}, wavelength=0.7093
)
widget.show()
widget.to_img(
    f"{pmv.utils.ROOT}/assets/svg/xrd-pattern-multiple.svg", width=700, transparent=True
)


# %% Compare a structure's pattern before and after randomly displacing its sites
rng = np.random.default_rng(seed=0)
struct = structures[key1]
pmv.xrd_plot(
    {
        f"{key1} {sigma=}": struct.replace(
            positions=struct.positions + rng.normal(scale=sigma, size=(len(struct), 3))
        )
        for sigma in (0, 0.1, 0.2)
    }
).show()
