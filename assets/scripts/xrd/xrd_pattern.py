"""XRD pattern examples."""

# %%
from glob import glob

from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure

import pymatviz as pmv
from pymatviz.utils.testing import TEST_FILES, formula_spg_str


# %%
structures = {
    formula_spg_str(struct := Structure.from_file(file)): struct
    for file in glob(f"{TEST_FILES}/xrd/*.cif")
    + glob(f"{TEST_FILES}/structures/*.json.gz")
}
xrd_patterns = {
    key: XRDCalculator().get_pattern(struct) for key, struct in structures.items()
}
key1, key2, key3, *_ = xrd_patterns


# %% Single pattern: MatterViz labels the 5 strongest peaks with hkl and 2θ
widget = pmv.xrd_pattern(xrd_patterns[key1], style="height: 400px")
widget.show()
widget.to_img(f"{pmv.ROOT}/assets/svg/xrd-pattern.svg", width=700)


# %% Overlay patterns of several structures (computed with a Mo K-alpha wavelength)
widget = pmv.xrd_pattern(
    {key: structures[key] for key in (key1, key2, key3)}, wavelength=0.7093
)
widget.show()
widget.to_img(f"{pmv.ROOT}/assets/svg/xrd-pattern-multiple.svg", width=700)


# %% Compare a structure's pattern before and after perturbing its sites
pmv.xrd_pattern(
    {f"{key1} {idx=}": structures[key1].copy().perturb(idx * 0.5) for idx in range(3)}
).show()
