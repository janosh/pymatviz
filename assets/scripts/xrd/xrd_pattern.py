"""XRD pattern examples."""

# %%
from glob import glob

from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure

import pymatviz as pmv
from pymatviz.utils.testing import TEST_FILES


pmv.set_plotly_template("pymatviz_white")


# %%
formula_spg_str = (
    lambda struct: f"{struct.formula} ({struct.get_space_group_info()[1]})"
)
structures = {
    formula_spg_str(struct := Structure.from_file(file)): struct
    for file in glob(f"{TEST_FILES}/xrd/*.cif")
    + glob(f"{TEST_FILES}/structures/*.json.gz")
}
xrd_patterns = {
    key: XRDCalculator().get_pattern(struct) for key, struct in structures.items()
}
key1, key2, key3, *_ = xrd_patterns


# %%
fig = pmv.xrd_pattern(xrd_patterns[key1], annotate_peaks=5)
fig.layout.margin.t = 40
fig.layout.title = dict(text=key1, x=0.5, y=0.97)
fig.show()
# pmv.io.save_and_compress_svg(fig, "xrd-pattern")


# %%
fig = pmv.xrd_pattern({key1: xrd_patterns[key1], key2: xrd_patterns[key2]})
fig.show()
# pmv.io.save_and_compress_svg(fig, "xrd-pattern-multiple")


# %%
fig = pmv.xrd_pattern(
    {key1: xrd_patterns[key1], key2: xrd_patterns[key2], key3: xrd_patterns[key3]},
    stack="horizontal",
    annotate_peaks=3,
    show_angles=True,
)
fig.layout.title = dict(text="Horizontally Stacked XRD Patterns", x=0.5, y=0.97)
fig.layout.margin.t = 40
fig.show()
# pmv.io.save_and_compress_svg(fig, "xrd-pattern-horizontal-stack")


# %% New example with vertical stacking and custom subplot titles
fig = pmv.xrd_pattern(
    {f"{key1} {idx=}": structures[key1].copy().perturb(idx * 0.5) for idx in range(3)},
    stack="vertical",
    annotate_peaks=1,
    show_angles=True,
    subtitle_kwargs=dict(x=1, xanchor="right", font_size=14),
)
fig.layout.title = dict(
    text="Vertically Stacked XRD Patterns with Custom Subplot Titles", x=0.5, y=0.97
)
fig.layout.margin.t = 40
fig.show()
# pmv.io.save_and_compress_svg(fig, "xrd-pattern-vertical-stack")
