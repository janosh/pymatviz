# %%
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure

from pymatviz import plot_xrd_pattern, set_plotly_template
from pymatviz.io import save_and_compress_svg
from pymatviz.utils import TEST_FILES


set_plotly_template("pymatviz_white")


# %%
structures = map(
    Structure.from_file,
    (
        f"{TEST_FILES}/xrd/Bi2Zr2O7-Fm3m-experimental-sqs.cif",
        f"{TEST_FILES}/xrd/mp-756175-Zr2Bi2O7.cif",
    ),
)
xrd_patterns = {
    struct.formula: XRDCalculator().get_pattern(struct) for struct in structures
}


# %%
fig = plot_xrd_pattern(xrd_patterns["Zr4 Bi4 O14"], annotate_peaks=5)
fig.show()
save_and_compress_svg(fig, "xrd-pattern")


# %%
fig = plot_xrd_pattern(xrd_patterns)
fig.show()
save_and_compress_svg(fig, "xrd-pattern-multiple")
