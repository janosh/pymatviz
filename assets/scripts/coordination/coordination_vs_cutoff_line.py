# %%
from glob import glob

from matminer.datasets import load_dataset
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import pymatviz as pmv
from pymatviz.enums import Key
from pymatviz.utils.testing import TEST_FILES


pmv.set_plotly_template("pymatviz_white")

df_phonon = load_dataset(data_name := "matbench_phonons")


# %%
formula_spg_str = (
    lambda struct: f"{struct.formula} ({struct.get_space_group_info()[1]})"
)
structures = {
    formula_spg_str(struct := Structure.from_file(file)): struct
    for file in (
        glob(f"{TEST_FILES}/structures/*.json.gz") + glob(f"{TEST_FILES}/xrd/*.cif")
    )
}
key1, key2, key3, *_ = structures

for struct in structures.values():
    spga = SpacegroupAnalyzer(struct)
    sym_struct = spga.get_symmetrized_structure()
    # add wyckoff symbols to each site
    struct.add_oxidation_state_by_guess()
    wyckoff_symbols = ["n/a"] * len(struct)
    for indices, symbol in zip(
        sym_struct.equivalent_indices, sym_struct.wyckoff_symbols, strict=True
    ):
        for index in indices:
            wyckoff_symbols[index] = symbol
    if any(sym == "n/a" for sym in wyckoff_symbols):
        raise ValueError(f"{struct.formula} has n/a {wyckoff_symbols=}")

    struct.add_site_property("wyckoff", wyckoff_symbols)


# %% Coordination vs Cutoff example for a single structure
fig = pmv.coordination_vs_cutoff_line(structures[key1])
fig.layout.title = dict(text=f"Coordination vs Cutoff: {key1}", x=0.5, y=0.98)
fig.layout.margin.t = 50
fig.show()
pmv.io.save_and_compress_svg(fig, "coordination-vs-cutoff-single")


# %% Coordination vs Cutoff example for multiple structures
fig = pmv.coordination_vs_cutoff_line(
    {key1: structures[key1], key2: structures[key2], key3: structures[key3]},
    strategy=(1.0, 6.0),
    num_points=100,
)
fig.layout.title = dict(
    text="Coordination vs Cutoff: Multiple Structures", x=0.5, y=0.98
)
fig.layout.legend.update(x=0, y=1, bgcolor="rgba(0,0,0,0)")
fig.layout.margin.t = 50
fig.show()
pmv.io.save_and_compress_svg(fig, "coordination-vs-cutoff-multiple")


# %% Coordination vs Cutoff example with custom color scheme
custom_colors = {"Pd": "red", "Zr": "blue"}
fig = pmv.coordination_vs_cutoff_line(
    structures[key1], element_color_scheme=custom_colors
)
fig.layout.margin.t = 25
fig.layout.legend.update(x=0, y=1)
fig.show()
pmv.io.save_and_compress_svg(fig, "coordination-vs-cutoff-custom-colors")


# %% plot first 10 structures
fig = pmv.coordination_vs_cutoff_line(df_phonon[Key.structure][:4].tolist())
fig.layout.margin.t = 25
fig.layout.legend.update(x=0, y=1)
fig.show()
