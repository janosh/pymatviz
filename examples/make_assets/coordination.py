# %%
from glob import glob

from matminer.datasets import load_dataset
from pymatgen.analysis.local_env import CrystalNN, VoronoiNN
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import pymatviz as pmv
from pymatviz.coordination import SplitMode
from pymatviz.enums import Key
from pymatviz.utils import TEST_FILES


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


# %% Single structure example
fig = pmv.coordination_hist(structures[key1], strategy=4.0)
fig.layout.title = dict(text=f"Coordination Histogram: {key1}", x=0.5, y=0.98)
fig.layout.margin.t = 50
fig.show()
pmv.io.save_and_compress_svg(fig, "coordination-hist-single")


# %% Example with CrystalNN
for strategy in (CrystalNN(), VoronoiNN()):
    cls_name = type(strategy).__name__
    fig = pmv.coordination_hist(structures[key1], strategy=strategy)
    fig.layout.title = dict(
        text=f"Coordination Histogram ({cls_name}): {key1}", x=0.5, y=0.98
    )
    fig.layout.margin.t = 50
    fig.show()
    pmv.io.save_and_compress_svg(fig, f"coordination-hist-{cls_name.lower()}")


# %% Custom analyzer example
fig = pmv.coordination_hist(structures[key1], strategy=VoronoiNN())
fig.layout.margin.t = 50
fig.layout.title = dict(
    text=f"Coordination Histogram (VoronoiNN): {key1}", x=0.5, y=0.98
)
fig.show()
pmv.io.save_and_compress_svg(fig, "coordination-hist-voronoi")


# %% Multiple structures example
fig = pmv.coordination_hist({key1: structures[key1], key2: structures[key2]})
fig.layout.margin.t = 50
fig.layout.title = dict(
    text="Coordination Histogram: Multiple Structures", x=0.5, y=0.98
)
fig.show()
pmv.io.save_and_compress_svg(fig, "coordination-hist-multiple")


# %% By element example (now default, but explicitly specified for clarity)
fig = pmv.coordination_hist(structures[key1], split_mode=SplitMode.by_element)
fig.layout.margin.t = 50
fig.layout.title = dict(
    text=f"Coordination Histogram by Element: {key1}", x=0.5, y=0.98
)
fig.show()
pmv.io.save_and_compress_svg(fig, "coordination-hist-by-element")


# %% Multiple structures with by_element split
fig = pmv.coordination_hist(
    {key1: structures[key1], key2: structures[key2], key3: structures[key3]},
)
fig.layout.margin.t = 50
fig.layout.title = dict(
    text="Coordination Histogram by Element: Multiple Structures", x=0.5, y=0.98
)
fig.show()
pmv.io.save_and_compress_svg(fig, "coordination-hist-multiple-by-element")


# %% Multiple structures with by_structure split
fig = pmv.coordination_hist(
    {key1: structures[key1], key2: structures[key2], key3: structures[key3]},
    split_mode=SplitMode.by_structure,
)
fig.layout.margin.t = 50
fig.layout.title = dict(text="Coordination Histogram by Structure", x=0.5, y=0.98)
fig.show()
pmv.io.save_and_compress_svg(fig, "coordination-hist-by-structure")


# %% Multiple structures with by_structure_and_element split
fig = pmv.coordination_hist(
    {key1: structures[key1], key2: structures[key2], key3: structures[key3]},
    split_mode=SplitMode.by_structure_and_element,
)
fig.layout.margin.t = 60
fig.layout.title = dict(
    text="Coordination Histogram by Structure and Element", x=0.5, y=0.98
)
fig.show()
pmv.io.save_and_compress_svg(fig, "coordination-hist-by-structure-and-element")


# %% Multiple structures with by_element split and custom hover data
fig = pmv.coordination_hist(
    {key1: structures[key1], key2: structures[key2], key3: structures[key3]},
    hover_data=("oxidation_state", "wyckoff"),
)
fig.layout.margin.t = 50
fig.layout.title = dict(
    text="Coordination Histogram by Element: Multiple Structures (with custom hover)",
    x=0.5,
    y=0.98,
)
fig.show()
pmv.io.save_and_compress_svg(fig, "coordination-hist-multiple-by-element-custom-hover")


# %% Multiple structures with by_element split and custom element color scheme
custom_colors = {"Fe": "red", "O": "blue", "H": "green"}
fig = pmv.coordination_hist(
    {key1: structures[key1], key2: structures[key2], key3: structures[key3]},
    hover_data=("oxidation_state", "wyckoff"),
    element_color_scheme=custom_colors,
)
fig.layout.margin.t = 60
fig.layout.title = dict(
    text="Coordination Histogram by Element: Multiple Structures (with custom colors)",
    x=0.5,
    y=0.99,
)
fig.show()
pmv.io.save_and_compress_svg(fig, "coordination-hist-multiple-by-element-custom-colors")


# %% Example with split_mode=SplitMode.none
fig = pmv.coordination_hist(
    {key1: structures[key1], key2: structures[key2], key3: structures[key3]},
    split_mode=SplitMode.none,
)
fig.layout.margin.t = 50
fig.layout.title = dict(
    text="Coordination Histogram: All Structures Combined", x=0.5, y=0.98
)
fig.show()
pmv.io.save_and_compress_svg(fig, "coordination-hist-all-combined")


# %% Example with split_mode=SplitMode.none and side-by-side bars
fig = pmv.coordination_hist(
    {key1: structures[key1], key2: structures[key2], key3: structures[key3]},
    split_mode=SplitMode.none,
    bar_mode="group",
    bar_kwargs=dict(width=0.2),
)
fig.layout.margin.t = 50
fig.layout.title = dict(
    text="Coordination Histogram: All Structures Combined (Side-by-side)", x=0.5, y=0.98
)
fig.show()
pmv.io.save_and_compress_svg(fig, "coordination-hist-all-combined-side-by-side")


# %% Coordination vs Cutoff example for a single structure
fig = pmv.coordination_vs_cutoff_line(structures[key1])
fig.layout.title = dict(text=f"Coordination vs Cutoff: {key1}", x=0.5, y=0.98)
fig.layout.margin.t = 50
fig.show()
pmv.io.save_and_compress_svg(fig, "coordination-vs-cutoff-single")


# %% Coordination vs Cutoff example for multiple structures
fig = pmv.coordination_vs_cutoff_line(
    {key1: structures[key1], key2: structures[key2], key3: structures[key3]},
    cutoff_range=(1.0, 6.0),
    num_points=100,
)
fig.layout.title = dict(
    text="Coordination vs Cutoff: Multiple Structures", x=0.5, y=0.98
)
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
