# %%
from glob import glob

from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import pymatviz as pmv
from pymatviz.coordination import SplitMode
from pymatviz.utils import TEST_FILES


pmv.set_plotly_template("pymatviz_white")


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
fig = pmv.coordination_hist(structures[key1])
fig.layout.title = dict(text=f"Coordination Histogram: {key1}", x=0.5, y=0.98)
fig.layout.margin.t = 50
fig.show()
pmv.io.save_and_compress_svg(fig, "coordination-hist-single")


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


# %% Custom analyzer example
custom_analyzer = VoronoiNN()
fig = pmv.coordination_hist(structures[key1], analyzer=custom_analyzer)
fig.layout.margin.t = 50
fig.layout.title = dict(
    text=f"Coordination Histogram (VoronoiNN): {key1}", x=0.5, y=0.98
)
fig.show()
pmv.io.save_and_compress_svg(fig, "coordination-hist-voronoi")


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
custom_colors = {
    "Fe": "#FF0000",
    "O": "#0000FF",
    "H": "#00FF00",
}  # Example custom colors
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
