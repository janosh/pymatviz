from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pymatgen.analysis.local_env import CrystalNN, NearNeighbors, VoronoiNN
from pymatgen.core import Lattice, Structure

from pymatviz.colors import ELEM_COLORS_JMOL
from pymatviz.coordination import coordination_hist, coordination_vs_cutoff_line
from pymatviz.coordination.helpers import CnSplitMode
from pymatviz.enums import ElemColorScheme


if TYPE_CHECKING:
    from collections.abc import Sequence


def test_coordination_hist_single_structure(structures: Sequence[Structure]) -> None:
    """Test coordination_hist with a single structure."""
    fig = coordination_hist(structures[0])
    assert len(fig.data) == len({site.specie.symbol for site in structures[0]})

    # Test y-axis range
    expected_y_max = max(max(trace.y) for trace in fig.data)  # get max CN count
    dev_fig = fig.full_figure_for_development(warn=False)
    y_min, y_max = dev_fig.layout.yaxis.range
    assert y_min == 0
    assert y_max == pytest.approx(expected_y_max, rel=0.1)

    # Test x-axis properties
    assert dev_fig.layout.xaxis.tick0 is not None
    assert dev_fig.layout.xaxis.dtick == 1
    assert dev_fig.layout.xaxis.range[0] < min(trace.x[0] for trace in dev_fig.data)

    # Test y-axis properties
    assert dev_fig.layout.yaxis.range[0] == 0
    # Y-axis title "Count" is added as an annotation
    assert any(anno.text == "Count" for anno in dev_fig.layout.annotations)


def test_coordination_hist_multiple_structures(structures: Sequence[Structure]) -> None:
    """Test coordination_hist with multiple structures."""
    struct_dict = {f"Structure_{idx}": struct for idx, struct in enumerate(structures)}
    fig = coordination_hist(struct_dict)
    expected_traces = sum(
        len({site.specie.symbol for site in struct}) for struct in structures
    )
    assert len(fig.data) == expected_traces


@pytest.mark.parametrize("split_mode", list(CnSplitMode))
def test_coordination_hist_split_modes(
    structures: Sequence[Structure], split_mode: CnSplitMode
) -> None:
    """Test coordination_hist with different split modes."""
    fig = coordination_hist(structures[0], split_mode=split_mode)

    if split_mode in (CnSplitMode.none, CnSplitMode.by_element):
        assert len(fig.data) == len({site.specie.symbol for site in structures[0]})
    elif split_mode == CnSplitMode.by_structure:
        assert len(fig.data) == 1
    elif split_mode == CnSplitMode.by_structure_and_element:
        assert len(fig.data) == len({site.specie.symbol for site in structures[0]})


@pytest.mark.parametrize(
    "strategy", [3.0, 6, VoronoiNN(), CrystalNN, CrystalNN(distance_cutoffs=(0.5, 2))]
)
def test_coordination_hist_custom_strategy(
    structures: Sequence[Structure],
    strategy: NearNeighbors | type[NearNeighbors] | float,
) -> None:
    """Test coordination_hist with a custom strategy."""
    fig = coordination_hist(structures[1], strategy=strategy)
    assert len(fig.data) == 3
    expected_max_x = {
        3.0: 9,
        6: 47,
        VoronoiNN(): 19,
        CrystalNN: 10,
        CrystalNN(distance_cutoffs=(0.5, 2)): 10,
    }[strategy]
    actual_max_x = max(max(trace.x) for trace in fig.data)
    assert actual_max_x == expected_max_x, f"{actual_max_x=} for {strategy=}"

    # Test with multiple structures
    fig_multi = coordination_hist(structures, strategy=strategy)
    assert fig_multi.data
    assert len(fig_multi.data) >= len(fig.data)

    # Test different split modes
    for split_mode in CnSplitMode:
        fig_split = coordination_hist(
            structures[:2], strategy=strategy, split_mode=split_mode
        )
        assert fig_split.data


def test_coordination_hist_bar_mode(structures: Sequence[Structure]) -> None:
    """Test coordination_hist with different bar modes."""
    fig_stack = coordination_hist(structures[0], bar_mode="stack")
    fig_group = coordination_hist(structures[0], bar_mode="group")
    assert fig_stack.layout.barmode == "stack"
    assert fig_group.layout.barmode == "group"


def test_coordination_hist_hover_data(structures: Sequence[Structure]) -> None:
    """Test coordination_hist with custom hover data."""
    structures[0].add_site_property("test_property", list(range(len(structures[0]))))
    fig = coordination_hist(structures[0], hover_data=["test_property"])
    assert "test_property" in fig.data[0].hovertext[0]


def test_coordination_hist_element_color_scheme(
    structures: Sequence[Structure],
) -> None:
    """Test coordination_hist with custom element color scheme."""
    elements = {site.specie.symbol for site in structures[0]}
    colors = ("red", "blue", "green", "yellow", "purple", "orange", "pink", "brown")
    custom_colors = dict(zip(elements, colors, strict=False))
    fig = coordination_hist(structures[0], element_color_scheme=custom_colors)
    for trace in fig.data:
        assert trace.marker.color == custom_colors[trace.name.split(" - ")[1]]


def test_coordination_hist_annotate_bars(structures: Sequence[Structure]) -> None:
    """Test coordination_hist with bar annotations."""
    fig = coordination_hist(structures[0], annotate_bars=True)
    elements = {site.specie.symbol for site in structures[0]} | {""}
    for trace in fig.data:
        assert {trace.text} <= elements, f"Invalid text: {trace.text}"


def test_coordination_hist_bar_kwargs(structures: Sequence[Structure]) -> None:
    """Test coordination_hist with custom bar kwargs."""
    bar_kwargs = {"opacity": 0.5, "width": 0.5}
    fig = coordination_hist(structures[0], bar_kwargs=bar_kwargs)
    for trace in fig.data:
        assert trace.opacity == 0.5
        assert trace.width == 0.5


def test_coordination_hist_invalid_input() -> None:
    """Test coordination_hist with invalid input."""
    with pytest.raises(TypeError):
        coordination_hist("invalid input")  # type: ignore[arg-type]


def test_coordination_hist_empty() -> None:
    """Test coordination_hist with an empty structure."""
    with pytest.raises(ValueError, match="Cannot plot empty set of structures"):
        coordination_hist(())


@pytest.mark.parametrize(
    "strategy",
    [
        (0, 3),  # int cutoff range
        (1.0, 5.0),  # float cutoff range
        # VoronoiNN(),  # NearNeighbors instance
        CrystalNN,  # NearNeighbors subclass
        CrystalNN(distance_cutoffs=(0.5, 2)),  # instance with custom params
    ],
)
def test_coordination_vs_cutoff_line(
    structures: Sequence[Structure],
    strategy: tuple[int | float, int | float] | NearNeighbors | type[NearNeighbors],
) -> None:
    """Test coordination_vs_cutoff_line function with different strategies."""
    # Test with a single structure
    fig = coordination_vs_cutoff_line(structures[0], strategy=strategy)
    assert len(fig.data) == len({site.specie.symbol for site in structures[0]})

    # Test with multiple structures
    fig_multi = coordination_vs_cutoff_line(structures[:2], strategy=strategy)
    assert fig_multi.data
    assert len(fig_multi.data) >= len(fig.data)

    # Test with custom number of points
    fig_custom_points = coordination_vs_cutoff_line(
        structures[0],
        strategy=strategy,
        num_points=100,
    )
    assert fig_custom_points.data
    assert len(fig_custom_points.data[0].x) == 100

    # Test with custom element color scheme
    custom_colors = {"Si": "red", "O": "blue"}
    fig_custom_colors = coordination_vs_cutoff_line(
        structures[0],
        strategy=strategy,
        element_color_scheme=custom_colors,
    )
    assert fig_custom_colors.data
    for trace in fig_custom_colors.data:
        element = trace.name.split(" (")[0]
        assert trace.line.color == custom_colors.get(
            element, ELEM_COLORS_JMOL.get(element)
        )

    # Test with built-in color schemes
    for color_scheme in ElemColorScheme:
        fig_color_scheme = coordination_vs_cutoff_line(
            structures[0],
            strategy=strategy,
            element_color_scheme=color_scheme,
        )
        assert fig_color_scheme.data

    # Test y-axis label
    assert fig.layout.yaxis.title.text == "Coordination Number"

    # Test with custom subplot_kwargs
    custom_subplot_kwargs = {
        "subplot_titles": ["Custom Title 1", "Custom Title 2"],
    }
    fig_custom_subplot = coordination_vs_cutoff_line(
        structures[:2],
        strategy=strategy,
        subplot_kwargs=custom_subplot_kwargs,
    )
    assert fig_custom_subplot.data
    assert fig_custom_subplot.layout.annotations[0].text == "Custom Title 1"
    assert fig_custom_subplot.layout.annotations[1].text == "Custom Title 2"


def test_coordination_vs_cutoff_line_invalid_input() -> None:
    """Test coordination_vs_cutoff_line with invalid input."""
    # Test empty sequences
    for inputs in ([], ()):
        with pytest.raises(ValueError, match="Cannot plot empty set of structures"):
            coordination_vs_cutoff_line(inputs)

    # Test invalid types
    for inputs in ("invalid input", None):
        with pytest.raises(
            TypeError,
            match="Input must be a pymatgen Structure, IStructure, Molecule, "
            "IMolecule, ASE Atoms, or PhonopyAtoms object",
        ):
            coordination_vs_cutoff_line(inputs)  # type: ignore[arg-type]


def test_coordination_vs_cutoff_line_invalid_strategy() -> None:
    """Test coordination_vs_cutoff_line with invalid strategy."""
    structure = Structure(Lattice.cubic(5), ["Si"], [[0, 0, 0]])
    with pytest.raises(TypeError, match="Invalid strategy="):
        coordination_vs_cutoff_line(structure, strategy="invalid")  # type: ignore[arg-type]


def test_coordination_hist_hover_text_formatting(
    structures: Sequence[Structure],
) -> None:
    """Test hover text formatting in coordination_hist."""
    # Add test property
    structures[0].add_site_property("test_prop", list(range(len(structures[0]))))

    # Test with single structure
    fig = coordination_hist(structures[0], hover_data=["test_prop"])
    hover_text = fig.data[0].hovertext[0]
    assert "Element:" in hover_text
    assert "Coordination number:" in hover_text
    assert "test_prop:" in hover_text

    # Test with multiple structures
    struct_dict = {"struct1": structures[0], "struct2": structures[1]}
    fig_multi = coordination_hist(struct_dict, hover_data=["test_prop"])
    hover_text_multi = fig_multi.data[0].hovertext[0]
    assert "Formula:" in hover_text_multi


def test_coordination_hist_subplot_layout(structures: Sequence[Structure]) -> None:
    """Test subplot layout in coordination_hist."""
    struct_dict = {f"s{idx}": struct for idx, struct in enumerate(structures[:3])}

    # Test by_structure layout
    fig = coordination_hist(struct_dict, split_mode=CnSplitMode.by_structure)
    # subplot titles + axis titles
    assert len(fig.layout.annotations) == len(struct_dict) + 2

    # Test by_element layout
    elements = {
        site.specie.symbol for struct in struct_dict.values() for site in struct
    }
    fig_elem = coordination_hist(struct_dict, split_mode=CnSplitMode.by_element)
    # Account for x and y axis titles which are also annotations
    assert len(fig_elem.layout.annotations) == len(elements) + 2


def test_coordination_hist_bar_customization(structures: Sequence[Structure]) -> None:
    """Test bar customization options in coordination_hist."""
    # Test bar width
    bar_kwargs = {"width": 0.5}
    fig = coordination_hist(structures[0], bar_kwargs=bar_kwargs)
    assert all(trace.width == 0.5 for trace in fig.data)

    # Test bar opacity
    bar_kwargs = {"opacity": 0.7}
    fig = coordination_hist(structures[0], bar_kwargs=bar_kwargs)
    assert all(trace.opacity == 0.7 for trace in fig.data)


def test_coordination_hist_color_schemes(structures: Sequence[Structure]) -> None:
    """Test different color schemes in coordination_hist."""
    # Test JMOL colors
    fig_jmol = coordination_hist(
        structures[0], element_color_scheme=ElemColorScheme.jmol
    )

    # Test VESTA colors
    fig_vesta = coordination_hist(
        structures[0], element_color_scheme=ElemColorScheme.vesta
    )

    # Colors should be different between schemes
    assert any(
        t1.marker.color != t2.marker.color
        for t1, t2 in zip(fig_jmol.data, fig_vesta.data, strict=True)
    )


def test_coordination_hist_invalid_elem_colors(structures: Sequence[Structure]) -> None:
    """Test invalid color scheme handling."""
    with pytest.raises(TypeError, match=r"Invalid.*element_color_scheme"):
        coordination_hist(structures[0], element_color_scheme="invalid")  # type: ignore[arg-type]


def test_coordination_hist_invalid_hover_data(structures: Sequence[Structure]) -> None:
    """Test invalid hover_data handling."""
    with pytest.raises(TypeError, match="Invalid hover_data"):
        coordination_hist(structures[0], hover_data=123)  # type: ignore[arg-type]


def test_coordination_hist_invalid_split_mode(structures: Sequence[Structure]) -> None:
    """Test invalid split_mode handling."""
    split_mode = "invalid_mode"
    with pytest.raises(ValueError, match=f"Invalid {split_mode=}"):
        coordination_hist(structures[0], split_mode=split_mode)


def test_coordination_hist_bar_annotations(structures: Sequence[Structure]) -> None:
    """Test bar annotation functionality."""
    # Test default annotation settings
    fig = coordination_hist(structures[0], annotate_bars=True)
    assert all(trace.text is not None for trace in fig.data)

    # Test custom annotation settings
    custom_annotations = {"size": 14, "color": "red"}
    fig = coordination_hist(structures[0], annotate_bars=custom_annotations)
    assert all(
        trace.textfont.size == 14 and trace.textfont.color == "red"
        for trace in fig.data
    )


def test_coordination_vs_cutoff_line_disordered_structure() -> None:
    """Test coordination_vs_cutoff_line with a disordered structure."""
    from pymatviz.structure import fe3co4_disordered

    # Test with a single disordered structure
    fig = coordination_vs_cutoff_line(fe3co4_disordered, strategy=(1, 5))
    assert len(fig.data) == 3

    # Disordered structure has one site with Fe:C (75:25) and one O site
    # For the disordered site, we should see both Fe and C contributions
    # The function uses specie.symbol which should handle disordered sites
    assert len(fig.data) >= 1, "Should have traces for disordered structure elements"

    # Check that traces have valid data
    for trace in fig.data:
        assert len(trace.x) == 50, "Default num_points=50"
        assert len(trace.y) == 50, "Should have 50 y-values"
        assert all(y_val >= 0 for y_val in trace.y), "CNs should be non-negative"
        assert trace.name, "Each trace should have a name"

    # Test with different strategies
    for strategy in [(2, 6), CrystalNN, VoronoiNN()]:
        fig_strat = coordination_vs_cutoff_line(fe3co4_disordered, strategy=strategy)
        assert fig_strat.data, f"Should work with {strategy=}"

    # Test with multiple disordered structures
    struct_dict = {
        "struct1": fe3co4_disordered,
        "struct2": fe3co4_disordered.copy(),
    }
    fig_multi = coordination_vs_cutoff_line(struct_dict, strategy=(1, 5))
    assert fig_multi.data
    assert len(fig_multi.data) >= len(fig.data)
