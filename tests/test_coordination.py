from collections.abc import Sequence

import pytest
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core import Structure

from pymatviz.coordination import SplitMode, coordination_hist


def test_coordination_hist_single_structure(structures: Sequence[Structure]) -> None:
    """Test coordination_hist with a single structure."""
    fig = coordination_hist(structures[0])
    assert fig.data
    assert len(fig.data) == len({site.specie.symbol for site in structures[0]})


def test_coordination_hist_multiple_structures(structures: Sequence[Structure]) -> None:
    """Test coordination_hist with multiple structures."""
    struct_dict = {f"Structure_{i}": struct for i, struct in enumerate(structures)}
    fig = coordination_hist(struct_dict)
    assert fig.data
    expected_traces = sum(
        len({site.specie.symbol for site in struct}) for struct in structures
    )
    assert len(fig.data) == expected_traces


@pytest.mark.parametrize("split_mode", list(SplitMode))
def test_coordination_hist_split_modes(
    structures: Sequence[Structure], split_mode: SplitMode
) -> None:
    """Test coordination_hist with different split modes."""
    fig = coordination_hist(structures[0], split_mode=split_mode)
    assert fig.data

    if split_mode in (SplitMode.none, SplitMode.by_element):
        assert len(fig.data) == len({site.specie.symbol for site in structures[0]})
    elif split_mode == SplitMode.by_structure:
        assert len(fig.data) == 1
    elif split_mode == SplitMode.by_structure_and_element:
        assert len(fig.data) == len({site.specie.symbol for site in structures[0]})


def test_coordination_hist_custom_analyzer(structures: Sequence[Structure]) -> None:
    """Test coordination_hist with a custom analyzer."""
    custom_analyzer = VoronoiNN()
    fig = coordination_hist(structures[0], analyzer=custom_analyzer)
    assert fig.data
    assert len(fig.data) == len({site.specie.symbol for site in structures[0]})


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
    assert fig.data
    assert "test_property" in fig.data[0].hovertext[0]


def test_coordination_hist_element_color_scheme(
    structures: Sequence[Structure],
) -> None:
    """Test coordination_hist with custom element color scheme."""
    elements = {site.specie.symbol for site in structures[0]}
    colors = ("red", "blue", "green", "yellow", "purple", "orange", "pink", "brown")
    custom_colors = dict(zip(elements, colors, strict=False))
    fig = coordination_hist(structures[0], element_color_scheme=custom_colors)
    assert fig.data
    for trace in fig.data:
        assert trace.marker.color == custom_colors[trace.name.split(" - ")[1]]


def test_coordination_hist_annotate_bars(structures: Sequence[Structure]) -> None:
    """Test coordination_hist with bar annotations."""
    fig = coordination_hist(structures[0], annotate_bars=True)
    assert fig.data
    elements = {site.specie.symbol for site in structures[0]} | {""}
    for trace in fig.data:
        assert {trace.text} <= elements, f"Invalid text: {trace.text}"


def test_coordination_hist_bar_kwargs(structures: Sequence[Structure]) -> None:
    """Test coordination_hist with custom bar kwargs."""
    bar_kwargs = {"opacity": 0.5, "width": 0.5}
    fig = coordination_hist(structures[0], bar_kwargs=bar_kwargs)
    assert fig.data
    for trace in fig.data:
        assert trace.opacity == 0.5
        assert trace.width == 0.5


def test_coordination_hist_y_axis_range(structures: Sequence[Structure]) -> None:
    """Test if y-axis range is 10% higher than the max count."""
    fig = coordination_hist(structures[0])
    assert fig.data
    max_count = max(max(trace.y) for trace in fig.data)
    expected_y_max = max_count * 1.1
    assert fig.layout.yaxis.range[1] == pytest.approx(expected_y_max, rel=1e-6)


def test_coordination_hist_invalid_input() -> None:
    """Test coordination_hist with invalid input."""
    with pytest.raises(TypeError):
        coordination_hist("invalid input")


def test_coordination_hist_empty() -> None:
    """Test coordination_hist with an empty structure."""
    with pytest.raises(TypeError, match="Invalid structures="):
        coordination_hist(())
