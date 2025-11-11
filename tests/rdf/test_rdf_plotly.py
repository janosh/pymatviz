from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
import pytest
from numpy.testing import assert_allclose
from pymatgen.core import Lattice, Structure

from pymatviz.rdf.figures import element_pair_rdfs, full_rdf
from tests.conftest import SI_ATOMS, SI_STRUCTS


@pytest.mark.parametrize(
    ("n_cols", "subplot_titles_input", "vertical_spacing", "structs", "expected"),
    [
        (  # Case 1: 1 pmg Structure in list, no titles
            1,
            None,
            0,
            [SI_STRUCTS[0]],
            dict(legend_visible=False, subplot_grid=(1, 1), annotations=[]),
        ),
        (  # Case 2: 2 pmg Structures in list, 3 titles
            3,
            ["t1", "t2", "t3"],
            0.1,
            SI_STRUCTS,
            dict(
                legend_visible=True, subplot_grid=(2, 3), annotations=["t1", "t2", "t3"]
            ),
        ),
        (  # Case 3: 1 pmg Structure, 1 ASE Atoms in list, 3 titles
            3,
            ["t1", "t2", "t3"],
            0.1,
            [SI_STRUCTS[0], SI_ATOMS[1]],
            dict(
                legend_visible=True, subplot_grid=(2, 3), annotations=["t1", "t2", "t3"]
            ),
        ),
        (  # Case 4: 1 pmg Structure, 1 ASE Atoms in dict, 3 titles
            3,
            ["t1", "t2", "t3"],
            0.1,
            {"A": SI_STRUCTS[0], "B": SI_ATOMS[1]},
            dict(
                legend_visible=True, subplot_grid=(2, 3), annotations=["t1", "t2", "t3"]
            ),
        ),
        (  # Case 5: 2 ASE Atoms in dict, 3 titles
            3,
            ["t1", "t2", "t3"],
            0.1,
            {"A": SI_ATOMS[0], "B": SI_ATOMS[1]},
            dict(
                legend_visible=True, subplot_grid=(2, 3), annotations=["t1", "t2", "t3"]
            ),
        ),
    ],
)
def test_element_pair_rdfs_layout_and_annotations(
    structs: Any,
    n_cols: int,
    subplot_titles_input: list[str] | None,
    vertical_spacing: float,
    expected: dict[str, Any],
) -> None:
    """Test element_pair_rdfs for legend, grid layout, annotations, and subplot titles
    with various inputs.
    """
    subplot_kwargs = dict(
        cols=n_cols,
        subplot_titles=subplot_titles_input,
        vertical_spacing=vertical_spacing,
    )
    fig = element_pair_rdfs(structs, subplot_kwargs=subplot_kwargs)

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text is None  # Assuming no main title is set
    assert fig.layout.yaxis.title.text == "g(r)"  # Common y-axis title

    # Assert legend visibility
    assert fig.layout.showlegend == expected["legend_visible"]

    # Assert grid dimensions
    actual_fig_rows = len(fig._grid_ref) if fig._grid_ref else 0
    actual_fig_cols = len(fig._grid_ref[0]) if fig._grid_ref and fig._grid_ref[0] else 0

    assert actual_fig_cols == expected["subplot_grid"][1]
    assert actual_fig_rows == expected["subplot_grid"][0]

    # Assert annotations (subplot titles)
    annotations = fig.layout.annotations
    current_annotation_texts = [anno.text for anno in annotations]

    assert len(current_annotation_texts) == len(expected["annotations"])
    assert current_annotation_texts == expected["annotations"]


def test_element_pair_rdfs_empty_structure() -> None:
    empty_struct = Structure(Lattice.cubic(1), [], [])
    for struct in (empty_struct, {"blank": empty_struct}):
        key = " blank" if isinstance(struct, dict) else ""
        with pytest.raises(ValueError, match=f"input structure{key} contains no sites"):
            element_pair_rdfs(struct)


def test_element_pair_rdfs_invalid_elements(structures: list[Structure]) -> None:
    err_msg = "Elements .* in element_pairs not present in any structure"
    with pytest.raises(ValueError, match=err_msg):
        # Assuming Zn is not in the structure
        element_pair_rdfs(structures[0], element_pairs=[("Zn", "Zn")])


def test_element_pair_rdfs_invalid_structure() -> None:
    err_msg = "Input must be a pymatgen Structure, IStructure, Molecule"
    with pytest.raises(TypeError, match=err_msg):
        element_pair_rdfs("not a structure")


def test_element_pair_rdfs_conflicting_bins_and_bin_size(
    structures: list[Structure],
) -> None:
    err_msg = "Cannot specify both n_bins=.* and bin_size=.*"
    with pytest.raises(ValueError, match=err_msg):
        element_pair_rdfs(structures, n_bins=100, bin_size=0.1)


@pytest.mark.parametrize(
    ("param", "values"),
    [("cutoff", (5, 10, 15, -1.5, -2)), ("bin_size", (0.05, 0.1, 0.2))],
)
def test_element_pair_rdfs_cutoff_and_bin_size(
    structures: list[Structure], param: str, values: tuple[float, ...]
) -> None:
    struct = structures[0]
    for value in values:
        fig = element_pair_rdfs(struct, **{param: value})  # type: ignore[arg-type]

        # Check that we have the correct number of traces (one for each element pair)
        n_elements = len({site.specie.symbol for site in struct})
        expected_traces = n_elements * (n_elements + 1) // 2
        assert len(fig.data) == expected_traces, (
            f"Expected {expected_traces} traces, got {len(fig.data)}"
        )

        max_cell_len = max(struct.lattice.abc)
        for trace in fig.data:
            if param == "cutoff":
                expected_cutoff = abs(value) * max_cell_len if value < 0 else value
                # Check that the x-axis data doesn't exceed the cutoff
                assert np.all(trace.x <= expected_cutoff), (
                    f"X-axis data exceeds cutoff of {expected_cutoff}"
                )
                # Check that the maximum x value is close to the cutoff
                assert max(trace.x) == pytest.approx(expected_cutoff), (
                    f"Maximum x={max(trace.x):.4} not close to cutoff {expected_cutoff}"
                )
            elif param == "bin_size":
                # When bin_size is specified but cutoff is None,
                # the default cutoff is 2 * max_cell_len
                default_cutoff = 2 * max_cell_len
                # Check that the number of bins is approximately correct
                expected_bins = int(default_cutoff / value)
                assert 0.85 <= expected_bins / len(trace.x) <= 1, (
                    f"{expected_bins=}, got {len(trace.x)}"
                )


def test_element_pair_rdfs_subplot_layout(structures: list[Structure]) -> None:
    for struct in structures:
        fig = element_pair_rdfs(struct)
        n_elements = len({site.specie.symbol for site in struct})
        expected_pairs = n_elements * (n_elements + 1) // 2
        assert len(fig.data) == expected_pairs
        assert all(isinstance(trace, go.Scatter) for trace in fig.data)


@pytest.mark.parametrize(
    "element_pairs", [[("Si", "Si")], [("Si", "Ru"), ("Pr", "Pr")], None]
)
def test_element_pair_rdfs_custom_element_pairs(
    structures: list[Structure], element_pairs: list[tuple[str, str]] | None
) -> None:
    structure = structures[1]  # Use the structure with Si, Ru, and Pr
    fig = element_pair_rdfs(structure, element_pairs=element_pairs)
    expected_pairs = sorted(
        element_pairs
        or [
            (el1, el2)
            for el1 in structure.symbol_set
            for el2 in structure.symbol_set
            if el1 <= el2
        ]
    )
    assert len(fig.data) == len(expected_pairs)
    # test subplot titles (in fig.layout.annotations) match element pairs
    assert [anno.text for anno in fig.layout.annotations] == [
        f"{pair[0]}-{pair[1]}" for pair in expected_pairs
    ]


def test_element_pair_rdfs_consistency(structures: list[Structure]) -> None:
    for struct in structures:
        fig1 = element_pair_rdfs(struct, cutoff=5, bin_size=0.1)
        fig2 = element_pair_rdfs(struct, cutoff=5, bin_size=0.1)
        for trace1, trace2 in zip(fig1.data, fig2.data, strict=True):
            assert_allclose(trace1.x, trace2.x)
            assert_allclose(trace1.y, trace2.y)


@pytest.mark.parametrize("structs_type", ["dict", "list"])
def test_element_pair_rdfs_list_dict_of_structures(
    structures: list[Structure], structs_type: str
) -> None:
    structs = (
        {f"{struct} {idx}": struct for idx, struct in enumerate(structures)}
        if structs_type == "dict"
        else structures
    )
    fig = element_pair_rdfs(structs)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 12
    labels = {trace.name for trace in fig.data}
    assert len(labels) == len(structs)
    assert labels == (
        set(structs)
        if isinstance(structs, dict)
        else {f"{idx + 1} {struct.formula}" for idx, struct in enumerate(structs)}
    )
    # Check that legend is shown for multiple structures
    assert fig.layout.showlegend is True
    assert sum(trace.showlegend for trace in fig.data) == len(structs)


def test_element_pair_rdfs_custom_colors_and_styles(
    structures: list[Structure],
) -> None:
    colors = ["red", "blue"]
    line_styles = ["solid", "dash"]
    fig = element_pair_rdfs(structures, colors=colors, line_styles=line_styles)
    assert fig.data[0].line.color == colors[0]
    assert fig.data[1].line.color == colors[1]
    assert fig.data[0].line.dash == line_styles[0]
    assert fig.data[1].line.dash == line_styles[1]


def test_element_pair_rdfs_reference_line(structures: list[Structure]) -> None:
    ref_line_kwargs = {"line_color": "teal", "line_width": 2}
    fig = element_pair_rdfs(structures, reference_line=ref_line_kwargs)
    assert fig._grid_ref is not None
    n_subplots = len(fig._grid_ref) * len(fig._grid_ref[0])
    assert (
        sum(
            shape.type == "line" and shape.line.color == "teal"
            for shape in fig.layout.shapes
        )
        == n_subplots
    )


def test_element_pair_rdfs_cutoff_and_bins(structures: list[Structure]) -> None:
    # Test positive cutoff
    cutoff, n_bins = 8.5, 88
    fig = element_pair_rdfs(structures, cutoff=cutoff, n_bins=n_bins)
    assert max(fig.data[0].x) == pytest.approx(cutoff)
    assert len(fig.data[0].x) == n_bins

    # Test negative cutoff
    struct = structures[0]
    max_cell_len = max(struct.lattice.abc)
    neg_cutoff = -1.5
    expected_cutoff = abs(neg_cutoff) * max_cell_len
    fig = element_pair_rdfs(struct, cutoff=neg_cutoff, n_bins=n_bins)
    assert max(fig.data[0].x) == pytest.approx(expected_cutoff)
    assert len(fig.data[0].x) == n_bins

    # Test None cutoff (should default to 2 * max_cell_len)
    fig = element_pair_rdfs(struct, cutoff=None, n_bins=n_bins)
    assert max(fig.data[0].x) == pytest.approx(2 * max_cell_len)
    assert len(fig.data[0].x) == n_bins

    large_struct = struct.make_supercell(5, in_place=False)
    fig = element_pair_rdfs(large_struct)
    # Default cutoff should be min(15, 2 * max_cell_len)
    assert max(fig.data[0].x) == pytest.approx(15)


def test_full_rdf_basic(structures: list[Structure]) -> None:
    for struct in structures:
        fig = full_rdf(struct)
        assert isinstance(fig, go.Figure)
        assert fig.layout.xaxis.title.text == "r [Å]"
        assert fig.layout.yaxis.title.text == "g(r)"
        assert len(fig.data) == 1
        assert fig.data[0].name == struct.formula
        assert fig.layout.title.text is None
        assert fig.layout.showlegend is False or fig.layout.showlegend is None
        assert not fig.data[0].showlegend


def test_full_rdf_empty_structure() -> None:
    empty_struct = Structure(Lattice.cubic(1), [], [])
    for struct in (empty_struct, {"blank": empty_struct}):
        key = " blank" if isinstance(struct, dict) else ""
        with pytest.raises(ValueError, match=f"input structure{key} contains no sites"):
            full_rdf(struct)


def test_full_rdf_invalid_structure() -> None:
    err_msg = "Input must be a pymatgen Structure, IStructure, Molecule"
    with pytest.raises(TypeError, match=err_msg):
        full_rdf("not a structure")


def test_full_rdf_conflicting_bins_and_bin_size(structures: list[Structure]) -> None:
    with pytest.raises(
        ValueError, match=r"Cannot specify both n_bins=.* and bin_size=.*"
    ):
        full_rdf(structures, n_bins=100, bin_size=0.1)


@pytest.mark.parametrize(
    ("param", "values"),
    [("cutoff", (5, 10, 15)), ("bin_size", (0.05, 0.1, 0.2))],
)
def test_full_rdf_cutoff_and_bin_size(
    structures: list[Structure], param: str, values: tuple[float, ...]
) -> None:
    structure = structures[0]
    for value in values:
        fig = full_rdf(structure, **{param: value})  # type: ignore[arg-type]

        assert len(fig.data) == 1
        trace = fig.data[0]

        if param == "cutoff":
            assert np.all(trace.x <= value), f"X-axis data exceeds cutoff of {value}"
            assert max(trace.x) == pytest.approx(value), (
                f"Maximum x value {max(trace.x):.4} not close to cutoff {value}"
            )
        elif param == "bin_size":
            default_cutoff = 15
            expected_bins = int(np.ceil(default_cutoff / value))
            assert abs(len(trace.x) - expected_bins) <= 1, (
                f"Expected around {expected_bins} bins, got {len(trace.x)}"
            )


def test_full_rdf_consistency(structures: list[Structure]) -> None:
    for struct in structures:
        fig1 = full_rdf(struct, cutoff=5, bin_size=0.1)
        fig2 = full_rdf(struct, cutoff=5, bin_size=0.1)
        assert_allclose(fig1.data[0].x, fig2.data[0].x)
        assert_allclose(fig1.data[0].y, fig2.data[0].y)
        assert len(fig1.data[0].x) == len(fig2.data[0].x)

        fig3 = full_rdf(struct, cutoff=6, bin_size=0.2)
        assert 30 == len(fig3.data[0].x) < len(fig1.data[0].x) == 50
        assert 30 == len(fig3.data[0].y) < len(fig1.data[0].y) == 50


@pytest.mark.parametrize("structs_type", ["dict", "list"])
def test_full_rdf_list_dict_of_structures(
    structures: list[Structure], structs_type: str
) -> None:
    structs_dict_or_list = (
        {f"{struct} {idx}": struct for idx, struct in enumerate(structures)}
        if structs_type == "dict"
        else structures
    )
    fig = full_rdf(structs_dict_or_list)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == len(structs_dict_or_list)
    labels = {trace.name for trace in fig.data}
    assert len(labels) == len(structs_dict_or_list)
    assert labels == (
        set(structs_dict_or_list)
        if isinstance(structs_dict_or_list, dict)
        else {
            f"{idx + 1} {struct.formula}"
            for idx, struct in enumerate(structs_dict_or_list)
        }
    )


def test_full_rdf_custom_colors_and_styles(structures: list[Structure]) -> None:
    colors = ["red", "blue", "green"]
    line_styles = ["solid", "dash", "dot"]
    fig = full_rdf(structures, colors=colors, line_styles=line_styles)
    for idx, trace in enumerate(fig.data):
        assert trace.line.color == colors[idx % len(colors)]
        assert trace.line.dash == line_styles[idx % len(line_styles)]


def test_full_rdf_reference_line(structures: list[Structure]) -> None:
    ref_line_kwargs = {"line_color": "red", "line_width": 2}
    fig = full_rdf(structures, reference_line=ref_line_kwargs)
    n_ref_lines = sum(
        shape.type == "line" and shape.line.color == "red"
        for shape in fig.layout.shapes
    )
    assert n_ref_lines == 1


def test_full_rdf_legend_position(structures: list[Structure]) -> None:
    # Test with a single structure
    fig_single = full_rdf(structures[0])
    assert fig_single.layout.legend == go.layout.Legend()
    assert len(fig_single.data) == 1

    # Test with multiple structures
    fig_multiple = full_rdf(structures)
    assert fig_multiple.layout.legend.orientation == "h"
    assert fig_multiple.layout.legend.yanchor == "bottom"
    assert fig_multiple.layout.legend.y == 1.02
    assert fig_multiple.layout.legend.xanchor == "center"
    assert fig_multiple.layout.legend.x == 0.5
    assert len(fig_multiple.data) == len(structures)
    for idx, trace in enumerate(fig_multiple.data):
        assert trace.name == f"{idx + 1} {structures[idx].formula}"
