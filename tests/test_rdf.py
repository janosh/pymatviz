from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest
from numpy.testing import assert_allclose
from pymatgen.core import Lattice, Structure

from pymatviz.rdf import calculate_rdf, element_pair_rdfs


@pytest.mark.parametrize(
    ("n_cols", "subplot_titles", "vertical_spacing"),
    [(1, None, 0), (3, ["title1", "title2", "title3"], 0.1)],
)
def test_element_pair_rdfs_basic(
    structures: list[Structure],
    n_cols: int,
    subplot_titles: list[str] | None,
    vertical_spacing: float,
) -> None:
    for structure in structures:
        subplot_kwargs = dict(
            cols=n_cols,
            subplot_titles=subplot_titles,
            vertical_spacing=vertical_spacing,
        )
        fig = element_pair_rdfs(structure, subplot_kwargs=subplot_kwargs)
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text is None
        assert fig.layout.showlegend is None
        assert fig.layout.yaxis.title.text == "g(r)"
        # check grid ref matches n_cols
        actual_rows = len(fig._grid_ref)
        actual_cols = len(fig._grid_ref[0])
        n_elem_pairs = len(structure.chemical_system_set) ** 2
        assert actual_cols == min(n_cols, n_elem_pairs)
        assert actual_rows == (len(fig.data) + n_cols - 1) // n_cols
        annotation_texts = [anno.text for anno in fig.layout.annotations]
        assert (
            annotation_texts == subplot_titles[: len(fig.data)]
            if subplot_titles
            else [""] * len(fig.data)
        )


def test_element_pair_rdfs_empty_structure() -> None:
    empty_struct = Structure(Lattice.cubic(1), [], [])
    for struct in (empty_struct, {"blank": empty_struct}):
        key = " blank" if isinstance(struct, dict) else ""
        with pytest.raises(ValueError, match=f"input structure{key} contains no sites"):
            element_pair_rdfs(struct)


def test_element_pair_rdfs_invalid_elements(structures: list[Structure]) -> None:
    with pytest.raises(
        ValueError,
        match="Elements .* in element_pairs not present in any structure",
    ):
        element_pair_rdfs(
            structures[0], element_pairs=[("Zn", "Zn")]
        )  # Assuming Zn is not in the structure


def test_element_pair_rdfs_invalid_structure() -> None:
    with pytest.raises(TypeError, match="Invalid input format for structures"):
        element_pair_rdfs("not a structure")


def test_element_pair_rdfs_conflicting_bins_and_bin_size(
    structures: list[Structure],
) -> None:
    with pytest.raises(
        ValueError, match="Cannot specify both n_bins=.* and bin_size=.*"
    ):
        element_pair_rdfs(structures, n_bins=100, bin_size=0.1)


@pytest.mark.parametrize(
    ("param", "values"),
    [("cutoff", (5, 10, 15)), ("bin_size", (0.05, 0.1, 0.2))],
)
def test_element_pair_rdfs_cutoff_and_bin_size(
    structures: list[Structure], param: str, values: tuple[float, ...]
) -> None:
    structure = structures[0]
    for value in values:
        fig = element_pair_rdfs(structure, **{param: value})  # type: ignore[arg-type]

        # Check that we have the correct number of traces (one for each element pair)
        n_elements = len({site.specie.symbol for site in structure})
        expected_traces = n_elements * (n_elements + 1) // 2
        assert (
            len(fig.data) == expected_traces
        ), f"Expected {expected_traces} traces, got {len(fig.data)}"

        for trace in fig.data:
            if param == "cutoff":
                # Check that the x-axis data doesn't exceed the cutoff
                assert np.all(
                    trace.x <= value
                ), f"X-axis data exceeds cutoff of {value}"
                # Check that the maximum x value is close to the cutoff
                assert max(trace.x) == pytest.approx(
                    value
                ), f"Maximum x value {max(trace.x):.4} not close to cutoff {value}"
            elif param == "bin_size":
                # Check that the number of bins is approximately correct
                default_cutoff = 15  # Assuming default cutoff is 10
                expected_bins = int(np.ceil(default_cutoff / value))
                assert (
                    abs(len(trace.x) - expected_bins) <= 1
                ), f"Expected around {expected_bins} bins, got {len(trace.x)}"


def test_element_pair_rdfs_subplot_layout(structures: list[Structure]) -> None:
    for structure in structures:
        fig = element_pair_rdfs(structure)
        n_elements = len({site.specie.symbol for site in structure})
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
    for structure in structures:
        fig1 = element_pair_rdfs(structure, cutoff=5, bin_size=0.1)
        fig2 = element_pair_rdfs(structure, cutoff=5, bin_size=0.1)
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
        else {structs.formula for structs in structs}
    )


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
    ref_line_kwargs = {"line_color": "red", "line_width": 2}
    fig = element_pair_rdfs(structures, reference_line=ref_line_kwargs)
    n_subplots = len(fig._grid_ref) * len(fig._grid_ref[0])
    assert (
        sum(
            shape.type == "line" and shape.line.color == "red"
            for shape in fig.layout.shapes
        )
        == n_subplots
    )


def test_element_pair_rdfs_cutoff_and_bins(structures: list[Structure]) -> None:
    cutoff, n_bins = 8.5, 88
    fig = element_pair_rdfs(structures, cutoff=cutoff, n_bins=n_bins)
    assert max(fig.data[0].x) == pytest.approx(cutoff)
    assert len(fig.data[0].x) == n_bins


def test_calculate_rdf(structures: list[Structure]) -> None:
    for structure in structures:
        elements = list({site.specie.symbol for site in structure})
        for el1 in elements:
            for el2 in elements:
                radii, rdf = calculate_rdf(structure, el1, el2, 10, 100)
                assert isinstance(radii, np.ndarray)
                assert isinstance(rdf, np.ndarray)
                assert len(radii) == len(rdf)
                assert np.all(rdf >= 0)


@pytest.mark.parametrize(
    ("composition", "n_atoms"),
    [(["Si"], 100), (["Si", "Ge"], 100), (["Al", "O"], 100), (["Fe", "Ni", "Cr"], 165)],
)
def test_calculate_rdf_normalization(composition: list[str], n_atoms: int) -> None:
    # Create large structure with random coordinates
    lattice = Lattice.cubic(30)
    elements = sum(([el] * n_atoms for el in composition), [])  # noqa: RUF017
    coords = np.random.default_rng(seed=0).uniform(size=(len(elements), 3))
    structure = Structure(lattice, elements, coords)

    # Calculate RDF for each element pair
    cutoff, n_bins = 12, 75
    for el1 in composition:
        for el2 in composition:
            radii, rdf = calculate_rdf(structure, el1, el2, cutoff, n_bins)

            # Check if RDF approaches 1 for large separations
            last_10_percent = int(0.9 * len(rdf))
            avg_last_10_percent = round(np.mean(rdf[last_10_percent:]), 4)
            assert 0.95 <= avg_last_10_percent <= 1.05, (
                f"RDF does not approach 1 for large separations in {el1}-{el2} pairs, "
                f"{avg_last_10_percent=}"
            )

            # Check if RDF starts from 0 at r=0
            assert (
                rdf[0] == 0
            ), f"{rdf[0]=} should start from 0 at r=0 for {el1}-{el2} pair"

            # Check there are no negative values in the RDF
            assert all(rdf >= 0), f"RDF contains negative values for {el1}-{el2} pair"

            # Check if the radii array is correct

            assert_allclose(
                radii,
                np.linspace(cutoff / n_bins, cutoff, n_bins),
                err_msg="Radii array is incorrect",
            )

            # Check if the RDF has the correct number of bins
            assert (
                len(rdf) == n_bins
            ), f"RDF should have {n_bins=}, got {len(rdf)} for {el1}-{el2} pair"


@pytest.mark.parametrize(
    "pbc",
    [(1, 1, 1), (1, 1, 0), (1, 0, 0), (0, 0, 0)],
)
def test_calculate_rdf_pbc_settings(pbc: tuple[int, int, int]) -> None:
    lattice = Lattice.cubic(5)
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
    structure = Structure(lattice, ["Si", "Si"], coords)

    cutoff, n_bins = 10, 100
    radii, rdf = calculate_rdf(
        structure,
        center_species="Si",
        neighbor_species="Si",
        cutoff=cutoff,
        n_bins=n_bins,
        pbc=pbc,
    )

    assert len(radii) == n_bins
    assert len(rdf) == n_bins
    assert np.all(rdf >= 0)
    assert rdf[0] == 0, f"RDF should start at 0 for PBC {pbc}"

    peak_index = int(4.33 / cutoff * n_bins)  # √3/2 * 5 ≈ 4.33
    if pbc == (1, 1, 1):
        assert rdf[peak_index] > 1, f"Expected peak at 4.33 for PBC {pbc}"
    elif pbc == (1, 1, 0):
        assert rdf[peak_index] > 0, f"Expected non-zero value at 4.33 for PBC {pbc}"


def test_calculate_rdf_pbc_consistency() -> None:
    lattice = Lattice.cubic(10)
    coords = np.random.default_rng(seed=0).uniform(size=(20, 3))
    structure = Structure(lattice, ["Si"] * 20, coords)

    cutoff, n_bins = 15, 150

    _radii_full, rdf_full = calculate_rdf(
        structure,
        center_species="Si",
        neighbor_species="Si",
        cutoff=cutoff,
        n_bins=n_bins,
        pbc=(True, True, True),
    )

    _radii_none, _rdf_none = calculate_rdf(
        structure,
        center_species="Si",
        neighbor_species="Si",
        cutoff=cutoff,
        n_bins=n_bins,
        pbc=(False, False, False),
    )

    assert np.sum(rdf_full > 0) > 0, "Full PBC should have non-zero values"


def test_calculate_rdf_different_species() -> None:
    lattice = Lattice.cubic(5)
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
    structure = Structure(lattice, ["Si", "Ge"], coords)

    cutoff, n_bins = 10, 100

    _radii_si_si, rdf_si_si = calculate_rdf(
        structure,
        center_species="Si",
        neighbor_species="Si",
        cutoff=cutoff,
        n_bins=n_bins,
    )
    _radii_si_ge, rdf_si_ge = calculate_rdf(
        structure,
        center_species="Si",
        neighbor_species="Ge",
        cutoff=cutoff,
        n_bins=n_bins,
    )
    _radii_ge_ge, rdf_ge_ge = calculate_rdf(
        structure,
        center_species="Ge",
        neighbor_species="Ge",
        cutoff=cutoff,
        n_bins=n_bins,
    )

    assert np.all(rdf_si_si == 0), "Si-Si RDF should be all zeros"
    assert np.all(rdf_ge_ge == 0), "Ge-Ge RDF should be all zeros"
    assert np.any(rdf_si_ge > 0), "Si-Ge RDF should have non-zero values"

    peak_index = int(4.33 / cutoff * n_bins)
    assert (
        rdf_si_ge[peak_index] > 0
    ), "Expected peak in Si-Ge RDF at sqrt(3)/2 * lattice constant"


@pytest.mark.parametrize(
    ("cutoff", "frac_coords"),
    [(4, [0.9, 0.9, 0.9]), (0.1, [0.1, 0.1, 0.1])],
)
def test_calculate_rdf_edge_cases(cutoff: float, frac_coords: list[float]) -> None:
    lattice = Lattice.cubic(5)

    # Test with a single atom
    single_atom = Structure(lattice, ["Si"], [[0, 0, 0]])
    _radii, rdf = calculate_rdf(
        single_atom,
        center_species="Si",
        neighbor_species="Si",
        cutoff=cutoff,
        n_bins=100,
    )
    assert np.all(rdf == 0), "RDF for a single atom should be all zeros"

    # Check RDF=0 everywhere for distant atoms (beyond cutoff)
    distant_atoms = Structure(lattice, ["Si", "Si"], [[0, 0, 0], frac_coords])
    _radii, rdf = calculate_rdf(
        distant_atoms,
        center_species="Si",
        neighbor_species="Si",
        cutoff=cutoff,
        n_bins=30,
        pbc=(0, 0, 0),
    )
    # get idx of first radial bin that is greater than 3
    assert np.all(rdf == 0), "RDF for distant atoms should be all zeros"
