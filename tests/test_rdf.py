import numpy as np
import plotly.graph_objects as go
import pytest
from numpy.testing import assert_allclose
from pymatgen.core import Lattice, Structure

from pymatviz.rdf import calculate_rdf, element_pair_rdfs


@pytest.mark.parametrize("n_cols", [1, 3])
def test_element_pair_rdfs_basic(structures: list[Structure], n_cols: int) -> None:
    for structure in structures:
        fig = element_pair_rdfs(structure, n_cols=n_cols)
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


def test_element_pair_rdfs_empty_structure() -> None:
    empty_structure = Structure(Lattice.cubic(1), [], [])
    with pytest.raises(ValueError, match="input structure contains no sites"):
        element_pair_rdfs(empty_structure)


def test_element_pair_rdfs_invalid_element_pairs(structures: list[Structure]) -> None:
    with pytest.raises(
        ValueError,
        match="Elements .* in element_pairs are not present in the structure",
    ):
        element_pair_rdfs(
            structures[0], element_pairs=[("Zn", "Zn")]
        )  # Assuming Zn is not in the structure


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
                ), f"Maximum x value {max(trace.x)} not close to cutoff {value}"
            elif param == "bin_size":
                # Check that the number of bins is approximately correct
                default_cutoff = 15  # Assuming default cutoff is 10.0
                expected_bins = int(np.ceil(default_cutoff / value))
                assert (
                    abs(len(trace.x) - expected_bins) <= 1
                ), f"Expected around {expected_bins} bins, got {len(trace.x)}"


def test_element_pair_rdfs_element_pairs(structures: list[Structure]) -> None:
    element_pairs = [("Si", "Si")]
    fig = element_pair_rdfs(structures[0], element_pairs=element_pairs)
    assert len(fig.data) == len(element_pairs)
    assert fig.data[0].name == "Si-Si"


def test_element_pair_rdfs_subplot_layout(structures: list[Structure]) -> None:
    for structure in structures:
        fig = element_pair_rdfs(structure)
        n_elements = len({site.specie.symbol for site in structure})
        expected_pairs = n_elements * (n_elements + 1) // 2
        assert len(fig.data) == expected_pairs
        assert all(isinstance(trace, go.Scatter) for trace in fig.data)


def test_calculate_rdf(structures: list[Structure]) -> None:
    for structure in structures:
        elements = list({site.specie.symbol for site in structure})
        for e1 in elements:
            for e2 in elements:
                radii, rdf = calculate_rdf(structure, e1, e2, 10.0, 100)
                assert isinstance(radii, np.ndarray)
                assert isinstance(rdf, np.ndarray)
                assert len(radii) == len(rdf)
                assert np.all(rdf >= 0)


def test_calculate_rdf_normalization() -> None:
    # Create large Silicon structure with random coordinates
    lattice = Lattice.cubic(30)
    n_atoms = 100
    coords = np.random.default_rng(seed=0).uniform(size=(n_atoms, 3))
    amorphous_si = Structure(lattice, ["Si"] * n_atoms, coords)

    # Calculate RDF with a large cutoff to see behavior at large separations
    cutoff, n_bins = 12, 500
    radii, rdf = calculate_rdf(amorphous_si, "Si", "Si", cutoff, n_bins)

    # Check if RDF approaches 1 for large separations
    # We'll check the average of the last 10% of the RDF
    last_10_percent = int(0.9 * len(rdf))
    avg_last_10_percent = float(np.mean(rdf[last_10_percent:]))
    assert (
        0.95 <= avg_last_10_percent <= 1.05
    ), f"RDF does not approach 1 for large separations, {avg_last_10_percent=}"

    # Check if RDF starts from 0 at r=0
    assert rdf[0] == 0, f"RDF does not start from 0 at r=0. First value: {rdf[0]}"

    # Check there are no negative values in the RDF
    assert np.all(rdf >= 0), "RDF contains negative values"

    # Check if the radii array is correct
    assert_allclose(
        radii,
        np.linspace(cutoff / n_bins, cutoff, n_bins),
        err_msg="Radii array is incorrect",
    )

    # Check if the RDF has the correct number of bins
    assert len(rdf) == n_bins, f"RDF should have {n_bins=}, got {len(rdf)}"


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
        if element_pairs
        else [
            (e1, e2)
            for e1 in structure.symbol_set
            for e2 in structure.symbol_set
            if e1 <= e2
        ]
    )
    assert len(fig.data) == len(expected_pairs)
    for trace, pair in zip(fig.data, expected_pairs, strict=True):
        assert trace.name == f"{pair[0]}-{pair[1]}"


def test_element_pair_rdfs_consistency(structures: list[Structure]) -> None:
    for structure in structures:
        fig1 = element_pair_rdfs(structure, cutoff=5.0, bin_size=0.1)
        fig2 = element_pair_rdfs(structure, cutoff=5.0, bin_size=0.1)
        for trace1, trace2 in zip(fig1.data, fig2.data, strict=True):
            assert np.allclose(trace1.x, trace2.x)
            assert np.allclose(trace1.y, trace2.y)
