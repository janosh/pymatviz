import numpy as np
import plotly.graph_objects as go
import pytest
from pymatgen.core import Lattice, Structure

from pymatviz.rdf import calculate_rdf, element_pair_rdfs


def test_element_pair_rdfs_basic(structures: list[Structure]) -> None:
    for structure in structures:
        fig = element_pair_rdfs(structure)
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text is None
        assert fig.layout.showlegend is None
        assert fig.layout.yaxis.title.text == "g(r)"


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
