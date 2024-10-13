from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pymatgen.core import Lattice, Structure

from pymatviz.rdf.helpers import calculate_rdf, normalize_structures


def test_normalize_structures() -> None:
    # Test with a single Structure
    single_structure = Structure(Lattice.cubic(5), ["Si"], [[0, 0, 0]])
    result = normalize_structures(single_structure)
    assert isinstance(result, dict)
    assert len(result) == 1
    assert "" in result
    assert isinstance(result[""], Structure)

    # Test with empty input
    assert normalize_structures([]) == {}

    # Test with a list of Structures
    structure_list = [
        Structure(Lattice.cubic(5), ["Si"], [[0, 0, 0]]),
        Structure(Lattice.cubic(5), ["Ge"], [[0, 0, 0]]),
    ]
    result = normalize_structures(structure_list)
    assert isinstance(result, dict)
    assert len(result) == 2
    assert all(isinstance(s, Structure) for s in result.values())
    assert set(result.keys()) == {"Si1", "Ge1"}

    # Test with a dictionary of Structures
    structure_dict = {
        "silicon": Structure(Lattice.cubic(5), ["Si"], [[0, 0, 0]]),
        "germanium": Structure(Lattice.cubic(5), ["Ge"], [[0, 0, 0]]),
    }
    result = normalize_structures(structure_dict)
    assert result == structure_dict

    # Test with invalid input
    with pytest.raises(TypeError, match="Invalid input format for structures="):
        normalize_structures("invalid input")

    # Test with mixed valid and invalid inputs in a list
    with pytest.raises(TypeError, match="Invalid input format for structures="):
        normalize_structures([single_structure, "invalid"])


def test_calculate_rdf(structures: list[Structure]) -> None:
    for struct in structures:
        elements = list({site.specie.symbol for site in struct})
        for el1 in elements:
            for el2 in elements:
                radii, rdf = calculate_rdf(struct, el1, el2, 10, 100)
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
