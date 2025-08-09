from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pymatgen.core import IStructure, Lattice, Structure
from pymatgen.core.composition import Composition

from pymatviz.rdf.helpers import calculate_rdf
from tests.conftest import SI_ATOMS, SI_STRUCTS


if TYPE_CHECKING:
    from typing import Any, Literal


def check_basic_rdf_properties(
    radii: np.ndarray, rdf: np.ndarray, n_bins: int, name: str = ""
) -> None:
    """Check basic properties that all RDFs should satisfy."""
    suffix = f" for {name}" if name else ""

    # Check array types and shapes
    assert isinstance(radii, np.ndarray), f"{radii=} should be a numpy array{suffix}"
    assert isinstance(rdf, np.ndarray), f"{rdf=} should be a numpy array{suffix}"
    assert len(radii) == len(rdf) == n_bins, (
        f"{rdf=} length should match n_bins{suffix}"
    )

    # Check RDF values
    assert np.all(rdf >= 0), f"{rdf=} should have non-negative values{suffix}"
    assert rdf[0] == 0, f"{rdf=} should start at 0{suffix}"


@pytest.mark.parametrize(
    ("structure_name", "structure"),
    [
        ("pymatgen_structure", SI_STRUCTS[0]),
        ("istructure", IStructure.from_sites(list(SI_STRUCTS[0].sites))),
        ("ase_atoms", SI_ATOMS[0]),
    ],
)
def test_calculate_rdf(structure_name: str, structure: Any) -> None:
    """Test basic RDF calculation for various structure types."""
    # Calculate RDF for the structure
    radii, rdf = calculate_rdf(structure, cutoff=10, n_bins=100)
    check_basic_rdf_properties(radii, rdf, 100, structure_name)

    # Get unique elements in the structure
    if hasattr(structure, "sites"):
        # For pymatgen Structure/IStructure
        elements = list({site.specie.symbol for site in structure})
    else:
        # For ASE Atoms
        from pymatgen.io.ase import AseAtomsAdaptor

        temp_struct = AseAtomsAdaptor.get_structure(structure)
        elements = list({site.specie.symbol for site in temp_struct})

    # Test partial RDFs for element pairs
    for el1 in elements:
        for el2 in elements:
            radii_partial, rdf_partial = calculate_rdf(structure, el1, el2, 10, 100)
            check_basic_rdf_properties(
                radii_partial, rdf_partial, 100, f"{structure_name}_{el1}_{el2}"
            )


@pytest.mark.parametrize(
    ("composition", "n_atoms"),
    [(["Si"], 100), (["Si", "Ge"], 100), (["Al", "O"], 100), (["Fe", "Ni", "Cr"], 165)],
)
def test_calculate_rdf_normalization(composition: list[str], n_atoms: int) -> None:
    """Test RDF normalization for different compositions."""
    # Create large structure with random coordinates
    elements = sum(([el] * n_atoms for el in composition), [])  # noqa: RUF017
    coords = np.random.default_rng(seed=0).uniform(size=(len(elements), 3))
    structure = Structure(Lattice.cubic(30), elements, coords)

    # Calculate RDF for each element pair
    cutoff, n_bins = 12, 75
    for el1 in composition:
        for el2 in composition:
            radii, rdf = calculate_rdf(structure, el1, el2, cutoff, n_bins)
            check_basic_rdf_properties(radii, rdf, n_bins, f"{el1}-{el2} pair")

            # Check if RDF approaches 1 for large separations
            last_10_percent = int(0.9 * len(rdf))
            avg_last_10_percent = round(np.mean(rdf[last_10_percent:]), 4)
            assert 0.95 <= avg_last_10_percent <= 1.05, (
                f"RDF does not approach 1 for large separations in {el1}-{el2} pairs, "
                f"{avg_last_10_percent=}"
            )
            err_msg = "Radii array is incorrect"
            assert_allclose(
                radii, np.linspace(cutoff / n_bins, cutoff, n_bins), err_msg=err_msg
            )


@pytest.mark.parametrize(
    ("pbc", "expected_peak"),
    [
        ((1, 1, 1), (4.33, 1.0)),  # (peak_position, min_height)
        ((1, 1, 0), (4.33, 0.0)),  # Just check for non-zero value
        ((1, 0, 0), None),  # No specific peak check
        ((0, 0, 0), None),  # No specific peak check
    ],
)
def test_calculate_rdf_pbc_settings(
    pbc: tuple[Literal[0, 1], Literal[0, 1], Literal[0, 1]],
    expected_peak: tuple[float, float] | None,
) -> None:
    """Test RDF calculation with different PBC settings."""
    structure = Structure(Lattice.cubic(5), ["Si"] * 2, [[0] * 3, [0.5] * 3])
    cutoff, n_bins = 10, 100

    radii, rdf = calculate_rdf(
        structure,
        center_species="Si",
        neighbor_species="Si",
        cutoff=cutoff,
        n_bins=n_bins,
        pbc=pbc,
    )

    check_basic_rdf_properties(radii, rdf, n_bins, f"PBC {pbc}")

    # Check for expected peak if specified
    if expected_peak:
        peak_pos, min_height = expected_peak
        peak_index = int(peak_pos / cutoff * n_bins)
        assert rdf[peak_index] > min_height, (
            f"Expected {min_height=} at {peak_pos=} for {pbc=}"
        )


def test_calculate_rdf_pbc_consistency() -> None:
    """Test consistency of RDF calculation with different PBC settings."""
    coords = np.random.default_rng(seed=0).uniform(size=(20, 3))
    structure = Structure(Lattice.cubic(10), ["Si"] * 20, coords)
    cutoff, n_bins = 15, 150

    (_radii_full_pbc, rdf_full_pbc), (_radii_no_pbc, rdf_no_pbc) = (
        calculate_rdf(
            structure,
            center_species="Si",
            neighbor_species="Si",
            cutoff=cutoff,
            n_bins=n_bins,
            pbc=pbc,
        )
        for pbc in ((1, 1, 1), (0, 0, 0))
    )

    assert np.sum(rdf_full_pbc > 0) > 0, "Full PBC should have non-zero values"
    assert np.sum(rdf_no_pbc > 0) < np.sum(rdf_full_pbc > 0), (
        "No PBC should have no non-zero values"
    )
    assert np.sum(rdf_no_pbc) == pytest.approx(30.55075158)
    assert np.sum(rdf_full_pbc) == pytest.approx(139.1986917)


def test_calculate_rdf_different_species() -> None:
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
    structure = Structure(Lattice.cubic(5), ["Si", "Ge"], coords)

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
    assert rdf_si_ge[peak_index] > 0, (
        "Expected peak in Si-Ge RDF at sqrt(3)/2 * lattice constant"
    )


@pytest.mark.parametrize(
    ("cutoff", "frac_coords"),
    [(4, [0.9, 0.9, 0.9]), (0.1, [0.1, 0.1, 0.1])],
)
def test_calculate_rdf_edge_cases(cutoff: float, frac_coords: list[float]) -> None:
    # Test with a single atom
    single_atom = Structure(Lattice.cubic(5), ["Si"], [[0, 0, 0]])
    _radii, rdf = calculate_rdf(
        single_atom,
        center_species="Si",
        neighbor_species="Si",
        cutoff=cutoff,
        n_bins=100,
    )
    assert np.all(rdf == 0), f"{rdf=} should be all zeros"

    # Check RDF=0 everywhere for distant atoms (beyond cutoff)
    distant_atoms = Structure(Lattice.cubic(5), ["Si", "Si"], [[0, 0, 0], frac_coords])
    _radii, rdf = calculate_rdf(
        distant_atoms,
        center_species="Si",
        neighbor_species="Si",
        cutoff=cutoff,
        n_bins=30,
        pbc=(0, 0, 0),
    )
    # get idx of first radial bin that is greater than 3
    assert np.all(rdf == 0), f"{rdf=} should be all zeros"


def test_calculate_rdf_disordered_structure(fe3co4_disordered: Structure) -> None:
    """Test RDF calculation for structures with disordered sites."""
    cutoff, n_bins = 10, 100
    radii, rdf = calculate_rdf(fe3co4_disordered, cutoff=cutoff, n_bins=n_bins)
    check_basic_rdf_properties(radii, rdf, n_bins, "disordered")

    # Check RDF shape properties
    peak_idx = 43
    assert np.argmax(rdf) == peak_idx
    assert rdf[peak_idx] == pytest.approx(41.10406)
    assert np.std(rdf) == pytest.approx(5.322650)

    # Test high-entropy alloy structure
    hea_structure = Structure(
        lattice=Lattice.cubic(a_len := 3.59),
        species=[Composition({"Co": 0.2, "Cr": 0.2, "Fe": 0.2, "Ni": 0.2, "Mn": 0.2})]
        * 4,
        coords=[(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
    )

    cutoff, n_bins = 10, 100
    radii, rdf = calculate_rdf(hea_structure, cutoff=cutoff, n_bins=n_bins)

    # Basic checks
    check_basic_rdf_properties(radii, rdf, n_bins, "HEA")

    # Check for expected FCC nearest neighbor peak
    fcc_nn_dist = a_len / np.sqrt(2)  # a/√2 ≈ 2.54Å
    fcc_nn_idx = int(fcc_nn_dist / cutoff * n_bins)
    peak_idx, peak_height = 25, 21.786465
    assert rdf[fcc_nn_idx] == pytest.approx(peak_height)
    assert np.argmax(rdf) == peak_idx
    assert np.std(rdf) == pytest.approx(3.305442597)


@pytest.mark.parametrize(
    ("param_name", "param_value", "cutoff", "n_bins", "expected_property"),
    [
        # Small cutoff test
        ("cutoff", 0.1, 0.1, 10, "all_zeros"),
        # Large bins test
        ("n_bins", 1000, 10, 1000, "has_nonzero"),
        # Reasonable bins test
        ("n_bins", 20, 5, 20, "has_variation"),
    ],
)
def test_calculate_rdf_extreme_parameters(
    param_name: str,
    param_value: float,
    cutoff: float,
    n_bins: int,
    expected_property: str,
) -> None:
    """Test RDF calculation with extreme parameters."""
    # Create a simple structure with known distances
    structure = Structure(
        Lattice.cubic(5),
        ["Si", "Si", "Si", "Si"],
        [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
    )

    radii, rdf = calculate_rdf(structure, cutoff=cutoff, n_bins=n_bins)

    # Check basic properties
    check_basic_rdf_properties(radii, rdf, n_bins, f"{param_name}={param_value}")

    # Check expected properties based on the test case
    if expected_property == "all_zeros":
        assert np.all(rdf == 0), f"Expected all zeros for {param_name}={param_value}"
    elif expected_property == "has_nonzero":
        assert np.any(rdf > 0), (
            f"Expected some non-zero values for {param_name}={param_value}"
        )
    elif expected_property == "has_variation":
        assert np.std(rdf) > 0, (
            f"Expected variation in RDF for {param_name}={param_value}"
        )


@pytest.mark.parametrize(
    ("center", "neighbor", "expect_zeros"),
    [
        ("Fe", "Si", True),  # non-existent center
        ("Si", "Fe", True),  # non-existent neighbor
        ("Fe", "Cu", True),  # both non-existent
        ("Si", "Si", False),  # both existing, same
        ("Si", "O", False),  # both existing, different
    ],
)
def test_calculate_rdf_with_no_atoms_of_requested_species(
    center: str, neighbor: str, expect_zeros: bool
) -> None:
    """Test RDF calculation when requested species don't exist in structure."""
    structure = Structure(
        Lattice.cubic(5),
        ["Si", "Si", "O", "O"],
        [(0, 0, 0), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25), (0.75, 0.75, 0.75)],
    )

    cutoff, n_bins = 10, 100
    radii, rdf = calculate_rdf(structure, center, neighbor, cutoff, n_bins)

    # Check basic properties
    check_basic_rdf_properties(radii, rdf, n_bins, f"{center}-{neighbor}")

    # Check expected RDF values
    if expect_zeros:
        assert np.all(rdf == 0), f"{rdf=} should be all zeros for {center}-{neighbor}"
    else:
        assert np.any(rdf > 0), (
            f"{rdf=} should have non-zero values for {center}-{neighbor}"
        )
        assert np.std(rdf) > 0.1, (
            f"{rdf=} should have significant variation for {center}-{neighbor}"
        )


@pytest.mark.parametrize(
    ("test_input", "expected_err_cls", "error_msg"),
    [
        (  # invalid_structure_type
            {"structure": "not a structure", "cutoff": 10, "n_bins": 10},
            TypeError,
            "Input must be a Pymatgen Structure, ASE Atoms",
        ),
        (  # zero_cutoff
            {"cutoff": 0},
            ValueError,
            "cutoff=0 must be positive",
        ),
        (  # negative_cutoff
            {"cutoff": -5},
            ValueError,
            "cutoff=-5 must be positive",
        ),
        (  # zero_n_bins
            {"n_bins": 0},
            ValueError,
            "n_bins=0 must be positive",
        ),
        (  # negative_n_bins
            {"n_bins": -5},
            ValueError,
            "n_bins=-5 must be positive",
        ),
        (  # invalid_pbc_type
            {"pbc": "invalid"},
            (TypeError, ValueError),  # Accept either error type
            None,  # Don't check the exact error message
        ),
    ],
)
def test_calculate_rdf_input_validation(
    test_input: dict[str, str | float],
    expected_err_cls: type | tuple[type, ...],
    error_msg: str | None,
) -> None:
    """Test input validation in calculate_rdf function."""
    # Create a simple structure for testing
    structure = Structure(Lattice.cubic(5), ["Si"], [[0, 0, 0]])

    # Set default parameters
    params = {"structure": structure, "cutoff": 10, "n_bins": 10} | test_input

    # Test for expected error
    structure_param = params.pop("structure")
    with pytest.raises(expected_err_cls, match=error_msg):
        calculate_rdf(structure_param, **params)


def test_calculate_rdf_invalid_structure_type() -> None:
    """Test that calculate_rdf raises appropriate error for invalid structure types."""
    with pytest.raises(
        TypeError, match="Input must be a Pymatgen Structure, ASE Atoms"
    ):
        calculate_rdf("not a structure", cutoff=10, n_bins=10)

    with pytest.raises(
        TypeError, match="Input must be a Pymatgen Structure, ASE Atoms"
    ):
        calculate_rdf(42, cutoff=10, n_bins=10)
