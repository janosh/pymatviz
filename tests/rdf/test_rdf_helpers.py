from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymatviz.rdf.helpers import calculate_rdf
from tests.conftest import FCC_SITES, SI_STRUCTS, cubic_structure


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    import matscikit as mk


def check_basic_rdf_properties(
    radii: np.ndarray, rdf: np.ndarray, n_bins: int, name: str = ""
) -> None:
    """Check array types, lengths, non-negativity and g(r)=0 in the first bin."""
    assert isinstance(radii, np.ndarray), name
    assert isinstance(rdf, np.ndarray), name
    assert len(radii) == len(rdf) == n_bins, f"{name}: {rdf=}"
    assert np.all(rdf >= 0), f"{name}: {rdf=}"
    assert rdf[0] == 0, f"{name}: {rdf=}"


def test_calculate_rdf() -> None:
    """Full and Si-Si partial RDFs of Si2 agree."""
    radii, rdf = calculate_rdf(SI_STRUCTS[0], cutoff=10, n_bins=100)
    check_basic_rdf_properties(radii, rdf, 100)
    radii_partial, rdf_partial = calculate_rdf(SI_STRUCTS[0], "Si", "Si", 10, 100)
    check_basic_rdf_properties(radii_partial, rdf_partial, 100)
    # Si2 has only Si, so the partial RDF equals the full RDF
    np.testing.assert_array_equal(rdf_partial, rdf)


@pytest.mark.parametrize(
    ("composition", "n_atoms"),
    [(["Si"], 100), (["Si", "Ge"], 100), (["Al", "O"], 100), (["Fe", "Ni", "Cr"], 165)],
)
def test_calculate_rdf_normalization(composition: list[str], n_atoms: int) -> None:
    """Partial RDFs of random structures approach 1 at large separations."""
    elements = [elem for elem in composition for _ in range(n_atoms)]
    coords = np.random.default_rng(seed=0).uniform(size=(len(elements), 3))
    structure = cubic_structure(30, elements, coords)
    cutoff, n_bins = 12, 75
    for el1 in composition:
        for el2 in composition:
            radii, rdf = calculate_rdf(structure, el1, el2, cutoff, n_bins)
            check_basic_rdf_properties(radii, rdf, n_bins, f"{el1}-{el2} pair")
            avg_tail = round(np.mean(rdf[int(0.9 * len(rdf)) :]), 4)
            assert 0.95 <= avg_tail <= 1.05, f"{el1}-{el2}: {avg_tail=}"
            assert_allclose(radii, np.linspace(cutoff / n_bins, cutoff, n_bins))


@pytest.mark.parametrize(
    ("pbc", "expected_peak"),
    [
        ((1, 1, 1), (4.33, 1.0)),  # (peak_position, min_height)
        ((1, 1, 0), (4.33, 0.0)),  # Just check for non-zero value
        ((1, 0, 0), None),  # No specific peak check
        ((0, 0, 0), None),
    ],
)
def test_calculate_rdf_pbc_settings(
    pbc: tuple[Literal[0, 1], Literal[0, 1], Literal[0, 1]],
    expected_peak: tuple[float, float] | None,
) -> None:
    """Body-centered Si2 RDF per PBC setting, with the sqrt(3)/2 a peak if periodic."""
    structure = cubic_structure(5, ["Si"] * 2, [[0] * 3, [0.5] * 3])
    cutoff, n_bins = 10, 100
    radii, rdf = calculate_rdf(structure, "Si", "Si", cutoff, n_bins, pbc=pbc)
    check_basic_rdf_properties(radii, rdf, n_bins, f"PBC {pbc}")
    if expected_peak:
        peak_pos, min_height = expected_peak
        assert rdf[int(peak_pos / cutoff * n_bins)] > min_height, f"{pbc=}"


def test_calculate_rdf_pbc_consistency() -> None:
    """Full PBC counts more neighbors than no PBC, with regression sums for both."""
    coords = np.random.default_rng(seed=0).uniform(size=(20, 3))
    structure = cubic_structure(10, ["Si"] * 20, coords)
    (_, rdf_full_pbc), (_, rdf_no_pbc) = (
        calculate_rdf(structure, "Si", "Si", cutoff=15, n_bins=150, pbc=pbc)
        for pbc in ((1, 1, 1), (0, 0, 0))
    )
    assert np.sum(rdf_full_pbc > 0) > 0
    assert np.sum(rdf_no_pbc > 0) < np.sum(rdf_full_pbc > 0)
    assert np.sum(rdf_no_pbc) == pytest.approx(29.0232140)
    # a periodic self-image at exactly the 10 Å lattice constant sits on a bin edge:
    # matscikit's 9.999999999999998 Å (2 ulp below pymatgen's 10.0) bins it at 9.9 Å
    assert np.sum(rdf_full_pbc) == pytest.approx(136.9477329)


def test_calculate_rdf_different_species() -> None:
    """Same-species RDFs count periodic images and cross-species peaks appear."""
    structure = cubic_structure(5, ["Si", "Ge"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    cutoff, n_bins = 10, 100
    _, rdf_si_si, _, rdf_si_ge, _, rdf_ge_ge = (
        arr
        for center, neighbor in (("Si", "Si"), ("Si", "Ge"), ("Ge", "Ge"))
        for arr in calculate_rdf(structure, center, neighbor, cutoff, n_bins)
    )
    # Same-species RDFs must count periodic images of the single atom at the lattice
    # constant (regression: they were incorrectly all-zero before)
    assert rdf_si_si[int(5.0 / cutoff * n_bins)] > 0
    assert np.allclose(rdf_si_si, rdf_ge_ge), "Si-Si and Ge-Ge differ"
    assert rdf_si_ge[int(4.33 / cutoff * n_bins)] > 0  # sqrt(3)/2 * lattice constant


@pytest.mark.parametrize(
    ("cutoff", "frac_coords"), [(4, [0.9, 0.9, 0.9]), (0.1, [0.1, 0.1, 0.1])]
)
def test_calculate_rdf_edge_cases(cutoff: float, frac_coords: list[float]) -> None:
    """RDF is zero when no neighbors lie within the cutoff."""
    single_atom = cubic_structure(5, ["Si"], [[0, 0, 0]])
    _, rdf = calculate_rdf(single_atom, "Si", "Si", cutoff=cutoff, n_bins=100)
    assert np.all(rdf == 0), f"{rdf=}"
    distant_atoms = cubic_structure(5, ["Si", "Si"], [[0, 0, 0], frac_coords])
    _, rdf = calculate_rdf(distant_atoms, "Si", "Si", cutoff, n_bins=30, pbc=(0, 0, 0))
    assert np.all(rdf == 0), f"{rdf=}"


def test_calculate_rdf_disordered_structure(fe3co4_disordered: mk.Structure) -> None:
    """Disordered sites (Fe3CO4 and an equimolar FCC high-entropy alloy) give
    regression peak positions, heights and spreads.
    """
    cutoff, n_bins = 10, 100
    radii, rdf = calculate_rdf(fe3co4_disordered, cutoff=cutoff, n_bins=n_bins)
    check_basic_rdf_properties(radii, rdf, n_bins, "disordered")
    assert np.argmax(rdf) == 43
    assert rdf[43] == pytest.approx(20.552033)
    assert np.std(rdf) == pytest.approx(3.12714)

    hea_species = [dict.fromkeys(("Co", "Cr", "Fe", "Ni", "Mn"), 0.2)] * 4
    hea_structure = cubic_structure(a_len := 3.59, hea_species, FCC_SITES)
    radii, rdf = calculate_rdf(hea_structure, cutoff=cutoff, n_bins=n_bins)
    check_basic_rdf_properties(radii, rdf, n_bins, "HEA")
    fcc_nn_idx = int(a_len / np.sqrt(2) / cutoff * n_bins)  # a/√2 ≈ 2.54Å
    assert rdf[fcc_nn_idx] == pytest.approx(16.339849)
    assert np.argmax(rdf) == 25
    assert np.std(rdf) == pytest.approx(2.547041905)


@pytest.mark.parametrize(
    ("cutoff", "n_bins", "holds"),
    [
        (0.1, 10, lambda rdf: np.all(rdf == 0)),
        (10, 1000, lambda rdf: np.any(rdf > 0)),
        (5, 20, lambda rdf: np.std(rdf) > 0),
    ],
    ids=["all_zeros", "has_nonzero", "has_variation"],
)
def test_calculate_rdf_extreme_parameters(
    cutoff: float, n_bins: int, holds: Callable[[np.ndarray], bool]
) -> None:
    """Tiny cutoffs give zeros, many or few bins still resolve the FCC shells."""
    structure = cubic_structure(5, ["Si"] * 4, FCC_SITES)
    radii, rdf = calculate_rdf(structure, cutoff=cutoff, n_bins=n_bins)
    check_basic_rdf_properties(radii, rdf, n_bins, f"{cutoff=}, {n_bins=}")
    assert holds(rdf), f"{cutoff=}, {n_bins=}: {rdf=}"


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
    """Absent center or neighbor species give all-zero RDFs."""
    structure = cubic_structure(
        5,
        ["Si", "Si", "O", "O"],
        [(0, 0, 0), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25), (0.75, 0.75, 0.75)],
    )
    radii, rdf = calculate_rdf(structure, center, neighbor, cutoff=10, n_bins=100)
    check_basic_rdf_properties(radii, rdf, 100, f"{center}-{neighbor}")
    if expect_zeros:
        assert np.all(rdf == 0), f"{center}-{neighbor}: {rdf=}"
    else:
        assert np.any(rdf > 0), f"{center}-{neighbor}: {rdf=}"
        assert np.std(rdf) > 0.1, f"{center}-{neighbor}: {rdf=}"


@pytest.mark.parametrize(
    ("test_input", "expected_err_cls", "error_msg"),
    [
        ({"structure": "not a structure"}, FileNotFoundError, "no structure file at"),
        ({"cutoff": 0}, ValueError, "cutoff=0 must be positive"),
        ({"cutoff": -5}, ValueError, "cutoff=-5 must be positive"),
        ({"n_bins": 0}, ValueError, "n_bins=0 must be positive"),
        ({"n_bins": -5}, ValueError, "n_bins=-5 must be positive"),
        ({"cutoff": float("nan")}, ValueError, "must be positive and finite"),
        ({"cutoff": float("inf")}, ValueError, "must be positive and finite"),
        ({"n_bins": 2.5}, ValueError, "must be positive and integral"),
        ({"pbc": "invalid"}, (TypeError, ValueError), None),
    ],
)
@pytest.mark.parametrize("empty", [False, True])
def test_calculate_rdf_input_validation(
    test_input: dict[str, str | float],
    expected_err_cls: type[Exception] | tuple[type[Exception], ...],
    error_msg: str | None,
    empty: bool,
) -> None:
    """Invalid structures, cutoffs, bin counts and pbc raise, also for empty cells."""
    structure = cubic_structure(5, [] if empty else ["Si"], [] if empty else [0, 0, 0])
    params = {"structure": structure, "cutoff": 10, "n_bins": 10} | test_input
    structure_param = params.pop("structure")
    with pytest.raises(expected_err_cls, match=error_msg):
        calculate_rdf(structure_param, **params)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("species", "frac_coords", "a_len", "first_shell_max", "expected_cn"),
    [
        ("Ag", FCC_SITES, 4.08, 3.4, 12),  # FCC: 12 neighbors at a/sqrt(2)
        ("Fe", [[0, 0, 0], [0.5, 0.5, 0.5]], 2.87, 2.7, 8),  # BCC: 8 at sqrt(3)/2 a
        ("Po", [[0, 0, 0]], 3.35, 4.0, 6),  # simple cubic (single-atom cell): 6 at a
    ],
)
def test_calculate_rdf_coordination_numbers(
    species: str,
    frac_coords: list[list[float]],
    a_len: float,
    first_shell_max: float,
    expected_cn: int,
) -> None:
    """Integrating g(r) over the first shell gives the textbook coordination
    number (regression: same-species RDFs dropped periodic images of the center
    atom, e.g. zeroing single-atom-cell RDFs entirely).
    """
    struct = cubic_structure(a_len, [species] * len(frac_coords), frac_coords)
    radii, rdf = calculate_rdf(struct, species, species, cutoff=6, n_bins=600)
    rho, bin_width = len(struct) / struct.volume, radii[1] - radii[0]
    mask = radii < first_shell_max  # integrate g(r) over the first shell
    coord_num = np.sum(rdf[mask] * rho * 4 * np.pi * radii[mask] ** 2 * bin_width)
    assert coord_num == pytest.approx(expected_cn, rel=1e-6)


def test_calculate_rdf_rejects_several_structures() -> None:
    """A sequence of structures raises instead of silently using the first."""
    with pytest.raises(TypeError, match="takes one structure, got 2"):
        calculate_rdf(SI_STRUCTS[:2])  # ty: ignore[invalid-argument-type]
