"""Helper functions for radial distribution functions (RDFs) of structures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pymatviz.structures import (
    normalize_periodic_structures,
    site_elements,
    to_structure,
)


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal

    from pymatviz.typing import AnyStructure


def partial_rdfs(
    struct: AnyStructure,
    pairs: Sequence[tuple[str | None, str | None]],
    cutoff: float = 15,
    n_bins: int = 75,
    pbc: tuple[Literal[0, 1], Literal[0, 1], Literal[0, 1]] = (1, 1, 1),
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Partial radial distribution functions of several (center, neighbor) species
    pairs from one neighbor search.

    Args:
        struct (AnyStructure): Periodic structure-like object (matscikit, pymatgen,
            ASE, phonopy; disordered sites allowed).
        pairs (Sequence[tuple[str | None, str | None]]): (center, neighbor) element
            symbols, None meaning all sites. (None, None) gives the full RDF.
        cutoff (float, optional): Maximum distance for RDF calculation. Default is 15 Å.
        n_bins (int, optional): Number of bins. Default is 75.
        pbc (tuple[int, int, int], optional): Periodic boundary conditions as any
            3-tuple of 0s/1s. Defaults to (1, 1, 1).

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: (radii, g(r)) per pair, all-zero g(r)
            for pairs whose species no site contains.

    Raises:
        ValueError: If cutoff is not positive and finite, n_bins is not a positive
            integer, or pbc does not contain three 0/1 flags.
    """
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError(f"{cutoff=} must be positive and finite")
    if not isinstance(n_bins, int | np.integer) or n_bins <= 0:
        raise ValueError(f"{n_bins=} must be positive and integral")
    pbc_array = np.asarray(pbc)
    if pbc_array.shape != (3,) or not np.isin(pbc_array, [0, 1]).all():
        raise ValueError(f"{pbc=} must contain three 0/1 flags")

    struct = to_structure(struct)
    bin_size = cutoff / n_bins
    radii = np.linspace(0, cutoff, n_bins + 1)[1:]

    # site_has[species][idx]: whether site idx contains species (all sites for None,
    # disordered sites contain several elements)
    elements_per_site = site_elements(struct)
    site_has = {
        species: np.array(
            [not species or species in elems for elems in elements_per_site]
        )
        for species in {species for pair in pairs for species in pair}
    }
    # pairs with a species no site contains have all-zero g(r)
    live_pairs = {
        (center, neighbor)
        for center, neighbor in pairs
        if site_has[center].any() and site_has[neighbor].any()
    }
    if not live_pairs:
        return [(radii, np.zeros_like(radii)) for _ in pairs]

    if tuple(pbc_array.astype(bool)) != struct.pbc:
        struct = struct.replace(pbc=pbc_array.astype(bool).tolist())
    center_ids, neighbor_ids, _, distances = struct.neighbor_list(cutoff)
    # Exclude coincident atoms (the self pair is never listed), count periodic
    # images of a site itself.
    in_range = distances > 1e-10
    bin_indices = np.minimum((distances[in_range] / bin_size).astype(int), n_bins - 1)
    center_ids = center_ids[in_range]
    # flat (neighbor site, bin) index of each neighbor
    site_bin_ids = neighbor_ids[in_range] * n_bins + bin_indices
    shell_volumes = 4 * np.pi * radii**2 * bin_size
    # histogram neighbor (site, bin) counts once per center species, then sum the
    # rows of each neighbor species' sites (integer counts, so exact)
    site_hists: dict[str | None, np.ndarray] = {}
    rdfs = []
    for center, neighbor in pairs:
        if (center, neighbor) not in live_pairs:
            rdfs.append((radii, np.zeros_like(radii)))
            continue
        is_center, is_neighbor = site_has[center], site_has[neighbor]
        if center not in site_hists:
            site_hists[center] = np.bincount(
                site_bin_ids[is_center[center_ids]], minlength=len(struct) * n_bins
            ).reshape(len(struct), n_bins)
        rdf = site_hists[center][is_neighbor].sum(axis=0).astype(float)
        # Neighbor density is N/V without a -1 correction: periodic self-images
        # count. Approximate shell volume using its outer radius and thickness.
        rdf /= is_center.sum() * is_neighbor.sum()
        rdf /= shell_volumes / struct.volume
        rdfs.append((radii, rdf))
    return rdfs


def calculate_rdf(
    structure: AnyStructure,
    center_species: str | None = None,
    neighbor_species: str | None = None,
    cutoff: float = 15,
    n_bins: int = 75,
    pbc: tuple[Literal[0, 1], Literal[0, 1], Literal[0, 1]] = (1, 1, 1),
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the radial distribution function (RDF) for a given structure.

    If center_species and neighbor_species are provided, calculates the partial RDF
    for the specified element pair. Otherwise, calculates the full RDF.

    Args:
        structure (AnyStructure): A periodic pymatgen Structure/IStructure, ASE Atoms,
            or PhonopyAtoms object (or mapping/sequence thereof).
        center_species (str, optional): Symbol of the central species. If None, all
            species are considered.
        neighbor_species (str, optional): Symbol of the neighbor species. If None, all
            species are considered.
        cutoff (float, optional): Maximum distance for RDF calculation. Default is 15 Å.
        n_bins (int, optional): Number of bins for RDF calculation.
            Default is 75.
        pbc (tuple[int, int, int], optional): Periodic boundary conditions as any
            3-tuple of 0s/1s. Defaults to (1, 1, 1).

    Returns:
        tuple[np.ndarray, np.ndarray]: Arrays of (radii, g(r)) values.

    Raises:
        ValueError: If cutoff is not positive and finite, n_bins is not a positive
            integer, or pbc does not contain three 0/1 flags.
        TypeError: If structure is unsupported or not periodic (e.g. a Molecule).
    """
    struct = next(iter(normalize_periodic_structures(structure).values()))
    pair = (center_species, neighbor_species)
    return partial_rdfs(struct, [pair], cutoff=cutoff, n_bins=n_bins, pbc=pbc)[0]
