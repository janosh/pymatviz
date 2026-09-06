"""Helper functions for radial distribution functions (RDFs) of pymatgen structures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

# compiled cython module, has no type stubs
from pymatgen.optimization.neighbors import (  # ty: ignore[unresolved-import]
    find_points_in_spheres,
)

from pymatviz.process_data import normalize_periodic_structures


if TYPE_CHECKING:
    from typing import Literal

    from pymatviz.typing import AnyStructure


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

    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError(f"{cutoff=} must be positive and finite")
    if not isinstance(n_bins, int | np.integer) or n_bins <= 0:
        raise ValueError(f"{n_bins=} must be positive and integral")
    pbc_array = np.asarray(pbc)
    if pbc_array.shape != (3,) or not np.isin(pbc_array, [0, 1]).all():
        raise ValueError(f"{pbc=} must contain three 0/1 flags")

    bin_size = cutoff / n_bins
    radii = np.linspace(0, cutoff, n_bins + 1)[1:]
    rdf = np.zeros_like(radii)

    # Import here to avoid circular import
    from pymatviz.structure.helpers import get_site_elements

    # Get indices of center and neighbor species
    if center_species:
        center_indices = [
            idx
            for idx, site in enumerate(struct)
            if center_species in get_site_elements(site)
        ]
    else:
        center_indices = list(range(len(struct)))

    if neighbor_species:
        neighbor_indices = [
            idx
            for idx, site in enumerate(struct)
            if neighbor_species in get_site_elements(site)
        ]
    else:
        neighbor_indices = list(range(len(struct)))

    if not center_indices or not neighbor_indices:
        return radii, rdf

    center_neighbors = find_points_in_spheres(
        all_coords=struct.cart_coords,
        center_coords=struct.cart_coords[center_indices],
        r=cutoff,
        pbc=pbc_array.astype(int),
        lattice=struct.lattice.matrix,
    )

    # Exclude the zero-distance self-pair, but count its periodic images.
    _, neighbor_ids, _, distances = center_neighbors
    selected = np.zeros(len(struct), dtype=bool)
    selected[neighbor_indices] = True
    distances = distances[
        selected[neighbor_ids] & (distances > 1e-10) & (distances < cutoff)
    ]
    bin_indices = np.minimum((distances / bin_size).astype(int), n_bins - 1)
    rdf = np.bincount(bin_indices, minlength=n_bins).astype(float)

    # Neighbor density is N/V without a -1 correction: periodic self-images count.
    normalization = len(center_indices) * len(neighbor_indices)

    # Approximate shell volume using its outer radius and thickness.
    rdf /= normalization
    shell_volumes = 4 * np.pi * radii**2 * bin_size
    rdf /= shell_volumes / struct.volume

    return radii, rdf
