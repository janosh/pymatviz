"""Helper functions for radial distribution functions (RDFs) of pymatgen structures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pymatgen.core import Structure
from pymatgen.optimization.neighbors import find_points_in_spheres


if TYPE_CHECKING:
    from pymatgen.util.typing import PbcLike


def calculate_rdf(
    structure: Structure,
    center_species: str | None = None,
    neighbor_species: str | None = None,
    cutoff: float = 15,
    n_bins: int = 75,
    pbc: PbcLike = (True, True, True),
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the radial distribution function (RDF) for a given structure.

    If center_species and neighbor_species are provided, calculates the partial RDF
    for the specified element pair. Otherwise, calculates the full RDF.

    Args:
        structure (Structure): A pymatgen Structure object.
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
        ValueError: If cutoff or n_bins are not positive values.
    """
    # Input validation
    if not isinstance(structure, Structure):
        raise TypeError(f"Expected pymatgen Structure, got {type(structure).__name__}")
    # Handle empty structure
    if len(structure) == 0:
        return np.linspace(cutoff / n_bins, cutoff, n_bins), np.zeros(n_bins)
    if cutoff <= 0:
        raise ValueError(f"{cutoff=} must be positive")
    if n_bins <= 0:
        raise ValueError(f"{n_bins=} must be positive")

    bin_size = cutoff / n_bins
    radii = np.linspace(0, cutoff, n_bins + 1)[1:]
    rdf = np.zeros_like(radii)

    # Get indices of center and neighbor species
    if center_species:
        center_indices = [
            idx
            for idx, site in enumerate(structure)
            if site.specie.symbol == center_species
        ]
    else:
        center_indices = list(range(len(structure)))

    if neighbor_species:
        neighbor_indices = [
            idx
            for idx, site in enumerate(structure)
            if site.specie.symbol == neighbor_species
        ]
    else:
        neighbor_indices = list(range(len(structure)))

    # If there are no center atoms or neighbor atoms, return an empty RDF
    if not center_indices or not neighbor_indices:
        return radii, rdf  # Return zeros if no centers or neighbors

    center_neighbors = find_points_in_spheres(
        all_coords=structure.cart_coords,
        center_coords=structure.cart_coords[center_indices],
        r=cutoff,
        # Convert bools to ints (needed for cython code)
        pbc=np.array([*map(int, pbc)]),
        lattice=structure.lattice.matrix,
    )

    # Filter distances for the specific neighbor species and bin them
    neighbor_set = set(neighbor_indices)
    for idx1, idx2, _, dist in zip(*center_neighbors, strict=True):
        if idx2 in neighbor_set and center_indices[idx1] != idx2 and 0 < dist < cutoff:
            bin_index = min(int(dist / bin_size), n_bins - 1)
            rdf[bin_index] += 1

    # Normalize RDF by the number of center-neighbor pairs and shell volumes
    n_center = len(center_indices)
    n_neighbor = len(neighbor_indices)
    if center_species == neighbor_species:
        normalization = n_center * (n_neighbor - 1)  # Exclude self-interactions
    else:
        normalization = n_center * n_neighbor

    if normalization == 0:  # Avoid division by zero
        return radii, np.zeros_like(radii)

    # Spherical shell volume = surface area (4πr²) times thickness (bin_size)
    rdf /= normalization
    shell_volumes = 4 * np.pi * radii**2 * bin_size
    rdf /= shell_volumes / structure.volume

    return radii, rdf
