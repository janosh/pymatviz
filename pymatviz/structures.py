"""Structure input types: detection, conversion to pymatgen, site helpers and space
group numbers for pymatgen, ASE and phonopy structures.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd
from pymatgen.core import IStructure, Structure


if TYPE_CHECKING:
    from ase.atoms import Atoms
    from phonopy.structure.atoms import PhonopyAtoms
    from pymatgen.core import Composition, IMolecule, Molecule, Site, Species
    from pymatgen.core.periodic_table import DummySpecies, Element
    from typing_extensions import TypeIs

    from pymatviz.typing import AnyStructure


def is_structure_like(obj: Any) -> bool:
    """Whether obj is a pymatgen Structure/Molecule, ASE Atoms or PhonopyAtoms."""
    from pymatgen.core import IMolecule

    return (
        isinstance(obj, (IStructure, IMolecule))
        or is_ase_atoms(obj)
        or is_phonopy_atoms(obj)
    )


def is_ase_atoms(struct: Any) -> TypeIs[Atoms]:
    """Check if the input is an ASE Atoms object without importing ase."""
    return any(
        f"{cls.__module__}.{cls.__qualname__}"
        in ("ase.atoms.Atoms", "pymatgen.io.ase.MSONAtoms")
        for cls in type(struct).__mro__
    )


MIN_ASE_CELL_VOLUME = 1e-6  # Å³


def get_site_species(site: Site) -> Composition | Species | Element | DummySpecies:
    """Species of an ordered site, composition of a disordered one."""
    return site.specie if site.is_ordered else site.species


def get_site_elements(site: Site) -> set[str]:
    """Element symbols on a site (several for disordered sites)."""
    return {species.symbol for species in site.species}


def get_site_symbol(site: Site) -> str:
    """Element symbol of a site, the highest-occupancy one for disordered sites."""
    el_amt_dict = site.species.get_el_amt_dict()
    return max(el_amt_dict, key=el_amt_dict.__getitem__)


def add_vacuum_if_needed(struct: Any) -> Any:
    """Center ASE Atoms without a proper cell (e.g. molecules) in 10 Å of vacuum."""
    if is_ase_atoms(struct) and struct.cell.volume < MIN_ASE_CELL_VOLUME:
        struct = struct.copy()
        struct.center(vacuum=10.0)
    return struct


def is_phonopy_atoms(obj: Any) -> TypeIs[PhonopyAtoms]:
    """Check if object is PhonopyAtoms or a subclass (Supercell, Primitive)."""
    return any(
        f"{cls.__module__}.{cls.__qualname__}" == "phonopy.structure.atoms.PhonopyAtoms"
        for cls in type(obj).__mro__
    )


type _StructureInput = AnyStructure | PhonopyAtoms
type _StructureInputs = (
    _StructureInput
    | Sequence[_StructureInput]
    | pd.Series
    | Mapping[Any, _StructureInput]
)


def normalize_structures(
    systems: _StructureInputs,
) -> dict[Hashable, Structure | IStructure | Molecule | IMolecule]:
    """Convert pymatgen Structures/IStructures, ASE Atoms, or PhonopyAtoms or
    sequences/dicts of them to a dictionary mapping hashable keys to pymatgen
    Structure-likes (Structure, IStructure, Molecule, and IMolecule).

    Keys are derived from formulas for single structures, indices for sequences,
    or preserved from input dicts/Series.
    """
    from pymatgen.core import IMolecule
    from pymatgen.io.ase import AseAtomsAdaptor

    def to_pmg_struct(item: Any) -> Structure | IStructure | Molecule | IMolecule:
        if is_ase_atoms(item):
            return AseAtomsAdaptor().get_structure(item)
        if isinstance(item, (IStructure, IMolecule)):
            return item
        if is_phonopy_atoms(item):  # convert PhonopyAtoms to pymatgen Structure
            from pymatgen.io.phonopy import get_pmg_structure

            return get_pmg_structure(item)
        raise TypeError(
            "Item must be a pymatgen Structure, IStructure, Molecule, IMolecule, "
            f"ASE Atoms, or PhonopyAtoms object, got {type(item)}"
        )

    if is_ase_atoms(systems) or is_phonopy_atoms(systems):
        # Handles single ASE Atoms or PhonopyAtoms object
        systems = to_pmg_struct(systems)

    # Check for single Structure/IStructure first, before checking for Sequence
    # since they are Sequences but we don't want to iterate over sites
    if isinstance(systems, (IStructure, IMolecule)):
        # Use formula as key for single structure input
        return {systems.formula: systems}

    if hasattr(systems, "__len__") and len(systems) == 0:
        raise ValueError("Cannot plot empty set of structures")

    if isinstance(systems, (pd.Series, Mapping)):  # keep original keys/index
        return {key: to_pmg_struct(val) for key, val in systems.items()}

    if isinstance(systems, Sequence) and not isinstance(systems, str):
        return {
            f"{idx} {(struct := to_pmg_struct(item)).formula}": struct
            for idx, item in enumerate(systems, start=1)
        }

    raise TypeError(
        "Input must be a pymatgen Structure, IStructure, Molecule, IMolecule, "
        "ASE Atoms, or PhonopyAtoms object, a sequence (list, tuple, pd.Series), "
        f"or a mapping. Got {type(systems)=}"
    )


def normalize_periodic_structures(
    systems: _StructureInputs,
) -> dict[Hashable, IStructure]:
    """Like normalize_structures but requires all inputs to be periodic.

    Raises:
        TypeError: If any of the normalized structures is a (non-periodic) molecule.
    """
    struct_dict: dict[Hashable, IStructure] = {}
    for key, struct in normalize_structures(systems).items():
        if not isinstance(struct, IStructure):
            raise TypeError(
                f"Expected periodic structure for key={key!r}, got "
                f"{type(struct).__name__}"
            )
        struct_dict[key] = struct
    return struct_dict


def get_spacegroup_number(structure: AnyStructure) -> int:
    """Determine an ordered periodic structure's space group without adapter extras."""
    from moyopy import Cell, MoyoDataset

    if isinstance(structure, IStructure):
        if not structure.is_ordered:
            raise ValueError("Structure must be ordered")
        cell = Cell(
            basis=structure.lattice.matrix.tolist(),
            positions=structure.frac_coords.tolist(),
            numbers=list(structure.atomic_numbers),
        )
    elif is_ase_atoms(structure):
        cell = Cell(
            basis=structure.cell.tolist(),
            positions=structure.get_scaled_positions().tolist(),
            numbers=structure.get_atomic_numbers().tolist(),
        )
    else:
        raise TypeError(
            f"Expected periodic Structure or ASE Atoms, got {type(structure)}"
        )
    return MoyoDataset(cell).number


def normalize_spacegroups(
    data: Sequence[int | str | Any] | pd.Series,
) -> pd.Series:
    """Normalize spacegroup data from structures, numbers, or symbols to a Series.

    Handles pymatgen Structure, ASE Atoms, space group numbers (1-230), or
    Hermann-Mauguin symbols.

    Args:
        data (Sequence[int | str | Structure] | pd.Series): Space group data. Can be:
            - Sequence of pymatgen Structures or ASE Atoms
            - Sequence of space group numbers (1-230)
            - Sequence of Hermann-Mauguin symbols
            - pandas Series of any of the above

    Returns:
        pd.Series: Integer space group numbers (1-230), preserving index and name.

    Raises:
        ValueError: If data is empty, contains missing values, invalid symbols,
            or nonintegral/out-of-range space group numbers.
    """
    if len(data) == 0:
        raise ValueError("Cannot normalize empty spacegroup data")

    # infer_objects lets object-dtype numbers (e.g. [225.5, 2]) reach the range and
    # integrality check below instead of being truncated by a later int cast
    result = pd.Series(data).infer_objects()
    if result.isna().any():
        raise ValueError("Space group data must not contain missing values")
    first_item = result.iloc[0]

    if is_structure_like(first_item):
        return result.map(get_spacegroup_number)

    if pd.api.types.is_numeric_dtype(result):
        invalid = result[(result < 1) | (result > 230) | (result % 1 != 0)]
        if len(invalid) > 0:
            raise ValueError(
                "Space group numbers must be in [1, 230] and integral, "
                f"got: {invalid.tolist()}"
            )
        return result.astype(int)

    # Convert Hermann-Mauguin symbols to space group numbers
    if isinstance(first_item, str):
        from pymatviz.utils.data import hm_symbol_to_spg_num_map

        numbers = result.map(hm_symbol_to_spg_num_map)
        if (missing := numbers.isna()).any():
            # pymatgen parses looser spellings (e.g. "P2_1/c", "R-3m:H") the
            # moyopy-derived lookup table lacks, but is slow, so only parse unique
            # misses
            from pymatgen.symmetry.groups import SpaceGroup

            parsed: dict[str, int] = {}
            for symbol in result[missing].unique():
                try:
                    parsed[symbol] = SpaceGroup(symbol).int_number
                except ValueError as exc:
                    raise ValueError(f"Invalid Hermann-Mauguin symbol: {exc}") from exc
            numbers = numbers.fillna(result.map(parsed))
        return numbers.astype(int)

    return result
