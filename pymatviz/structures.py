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


def _subclasses(obj: Any, *qualnames: str) -> bool:
    """Whether obj's class or a base has one of these dotted names, without imports."""
    return any(
        f"{cls.__module__}.{cls.__qualname__}" in qualnames for cls in type(obj).__mro__
    )


def is_ase_atoms(struct: Any) -> TypeIs[Atoms]:
    """Check if the input is an ASE Atoms object without importing ase."""
    return _subclasses(struct, "ase.atoms.Atoms", "pymatgen.io.ase.MSONAtoms")


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
    return _subclasses(obj, "phonopy.structure.atoms.PhonopyAtoms")


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
    """Map pymatgen structures/molecules, ASE Atoms or PhonopyAtoms (single, sequence
    or mapping) to pymatgen objects, keyed by formula for a single input, by
    "{index} {formula}" for sequences, and by the given keys for mappings/Series.
    """
    from pymatgen.core import IMolecule
    from pymatgen.io.ase import AseAtomsAdaptor

    def to_pmg_struct(item: Any) -> Structure | IStructure | Molecule | IMolecule:
        if is_ase_atoms(item):
            return AseAtomsAdaptor().get_structure(item)
        if isinstance(item, (IStructure, IMolecule)):
            return item
        if is_phonopy_atoms(item):
            from pymatgen.io.phonopy import get_pmg_structure

            return get_pmg_structure(item)
        raise TypeError(
            "Item must be a pymatgen Structure, IStructure, Molecule, IMolecule, "
            f"ASE Atoms, or PhonopyAtoms object, got {type(item)}"
        )

    if is_ase_atoms(systems) or is_phonopy_atoms(systems):
        systems = to_pmg_struct(systems)
    # structures are sequences of sites, so check them before Sequence
    if isinstance(systems, (IStructure, IMolecule)):
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


def str_labels[T](labeled: Mapping[Any, T]) -> dict[str, T]:
    """Mapping with its keys as strings.

    Raises:
        ValueError: If distinct keys share a string form (e.g. 1 and "1").
    """
    as_str = {str(key): val for key, val in labeled.items()}
    if len(as_str) < len(labeled):
        raise ValueError(f"labels must be unique as strings, got {list(labeled)}")
    return as_str


def normalize_periodic_structures(
    systems: _StructureInputs,
) -> dict[Hashable, IStructure]:
    """normalize_structures, raising TypeError for (non-periodic) molecules."""
    struct_dict = normalize_structures(systems)
    periodic = {
        key: val for key, val in struct_dict.items() if isinstance(val, IStructure)
    }
    if molecules := {
        key: type(val).__name__
        for key, val in struct_dict.items()
        if key not in periodic
    }:
        raise TypeError(f"Expected periodic structures, got {molecules}")
    return periodic


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
    """Space group numbers (1-230) from structures (pymatgen or ASE), numbers or
    Hermann-Mauguin symbols, keeping a Series' index and name.

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

    if isinstance(first_item, str):
        from pymatviz.utils.data import hm_symbol_to_spg_num_map

        numbers = result.map(hm_symbol_to_spg_num_map)
        if (missing := numbers.isna()).any():
            # slow pymatgen parses looser spellings ("P2_1/c", "R-3m:H"), once each
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
