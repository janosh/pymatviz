"""Structure inputs: detection, conversion to matscikit Structures, per-site elements
and space group numbers for matscikit, pymatgen, ASE and phonopy structures.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import matscikit as mk
import pandas as pd
from matscikit.interop import has_class, structure_to_pymatgen_dict, to_structure


if TYPE_CHECKING:
    from ase.atoms import Atoms
    from phonopy.structure.atoms import PhonopyAtoms
    from pymatgen.core import IMolecule, IStructure
    from typing_extensions import TypeIs

    from pymatviz.typing import AnyStructure


def is_ase_atoms(struct: Any) -> TypeIs[Atoms]:
    """Check if the input is an ASE Atoms object without importing ase."""
    return has_class(struct, "ase.atoms.Atoms", "pymatgen.io.ase.MSONAtoms")


def is_phonopy_atoms(obj: Any) -> TypeIs[PhonopyAtoms]:
    """Check if object is PhonopyAtoms or a subclass (Supercell, Primitive)."""
    return has_class(obj, "phonopy.structure.atoms.PhonopyAtoms")


def is_pymatgen_structure(obj: Any) -> TypeIs[IStructure | IMolecule]:
    """Whether obj is a pymatgen Structure/Molecule (mutable or not), without
    importing pymatgen.
    """
    return has_class(
        obj, "pymatgen.core.structure.IStructure", "pymatgen.core.structure.IMolecule"
    )


def is_structure_like(obj: Any) -> bool:
    """Whether obj is a matscikit or pymatgen structure, ASE Atoms or PhonopyAtoms."""
    return (
        isinstance(obj, mk.Structure)
        or is_pymatgen_structure(obj)
        or is_ase_atoms(obj)
        or is_phonopy_atoms(obj)
    )


def structure_dict(obj: Any) -> dict[str, Any]:
    """Dict in pymatgen's format (MatterViz's JSON structure format) of a structure-like
    object. pymatgen objects keep their own ``as_dict()`` (oxidation states, labels,
    site properties), non-periodic structures become molecule dicts without lattice.
    """
    if is_pymatgen_structure(obj):
        return obj.as_dict()
    struct = to_structure(obj)
    dct = structure_to_pymatgen_dict(struct)
    if not any(struct.pbc):
        info = struct.info
        dct = {key: val for key, val in dct.items() if key != "lattice"}
        dct |= {
            "@class": "Molecule",
            "charge": info.get("charge", 0),
            "spin_multiplicity": info.get("spin_multiplicity"),
            "sites": [
                {key: val for key, val in site.items() if key != "abc"}
                for site in dct["sites"]
            ],
        }
    return dct


def site_elements(struct: mk.Structure) -> list[set[str]]:
    """Element symbols on each site (several for disordered sites)."""
    return [set(occupancies) for occupancies in struct.site_occupancies]


def site_symbols(struct: mk.Structure) -> list[str]:
    """Element symbol of each site, the highest-occupancy one for disordered sites."""
    return [max(occu, key=occu.__getitem__) for occu in struct.site_occupancies]


type _StructureInput = AnyStructure | PhonopyAtoms
type _StructureInputs = (
    _StructureInput
    | Sequence[_StructureInput]
    | pd.Series
    | Mapping[Any, _StructureInput]
)


def normalize_structures(systems: _StructureInputs) -> dict[Hashable, mk.Structure]:
    """Map structure-like inputs (single, sequence or mapping) to matscikit
    Structures, keyed by formula for a single input, by "{index} {formula}" for
    sequences, and by the given keys for mappings/Series.

    Raises:
        ValueError: If systems is empty.
        TypeError: For unsupported input or item types.
    """
    if is_structure_like(systems):
        struct = to_structure(systems)
        return {struct.composition.formula: struct}

    if hasattr(systems, "__len__") and len(systems) == 0:
        raise ValueError("Cannot plot empty set of structures")

    if isinstance(systems, (pd.Series, Mapping)):  # keep original keys/index
        return {key: to_structure(val) for key, val in systems.items()}

    if isinstance(systems, Sequence) and not isinstance(systems, str):
        return {
            f"{idx} {(struct := to_structure(item)).composition.formula}": struct
            for idx, item in enumerate(systems, start=1)
        }

    raise TypeError(
        "Input must be a matscikit or pymatgen Structure/Molecule, ASE Atoms or "
        "PhonopyAtoms, a sequence (list, tuple, pd.Series) or a mapping of them. "
        f"Got {type(systems)=}"
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
) -> dict[Hashable, mk.Structure]:
    """normalize_structures, raising TypeError for non-periodic ones (molecules)."""
    struct_dict = normalize_structures(systems)
    if molecules := [key for key, val in struct_dict.items() if not any(val.pbc)]:
        raise TypeError(f"Expected periodic structures, got molecules {molecules}")
    return struct_dict


def get_spacegroup_number(structure: Any) -> int:
    """Space group number of a periodic structure-like object (disordered sites
    are told apart by species and occupancy).

    Raises:
        TypeError: For unsupported or non-periodic inputs.
    """
    struct = next(iter(normalize_periodic_structures(structure).values()))
    return mk.symmetry_dataset(struct).number


def normalize_spacegroups(
    data: Sequence[int | str | Any] | pd.Series,
) -> pd.Series:
    """Space group numbers (1-230) from structures, numbers or Hermann-Mauguin
    symbols, keeping a Series' index and name.

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
        numbers: dict[str, int] = {}
        invalid: list[str] = []
        for symbol in result.unique():  # loose spellings, settings, Hall symbols
            try:
                numbers[symbol] = mk.space_group_number(symbol)
            except ValueError:
                invalid.append(symbol)
        if invalid:
            raise ValueError(f"Invalid space group symbols: {invalid}")
        return result.map(numbers).astype(int)

    return result
