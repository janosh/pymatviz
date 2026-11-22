"""Structure inputs: detection, conversion to kemio Structures, per-site elements
and space group numbers for kemio, pymatgen, ASE and phonopy structures.
"""

from __future__ import annotations

import os
from collections.abc import Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import kemio
import pandas as pd
from kemio.interop import (
    is_pymatgen_structure,
    is_structure_like,
    structure_to_pymatgen_dict,
    to_structure,
)


if TYPE_CHECKING:
    from pymatviz.typing import AnyStructure, PeriodicStructure, SpacegroupLike


def structure_dict(obj: AnyStructure) -> dict[str, Any]:
    """Dict in pymatgen's format (MatterViz's JSON structure format) of a structure-like
    object: pymatgen's own ``as_dict()`` for pymatgen objects, else kemio's
    ``Structure.as_dict()``/``Molecule.as_dict()`` payload (molecules without lattice),
    with ``Structure.info`` (e.g. ASE ``Atoms.info``) as ``properties`` like pymatgen.
    """
    if is_pymatgen_structure(obj):
        return obj.as_dict()
    struct = to_structure(obj)
    struct_dict = structure_to_pymatgen_dict(struct)
    struct_dict["properties"] = struct.info | struct_dict["properties"]
    return struct_dict


def site_elements(struct: kemio.Structure) -> list[set[str]]:
    """Element symbols on each site (several for disordered sites)."""
    return [set(occupancies) for occupancies in struct.site_occupancies]


def site_symbols(struct: kemio.Structure) -> list[str]:
    """Element symbol of each site, the highest-occupancy one for disordered sites,
    "X" for vacant sites (pymatgen's vacancy DummySpecies).
    """
    return [
        max(occu, key=occu.__getitem__) if occu else "X"
        for occu in struct.site_occupancies
    ]


# structure file paths count as structures, alone or as sequence/mapping items
_PATHS = (str, os.PathLike)
type _StructureItem = AnyStructure | str | os.PathLike[str]
type _StructureInputs = (
    _StructureItem | Sequence[_StructureItem] | pd.Series | Mapping[Any, _StructureItem]
)


def normalize_structures(systems: _StructureInputs) -> dict[Hashable, kemio.Structure]:
    """Map structure-like inputs (single, sequence or mapping) to kemio
    Structures, keyed by formula for a single input, by "{index} {formula}" for
    sequences, and by the given keys for mappings/Series. A structure file path or
    pymatgen-format structure dict (with "sites") counts as a single input.

    Raises:
        ValueError: If systems is empty.
        TypeError: For unsupported input or item types.
        FileNotFoundError: For a path that isn't a file.
    """
    is_single_dict = isinstance(systems, Mapping) and "sites" in systems
    if is_structure_like(systems) or is_single_dict or isinstance(systems, _PATHS):
        struct = to_structure(systems)
        return {struct.composition.formula: struct}

    if hasattr(systems, "__len__") and len(systems) == 0:
        raise ValueError("Cannot plot empty set of structures")

    if isinstance(systems, (pd.Series, Mapping)):  # keep original keys/index
        if isinstance(systems, pd.Series) and systems.index.has_duplicates:
            dupes = systems.index[systems.index.duplicated()].unique().tolist()
            raise ValueError(f"structure labels must be unique, got duplicates {dupes}")
        return {key: to_structure(val) for key, val in systems.items()}

    if isinstance(systems, Sequence):
        return {
            f"{idx} {(struct := to_structure(item)).composition.formula}": struct
            for idx, item in enumerate(systems, start=1)
        }

    raise TypeError(
        "Input must be a kemio or pymatgen Structure/Molecule, ASE Atoms or "
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
) -> dict[Hashable, kemio.Structure]:
    """normalize_structures, raising TypeError for structures not periodic along all
    three axes (molecules, slabs, wires, ASE Atoms with the default pbc=False).
    """
    struct_dict = normalize_structures(systems)
    if non_periodic := {
        key: val.pbc for key, val in struct_dict.items() if not val.is_periodic
    }:
        raise TypeError(
            f"Expected structures periodic along a, b and c, got pbc {non_periodic} "
            "(set atoms.pbc = True for ASE Atoms of a crystal)"
        )
    return struct_dict


def spacegroup_numbers(structures: Sequence[PeriodicStructure]) -> list[int]:
    """Space group numbers of periodic structure-like objects, searched in parallel
    (disordered sites are told apart by species and occupancy).

    Raises:
        TypeError: For unsupported or non-periodic inputs, or structures whose
            symmetry search fails (naming their indices).
    """
    structs = list(normalize_periodic_structures(list(structures)).values())
    numbers = kemio.symmetry_summary(structs).number
    if failed := [idx for idx, number in enumerate(numbers) if number is None]:
        raise TypeError(f"Symmetry search failed for structures at indices {failed}")
    return [number for number in numbers if number is not None]


def normalize_spacegroups(data: Sequence[SpacegroupLike] | pd.Series) -> pd.Series:
    """Space group numbers (1-230) from structures, numbers or Hermann-Mauguin/Hall
    symbols (one kind per input), keeping a Series' index and name.

    Raises:
        ValueError: If data is empty, contains missing values, invalid symbols,
            or nonintegral/out-of-range space group numbers.
        TypeError: For mixed or unsupported item types.
    """
    if len(data) == 0:
        raise ValueError("Cannot normalize empty spacegroup data")

    # infer_objects lets object-dtype numbers (e.g. [225.5, 2]) reach the range and
    # integrality check below instead of being truncated by a later int cast
    result = pd.Series(data).infer_objects()
    if result.isna().any():
        raise ValueError("Space group data must not contain missing values")

    # first item short-circuits number/symbol inputs, mixed ones fall to the TypeError
    if is_structure_like(result.iloc[0]) and all(map(is_structure_like, result)):
        numbers = spacegroup_numbers(result.tolist())
        return pd.Series(numbers, index=result.index, name=result.name)

    if pd.api.types.is_numeric_dtype(result) and not pd.api.types.is_bool_dtype(result):
        invalid = result[(result < 1) | (result > 230) | (result % 1 != 0)]
        if len(invalid) > 0:
            raise ValueError(
                "Space group numbers must be in [1, 230] and integral, "
                f"got: {invalid.tolist()}"
            )
        return result.astype(int)

    if all(isinstance(symbol, str) for symbol in result):
        numbers_of: dict[str, int] = {}
        invalid_symbols: list[str] = []
        for symbol in result.unique():  # loose spellings, settings, Hall symbols
            try:
                numbers_of[symbol] = kemio.space_group_number(symbol)
            except ValueError:
                invalid_symbols.append(symbol)
        if invalid_symbols:
            raise ValueError(f"Invalid space group symbols: {invalid_symbols}")
        return result.map(numbers_of).astype(int)

    item_types = sorted({type(item).__name__ for item in result})
    raise TypeError(
        "Expected structures, space group numbers or symbols (one kind), got "
        f"{item_types}"
    )


def formula_spg_str(struct: PeriodicStructure) -> str:
    """Get a string with formula and space group number for a structure.

    Args:
        struct: A periodic structure (kemio, pymatgen, ASE, phonopy).

    Returns:
        String in format "Formula (spg_number)", e.g. "Na4 Cl4 (225)".
    """
    struct = to_structure(struct)
    return f"{struct.composition.formula} ({kemio.symmetry_dataset(struct).number})"
