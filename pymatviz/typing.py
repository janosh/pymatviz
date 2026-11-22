"""Typing related: TypeAlias, generic types and so on."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

import pandas as pd


if TYPE_CHECKING:
    from ase.atoms import Atoms as AseAtoms
    from matscikit import Composition, Structure
    from matscikit import PhononDos as MskPhononDos
    from phonopy.api_phonopy import Phonopy
    from phonopy.phonon.dos import TotalDos
    from phonopy.structure.atoms import PhonopyAtoms
    from pymatgen.core import Composition as PmgComposition
    from pymatgen.core import IMolecule, IStructure
    from pymatgen.phonon.dos import PhononDos

    from pymatviz.enums import ElemCountMode

# Quoted unions: PEP 695 treats a bare `"Name"` as Literal, not a forward ref
# structure-like inputs, converted to matscikit Structures by normalize_structures
type AnyStructure = "Structure | IStructure | IMolecule | AseAtoms | PhonopyAtoms"
type PeriodicStructure = "Structure | IStructure | AseAtoms | PhonopyAtoms"
type AnyDos = "MskPhononDos | PhononDos | TotalDos | Phonopy"
# formula strings, {element: amount} mappings or Composition objects
type CompositionLike = "str | Mapping[str, float] | Composition | PmgComposition"
# also chemical system strings ("Fe-O") and structures, as count_formulas takes them
type FormulaLike = "CompositionLike | AnyStructure"
# space group numbers, Hermann-Mauguin/Hall symbols or periodic structures
type SpacegroupLike = "int | float | str | PeriodicStructure"

# ElemCountMode values, so callers can pass plain strings (count_mode="occurrence")
ElemCountModeName = Literal[
    "composition", "fractional_composition", "reduced_composition", "occurrence"
]
type ElemCountModeLike = "ElemCountMode | ElemCountModeName"

# Bare Literal assignments (not `type`): get_args() is empty on PEP 695 type aliases
CrystalSystem = Literal[
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
]

type ElemValues = (
    Mapping[str, int | float]
    | Mapping[int, int | float]
    | pd.Series
    | Sequence["str | Composition | PmgComposition"]
)

type Rgb256ColorType = tuple[int, int, int]  # 8-bit RGB
type RgbColorType = tuple[float, float, float] | str  # normalized to [0, 1]
type RgbAColorType = (  # normalized to [0, 1] with alpha
    str  # "none" or "#RRGGBBAA"/"#RGBA" hex strings
    | tuple[float, float, float, float]
    | tuple[RgbColorType, float]
    | tuple[tuple[float, float, float, float], float]
)
type ColorType = RgbColorType | RgbAColorType
FormulaGroupBy = Literal["formula", "reduced_formula", "chem_sys"]
