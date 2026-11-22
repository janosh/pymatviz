"""Typing related: TypeAlias, generic types and so on."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

import pandas as pd


if TYPE_CHECKING:
    from ase.atoms import Atoms as AseAtoms
    from matscikit import Composition, Structure
    from matscikit.interop import CompositionLike, PhononDosLike, StructureLike
    from phonopy.structure.atoms import PhonopyAtoms
    from pymatgen.core import Composition as PmgComposition
    from pymatgen.core import IStructure

    from pymatviz.enums import ElemCountMode

# Quoted unions: PEP 695 treats a bare `"Name"` as Literal, not a forward ref
# structure-like inputs (matscikit, pymatgen, ASE, phonopy), converted to matscikit
# Structures by normalize_structures; matscikit's interop types
type AnyStructure = "StructureLike"
type PeriodicStructure = "Structure | IStructure | AseAtoms | PhonopyAtoms"
type AnyDos = "PhononDosLike"
# CompositionLike (matscikit): formula strings, {element: amount} mappings (pymatgen
# Compositions among them) or matscikit Compositions. FormulaLike adds chemical system
# strings ("Fe-O") and structures, as count_formulas takes them
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
