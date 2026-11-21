"""Typing related: TypeAlias, generic types and so on."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

import pandas as pd


if TYPE_CHECKING:
    from ase.atoms import Atoms as AseAtoms
    from phonopy.phonon.dos import TotalDos
    from phonopy.structure.atoms import PhonopyAtoms
    from pymatgen.core import Composition, IMolecule, IStructure, Molecule, Structure
    from pymatgen.io.ase import MSONAtoms
    from pymatgen.phonon.dos import PhononDos

    from pymatviz.enums import ElemCountMode

# Quoted unions: PEP 695 treats a bare `"Name"` as Literal, not a forward ref
type AnyStructure = (
    "Structure | IStructure | Molecule | IMolecule | MSONAtoms | AseAtoms"
)
type PeriodicStructure = "IStructure | AseAtoms | PhonopyAtoms"
type AnyDos = "PhononDos | TotalDos"

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
    | Sequence["str | Composition"]
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
