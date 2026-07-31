"""Typing related: TypeAlias, generic types and so on."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, TypeVar, get_args

import pandas as pd


if TYPE_CHECKING:
    from ase.atoms import Atoms as AseAtoms
    from phonopy.phonon.dos import TotalDos
    from pymatgen.core import Composition, IMolecule, IStructure, Molecule, Structure
    from pymatgen.io.ase import MSONAtoms
    from pymatgen.phonon.dos import PhononDos

type Xyz = tuple[float, float, float]
# Quoted unions: PEP 695 treats a bare `"Name"` as Literal, not a forward ref
type AnyStructure = (
    "Structure | IStructure | Molecule | IMolecule | MSONAtoms | AseAtoms"
)
type AnyDos = "PhononDos | TotalDos"

# Bare Literal assignments (not `type`): get_args() is empty on PEP 695 type aliases
ColorElemTypeStrategy = Literal["symbol", "background", "both", "off"]
VALID_COLOR_ELEM_STRATEGIES = get_args(ColorElemTypeStrategy)

PTableSplitOrientation = Literal["diagonal", "horizontal", "vertical", "grid"]
PTABLE_SPLIT_ORIENTATIONS: tuple[PTableSplitOrientation, ...] = get_args(
    PTableSplitOrientation
)

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

T = TypeVar("T")  # generic type for input validation

SetMode = Literal["union", "intersection", "strict"]
SET_MODE = SET_UNION, SET_INTERSECTION, SET_STRICT = get_args(SetMode)

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
Corner = Literal["top-left", "top-right", "bottom-left", "bottom-right"]
VALID_CORNERS = TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT = get_args(Corner)
ShowCounts = Literal["value", "percent", "value+percent", False]
