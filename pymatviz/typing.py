"""Typing related: TypeAlias, generic types and so on."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

import pandas as pd


if TYPE_CHECKING:
    from kemio.interop import CompositionLike, PhononDosLike, StructureLike

    from pymatviz.enums import ElemCountMode

# Quoted unions: PEP 695 treats a bare `"Name"` as Literal, not a forward ref
# structure-like inputs (kemio, pymatgen, ASE, phonopy), converted to kemio
# Structures by normalize_structures; kemio's interop types
# (periodic ones checked at runtime)
type AnyStructure = "StructureLike"
type PeriodicStructure = "StructureLike"
type AnyDos = "PhononDosLike"
# CompositionLike (kemio): formula strings, {element: amount} mappings (pymatgen
# Compositions among them) or kemio Compositions. FormulaLike adds chemical system
# strings ("Fe-O") and structures, as count_formulas takes them
type FormulaLike = "CompositionLike | AnyStructure"
# space group numbers, Hermann-Mauguin/Hall symbols or periodic structures
type SpacegroupLike = "int | float | str | PeriodicStructure"

# ElemCountMode values, so callers can pass plain strings (count_mode="occurrence")
ElemCountModeName = Literal[
    "composition", "fractional_composition", "reduced_composition", "occurrence"
]
type ElemCountModeLike = "ElemCountMode | ElemCountModeName"

type ElemValues = (
    Mapping[str, int | float]
    | Mapping[int, int | float]
    | pd.Series
    | Sequence["CompositionLike"]
)

type RgbColorType = tuple[float, float, float] | str  # normalized to [0, 1]
type RgbAColorType = (  # normalized to [0, 1] with alpha
    str  # "none" or "#RRGGBBAA"/"#RGBA" hex strings
    | tuple[float, float, float, float]
    | tuple[RgbColorType, float]
    | tuple[tuple[float, float, float, float], float]
)
type ColorType = RgbColorType | RgbAColorType
FormulaGroupBy = Literal["formula", "reduced_formula", "chem_sys"]
