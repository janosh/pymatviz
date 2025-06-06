"""Typing related: TypeAlias, generic types and so on."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, ParamSpec, TypeVar, Union, get_args

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from pymatgen.core import IStructure, Structure
from pymatgen.io.ase import MSONAtoms


if TYPE_CHECKING:
    from typing import TypeAlias

    from ase.atoms import Atoms as AseAtoms

AxOrFig: TypeAlias = plt.Axes | plt.Figure | go.Figure
Xyz: TypeAlias = tuple[float, float, float]
AnyStructure: TypeAlias = Union[Structure, IStructure, MSONAtoms, "AseAtoms"]

Backend: TypeAlias = Literal["matplotlib", "plotly"]
BACKENDS = MATPLOTLIB, PLOTLY = get_args(Backend)

ColorElemTypeStrategy: TypeAlias = Literal["symbol", "background", "both", "off"]
VALID_COLOR_ELEM_STRATEGIES = get_args(ColorElemTypeStrategy)

CrystalSystem: TypeAlias = Literal[
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
]

ElemValues: TypeAlias = dict[str | int, float] | pd.Series | Sequence[str]

T = TypeVar("T")  # generic type for input validation
P = ParamSpec("P")  # generic type for function parameters
R = TypeVar("R")  # generic type for return value

SetMode: TypeAlias = Literal["union", "intersection", "strict"]
SET_MODE = SET_UNION, SET_INTERSECTION, SET_STRICT = get_args(SetMode)


VALID_FIG_TYPES = get_args(AxOrFig)
VALID_FIG_NAMES: str = " | ".join(
    f"{t.__module__}.{t.__qualname__}" for t in VALID_FIG_TYPES
)


Rgb256ColorType: TypeAlias = tuple[int, int, int]  # 8-bit RGB

RgbColorType: TypeAlias = tuple[float, float, float] | str  # normalized to [0, 1]

RgbAColorType: TypeAlias = (  # normalized to [0, 1] with alpha
    str  # "none" or "#RRGGBBAA"/"#RGBA" hex strings
    | tuple[float, float, float, float]
    | tuple[RgbColorType, float]
    | tuple[tuple[float, float, float, float], float]
)
ColorType: TypeAlias = RgbColorType | RgbAColorType
FormulaGroupBy = Literal["formula", "reduced_formula", "chem_sys"]
Corner: TypeAlias = Literal["top-left", "top-right", "bottom-left", "bottom-right"]
VALID_CORNERS = TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT = get_args(Corner)
ShowCounts = Literal["value", "percent", "value+percent", False]
