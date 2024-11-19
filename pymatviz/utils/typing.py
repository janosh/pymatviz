"""Typing related: TypeAlias, ..."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, ParamSpec, TypeVar, get_args

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go


if TYPE_CHECKING:
    from typing import TypeAlias

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


Backend: TypeAlias = Literal["matplotlib", "plotly"]
BACKENDS = MATPLOTLIB, PLOTLY = get_args(Backend)

AxOrFig: TypeAlias = plt.Axes | plt.Figure | go.Figure

VALID_FIG_TYPES = get_args(AxOrFig)
VALID_FIG_NAMES: str = " | ".join(
    f"{t.__module__}.{t.__qualname__}" for t in VALID_FIG_TYPES
)

ColorElemTypeStrategy: TypeAlias = Literal["symbol", "background", "both", "off"]
VALID_COLOR_ELEM_STRATEGIES = get_args(ColorElemTypeStrategy)
