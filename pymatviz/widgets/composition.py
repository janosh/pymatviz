"""Composition visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import traitlets as tl

from pymatviz.widgets._schema_traits import CompositionTraits
from pymatviz.widgets._traits import ELEMENT_COLOR_SCHEMES, JsonBool, normalized_trait
from pymatviz.widgets.matterviz import MatterVizWidget


if TYPE_CHECKING:
    from pymatviz.typing import FormulaLike


def _composition_dict(composition: FormulaLike) -> dict[str, float]:
    """``{element: amount}`` of a formula, mapping, Composition or structure."""
    from pymatviz.compositions import to_composition

    return to_composition(composition, allow_negative=True).as_dict()


class CompositionWidget(CompositionTraits, MatterVizWidget):
    """MatterViz widget for visualizing compositions in Python notebooks.

    Examples:
        Basic usage:
        >>> import pymatviz as pmv
        >>> import kemio
        >>> comp = kemio.Composition("Fe2O3")
        >>> widget = pmv.composition(composition=comp)
        >>> widget

        With custom visualization options:
        >>> widget = pmv.composition(
        ...     composition=comp,
        ...     mode="bar",
        ...     show_percentages=True,
        ...     color_scheme="Jmol",
        ...     style="width: 400px; margin: 20px auto;",
        ... )
    """

    composition = normalized_trait(_composition_dict)  # on every assignment
    # Visualization options
    mode = tl.CaselessStrEnum(["pie", "bar", "bubble"], default_value="pie").tag(
        sync=True
    )
    show_percentages = JsonBool(default_value=False).tag(sync=True)
    color_scheme = tl.CaselessStrEnum(ELEMENT_COLOR_SCHEMES, default_value="Jmol").tag(
        sync=True
    )

    def __init__(self, composition: FormulaLike | None = None, **kwargs: Any) -> None:
        """Initialize the CompositionWidget.

        Args:
            composition: A formula string, {element: amount} dict, kemio or
                pymatgen Composition, or structure (negative amounts allowed).
            **kwargs: Additional widget properties
        """
        super().__init__(composition=composition, **kwargs)
