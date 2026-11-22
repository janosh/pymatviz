"""Composition visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import traitlets as tl

from pymatviz.widgets._schema_traits import CompositionTraits
from pymatviz.widgets._traits import ELEMENT_COLOR_SCHEMES
from pymatviz.widgets.matterviz import MatterVizWidget


if TYPE_CHECKING:
    from pymatviz.typing import FormulaLike


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

    # Visualization options
    mode = tl.CaselessStrEnum(["pie", "bar", "bubble"], default_value="pie").tag(
        sync=True
    )
    show_percentages = tl.Bool(default_value=False).tag(sync=True)
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
        from pymatviz.composition import to_composition

        comp_dict = (
            None
            if composition is None
            else to_composition(composition, allow_negative=True).as_dict()
        )
        super().__init__(composition=comp_dict, **kwargs)
