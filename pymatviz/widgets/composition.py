"""Composition visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import traitlets as tl
from pymatgen.core import Composition

from pymatviz.widgets._schema_traits import CompositionTraits
from pymatviz.widgets._traits import ELEMENT_COLOR_SCHEMES
from pymatviz.widgets.matterviz import MatterVizWidget


if TYPE_CHECKING:
    from pymatgen.util.typing import CompositionLike


class CompositionWidget(CompositionTraits, MatterVizWidget):
    """MatterViz widget for visualizing compositions in Python notebooks.

    Examples:
        Basic usage:
        >>> import pymatviz as pmv
        >>> from pymatgen.core import Composition
        >>> comp = Composition("Fe2O3")
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

    # kwargs for the pymatgen Composition constructor. Python-side only (not
    # synced): the frontend has no such prop, it only affects parsing here.
    pymatgen_kwargs = tl.Dict(allow_none=True)

    def __init__(
        self, composition: CompositionLike | None = None, **kwargs: Any
    ) -> None:
        """Initialize the CompositionWidget.

        Args:
            composition: Composition data (pymatgen Composition object or dict)
            **kwargs: Additional widget properties
        """
        if composition is None:
            comp_dict = None
        else:
            comp_kwargs = dict(strict=True, allow_negative=True)
            # use get (not pop) so pymatgen_kwargs still reaches the traitlet
            comp_kwargs |= kwargs.get("pymatgen_kwargs") or {}
            comp_dict = Composition(composition, **comp_kwargs).as_dict()
        super().__init__(composition=comp_dict, **kwargs)
