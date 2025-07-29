"""Composition visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import traitlets as tl
from pymatgen.core import Composition

from pymatviz.widgets.matterviz import MatterVizWidget


if TYPE_CHECKING:
    from pymatgen.util.typing import CompositionLike

    from pymatviz.widgets import MattervizElementColorSchemes


class CompositionWidget(MatterVizWidget):
    """MatterViz widget for visualizing compositions in Python notebooks.

    Examples:
        Basic usage:
        >>> from pymatviz import CompositionWidget
        >>> from pymatgen.core import Composition
        >>> comp = Composition("Fe2O3")
        >>> widget = CompositionWidget(composition=comp)
        >>> widget

        With custom visualization options:
        >>> widget = CompositionWidget(
        ...     composition=comp,
        ...     mode="bar",
        ...     show_percentages=True,
        ...     color_scheme="Jmol",
        ...     style="width: 400px; margin: 20px auto;",
        ... )
    """

    composition = tl.Dict(allow_none=True).tag(sync=True)

    # Visualization options
    mode: Literal["pie", "bar", "bubble"] = tl.Unicode("pie").tag(sync=True)
    show_percentages = tl.Bool(default_value=False).tag(sync=True)
    color_scheme: MattervizElementColorSchemes = tl.Unicode("Jmol").tag(sync=True)

    # Widget styling
    style = tl.Unicode(allow_none=True).tag(sync=True)  # Custom CSS styles

    # PyMatGen composition kwargs
    pymatgen_kwargs = tl.Dict(allow_none=True).tag(sync=True)

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
            comp_kwargs |= kwargs.pop("pymatgen_kwargs", {})
            comp_dict = Composition(composition, **comp_kwargs).as_dict()
        super().__init__(composition=comp_dict, **kwargs)
