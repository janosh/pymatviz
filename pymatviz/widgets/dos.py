"""Density of states visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets.matterviz import MatterVizWidget


class DosWidget(MatterVizWidget):
    """MatterViz widget for visualizing electronic and phonon density of states.

    Accepts pymatgen Dos, CompleteDos, or PhononDos objects, or a pre-computed dict.

    Examples:
        From a pymatgen CompleteDos:
        >>> from pymatviz import DosWidget
        >>> widget = DosWidget(dos=complete_dos)
        >>> widget

        With custom options:
        >>> widget = DosWidget(
        ...     dos=complete_dos,
        ...     sigma=0.05,
        ...     spin_mode="combined",
        ...     style="height: 500px;",
        ... )
    """

    dos = tl.Dict(allow_none=True).tag(sync=True)

    # Display options
    stack = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    sigma = tl.Float(allow_none=True, default_value=None).tag(sync=True)
    normalize = tl.Unicode(allow_none=True, default_value=None).tag(sync=True)
    orientation = tl.Unicode(allow_none=True, default_value=None).tag(sync=True)
    show_legend = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    spin_mode = tl.Unicode(allow_none=True, default_value=None).tag(sync=True)

    def __init__(self, dos: Any | None = None, **kwargs: Any) -> None:
        """Initialize the DosWidget.

        Args:
            dos: DOS data -- a pymatgen Dos, CompleteDos, PhononDos, or dict.
            **kwargs: Additional widget properties.
        """
        from pymatviz.widgets._normalize import _to_dict

        super().__init__(widget_type="dos", dos=_to_dict(dos, "DOS"), **kwargs)
