"""Density of states visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import _to_dict
from pymatviz.widgets._traits import PlotControlsTraits
from pymatviz.widgets.matterviz import MatterVizWidget


class DosWidget(PlotControlsTraits, MatterVizWidget):
    """MatterViz widget for visualizing electronic and phonon density of states.

    Accepts pymatgen Dos, CompleteDos, or PhononDos objects, or a pre-computed dict.
    Pass a ``{label: dos}`` dict to overlay several DOS curves.

    Examples:
        From a pymatgen CompleteDos:
        >>> from pymatviz import DosWidget
        >>> widget = DosWidget(doses=complete_dos)
        >>> widget

        With custom options:
        >>> widget = DosWidget(
        ...     doses={"PBE": dos_pbe, "HSE": dos_hse},
        ...     sigma=0.05,
        ...     spin_mode="combined",
        ...     style="height: 500px;",
        ... )
    """

    # one DOS, or {label: dos} to overlay several
    doses = tl.Dict(allow_none=True).tag(sync=True)

    # Display options
    stack = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    sigma = tl.Float(allow_none=True, default_value=None).tag(sync=True)
    normalize = tl.Unicode(allow_none=True, default_value=None).tag(sync=True)
    orientation = tl.Unicode(allow_none=True, default_value=None).tag(sync=True)
    show_legend = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    spin_mode = tl.Unicode(allow_none=True, default_value=None).tag(sync=True)

    def __init__(self, doses: Any | None = None, **kwargs: Any) -> None:
        """Initialize the DosWidget.

        Args:
            doses: DOS data -- a pymatgen Dos, CompleteDos, PhononDos, a dict, or a
                ``{label: dos}`` dict of those to overlay.
            **kwargs: Additional widget properties.
        """
        super().__init__(widget_type="dos", doses=_to_dict(doses, "DOS"), **kwargs)
