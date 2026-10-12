"""Density of states visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import _to_dict
from pymatviz.widgets._traits import PlotControlsTraits, optional_trait
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
    stack = optional_trait(tl.Bool)
    # Gaussian smearing width in the data unit: THz for phonon DOS (pymatgen PhononDos
    # frequencies), eV for electronic DOS. matterviz reads sigma in the data unit
    # whatever frequency unit the plot displays.
    sigma = optional_trait(tl.Float, min=0.0)
    normalize = optional_trait(tl.Unicode)
    orientation = optional_trait(tl.Unicode)
    show_legend = optional_trait(tl.Bool)
    spin_mode = optional_trait(tl.Unicode)

    def __init__(self, doses: Any | None = None, **kwargs: Any) -> None:
        """Initialize the DosWidget.

        Args:
            doses: DOS data -- a pymatgen Dos, CompleteDos, PhononDos, a dict, or a
                ``{label: dos}`` dict of those to overlay.
            **kwargs: Additional widget properties.
        """
        super().__init__(widget_type="dos", doses=_to_dict(doses, "DOS"), **kwargs)
