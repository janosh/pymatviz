"""Combined band structure and DOS visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import _to_dict
from pymatviz.widgets._traits import PlotControlsTraits, optional_trait
from pymatviz.widgets.matterviz import MatterVizWidget


class BandsAndDosWidget(PlotControlsTraits, MatterVizWidget):
    """MatterViz widget for combined band structure + DOS visualization.

    Renders bands on the left and DOS on the right with linked energy/frequency axes.
    Accepts pymatgen band structure and DOS objects, or pre-computed dicts. Pass
    ``{label: obj}`` dicts to overlay several band structures / DOS curves.

    Examples:
        From pymatgen objects:
        >>> from pymatviz import BandsAndDosWidget
        >>> widget = BandsAndDosWidget(band_structs=bs, doses=dos)
        >>> widget

        With custom options:
        >>> widget = BandsAndDosWidget(
        ...     band_structs=bs,
        ...     doses=dos,
        ...     style="height: 600px;",
        ... )
    """

    band_structs = tl.Dict(allow_none=True).tag(sync=True)
    doses = tl.Dict(allow_none=True).tag(sync=True)

    # Config forwarded to the embedded Bands (band_type, show_legend) and Dos
    # (stack, sigma, normalize, spin_mode) panels; the control-pane traits reach both.
    # fermi_level/reference_frequency and dos orientation are controlled internally
    # by the combined view.
    band_type = optional_trait(tl.CaselessStrEnum, values=["phonon", "electronic"])
    show_legend = optional_trait(tl.Bool)
    stack = optional_trait(tl.Bool)
    sigma = optional_trait(tl.Float)
    normalize = optional_trait(tl.CaselessStrEnum, values=["max", "sum", "integral"])
    spin_mode = optional_trait(
        tl.CaselessStrEnum, values=["mirror", "overlay", "up_only", "down_only"]
    )

    def __init__(
        self,
        band_structs: Any | None = None,
        doses: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the BandsAndDosWidget.

        Args:
            band_structs: Band structure data -- a pymatgen BandStructure,
                BandStructureSymmLine, PhononBandStructureSymmLine, a dict, or a
                ``{label: band_structure}`` dict of those.
            doses: DOS data -- a pymatgen Dos, CompleteDos, PhononDos, a dict, or a
                ``{label: dos}`` dict of those.
            **kwargs: Additional widget properties, e.g. ``band_type`` and
                ``show_legend`` for the bands panel, or ``stack``, ``sigma``,
                ``normalize``, and ``spin_mode`` for the DOS panel.
        """
        super().__init__(
            widget_type="bands_and_dos",
            band_structs=_to_dict(band_structs, "band structure"),
            doses=_to_dict(doses, "DOS"),
            **kwargs,
        )
