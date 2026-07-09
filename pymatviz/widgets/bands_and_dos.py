"""Combined band structure and DOS visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._traits import optional_trait
from pymatviz.widgets.matterviz import MatterVizWidget


class BandsAndDosWidget(MatterVizWidget):
    """MatterViz widget for combined band structure + DOS visualization.

    Renders bands on the left and DOS on the right with linked energy/frequency axes.
    Accepts pymatgen band structure and DOS objects, or pre-computed dicts.

    Examples:
        From pymatgen objects:
        >>> from pymatviz import BandsAndDosWidget
        >>> widget = BandsAndDosWidget(band_structure=bs, dos=dos)
        >>> widget

        With custom options:
        >>> widget = BandsAndDosWidget(
        ...     band_structure=bs,
        ...     dos=dos,
        ...     style="height: 600px;",
        ... )
    """

    band_structure = tl.Dict(allow_none=True).tag(sync=True)
    dos = tl.Dict(allow_none=True).tag(sync=True)

    # Config forwarded to the embedded Bands (band_type, show_legend) and Dos
    # (stack, sigma, normalize, spin_mode) panels. fermi_level/reference_frequency
    # and dos orientation are controlled internally by the combined view.
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
        band_structure: Any | None = None,
        dos: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the BandsAndDosWidget.

        Args:
            band_structure: Band structure data -- a pymatgen BandStructure,
                BandStructureSymmLine, PhononBandStructureSymmLine, or dict.
            dos: DOS data -- a pymatgen Dos, CompleteDos, PhononDos, or dict.
            **kwargs: Additional widget properties, e.g. ``band_type`` and
                ``show_legend`` for the bands panel, or ``stack``, ``sigma``,
                ``normalize``, and ``spin_mode`` for the DOS panel.
        """
        from pymatviz.widgets._normalize import _to_dict

        super().__init__(
            widget_type="bands_and_dos",
            band_structure=_to_dict(band_structure, "band structure"),
            dos=_to_dict(dos, "DOS"),
            **kwargs,
        )
