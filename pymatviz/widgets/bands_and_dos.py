"""Combined band structure and DOS visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

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
            **kwargs: Additional widget properties.
        """
        from pymatviz.widgets._normalize import _to_dict

        bs_data = _to_dict(band_structure, "band structure")
        dos_data = _to_dict(dos, "DOS")
        super().__init__(
            widget_type="bands_and_dos",
            band_structure=bs_data,
            dos=dos_data,
            **kwargs,
        )
