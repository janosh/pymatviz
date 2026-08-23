"""Band structure visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import _to_dict
from pymatviz.widgets._traits import PlotControlsTraits
from pymatviz.widgets.matterviz import MatterVizWidget


class BandStructureWidget(PlotControlsTraits, MatterVizWidget):
    """MatterViz widget for visualizing electronic and phonon band structures.

    Accepts pymatgen BandStructure, BandStructureSymmLine, or
    PhononBandStructureSymmLine objects, or a pre-computed dict. Pass a
    ``{label: band_structure}`` dict to overlay several band structures.

    Examples:
        From a pymatgen BandStructureSymmLine:
        >>> from pymatviz import BandStructureWidget
        >>> widget = BandStructureWidget(band_structs=bs)
        >>> widget

        With custom options:
        >>> widget = BandStructureWidget(
        ...     band_structs={"PBE": bs_pbe, "HSE": bs_hse},
        ...     show_legend=True,
        ...     fermi_level=0.0,
        ...     style="height: 500px;",
        ... )
    """

    # one band structure, or {label: band structure} to overlay several
    band_structs = tl.Dict(allow_none=True).tag(sync=True)

    # Display options
    band_type = tl.Unicode(allow_none=True, default_value=None).tag(sync=True)
    show_legend = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    fermi_level = tl.Float(allow_none=True, default_value=None).tag(sync=True)
    reference_frequency = tl.Float(allow_none=True, default_value=None).tag(sync=True)

    def __init__(self, band_structs: Any | None = None, **kwargs: Any) -> None:
        """Initialize the BandStructureWidget.

        Args:
            band_structs: Band structure data -- a pymatgen BandStructure,
                BandStructureSymmLine, PhononBandStructureSymmLine, a dict, or a
                ``{label: band_structure}`` dict of those to overlay.
            **kwargs: Additional widget properties.
        """
        super().__init__(
            widget_type="band_structure",
            band_structs=_to_dict(band_structs, "band structure"),
            **kwargs,
        )
