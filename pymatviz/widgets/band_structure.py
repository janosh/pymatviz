"""Band structure visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets.matterviz import MatterVizWidget


class BandStructureWidget(MatterVizWidget):
    """MatterViz widget for visualizing electronic and phonon band structures.

    Accepts pymatgen BandStructure, BandStructureSymmLine, or
    PhononBandStructureSymmLine objects, or a pre-computed dict.

    Examples:
        From a pymatgen BandStructureSymmLine:
        >>> from pymatviz import BandStructureWidget
        >>> widget = BandStructureWidget(band_structure=bs)
        >>> widget

        With custom options:
        >>> widget = BandStructureWidget(
        ...     band_structure=bs,
        ...     show_legend=True,
        ...     fermi_level=0.0,
        ...     style="height: 500px;",
        ... )
    """

    band_structure = tl.Dict(allow_none=True).tag(sync=True)

    # Display options
    band_type = tl.Unicode(allow_none=True, default_value=None).tag(sync=True)
    show_legend = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    fermi_level = tl.Float(allow_none=True, default_value=None).tag(sync=True)
    reference_frequency = tl.Float(allow_none=True, default_value=None).tag(sync=True)

    def __init__(self, band_structure: Any | None = None, **kwargs: Any) -> None:
        """Initialize the BandStructureWidget.

        Args:
            band_structure: Band structure data -- a pymatgen BandStructure,
                BandStructureSymmLine, PhononBandStructureSymmLine, or dict.
            **kwargs: Additional widget properties.
        """
        from pymatviz.widgets._normalize import _to_dict

        bs_data = _to_dict(band_structure, "band structure")
        super().__init__(widget_type="band_structure", band_structure=bs_data, **kwargs)
