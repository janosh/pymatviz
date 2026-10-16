"""Combined band structure and DOS visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import (
    _to_dict,
    normalize_json_kwargs,
    structure_to_dict,
)
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
    structure = optional_trait(tl.Dict)
    bz_props = optional_trait(tl.Dict)  # BrillouinZone props for the structure's panel

    # Shared electronic reference energy for both panels (None = dataset metadata)
    fermi_level = optional_trait(tl.Float)
    # Config forwarded to the embedded Bands (show_legend) and Dos (stack, sigma,
    # normalize, spin_mode) panels; the control-pane traits reach both. Hover
    # frequency and DOS orientation are controlled internally by the combined view.
    show_legend = optional_trait(tl.Bool)
    stack = optional_trait(tl.Bool)
    # DOS smearing width in the data unit: THz for phonons, eV for electrons
    sigma = optional_trait(tl.Float, min=0.0)
    normalize = optional_trait(tl.CaselessStrEnum, values=["max", "sum", "integral"])
    spin_mode = optional_trait(
        tl.CaselessStrEnum, values=["mirror", "overlay", "up_only", "down_only"]
    )

    def __init__(
        self,
        band_structs: Any | None = None,
        doses: Any | None = None,
        structure: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the BandsAndDosWidget.

        Args:
            band_structs: Band structure data -- a pymatgen BandStructure,
                BandStructureSymmLine, PhononBandStructureSymmLine, a dict, or a
                ``{label: band_structure}`` dict of those.
            doses: DOS data -- a pymatgen Dos, CompleteDos, PhononDos, a dict, or a
                ``{label: dos}`` dict of those.
            structure: Optional crystal (pymatgen Structure, ASE Atoms, PhonopyAtoms
                or structure dict) adding its Brillouin zone with the bands' k-path.
            **kwargs: Additional widget properties, e.g. ``fermi_level`` for both
                panels, ``show_legend`` for the bands panel, or ``stack``, ``sigma``
                (THz for phonons, eV for electrons), ``normalize``, and ``spin_mode``
                for the DOS panel.
        """
        normalize_json_kwargs(kwargs, ("bz_props",), "BandsAndDos")
        super().__init__(
            widget_type="bands_and_dos",
            band_structs=_to_dict(band_structs, "band structure"),
            doses=_to_dict(doses, "DOS"),
            structure=structure_to_dict(structure, periodic=True),
            **kwargs,
        )
