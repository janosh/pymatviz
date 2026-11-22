"""Band structure visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

from pymatviz.widgets._normalize import _to_dict
from pymatviz.widgets._schema_traits import BandStructureTraits
from pymatviz.widgets._traits import PlotControlsTraits, normalized_trait
from pymatviz.widgets.matterviz import MatterVizWidget


class BandStructureWidget(PlotControlsTraits, BandStructureTraits, MatterVizWidget):
    """MatterViz widget for visualizing electronic and phonon band structures.

    Accepts kemio PhononBands and ElectronicBands, pymatgen BandStructure,
    BandStructureSymmLine and PhononBandStructureSymmLine, phonopy BandStructure,
    vasprun.xml paths, or a pre-computed dict. Pass a ``{label: band_structure}``
    dict to overlay several band structures.

    Examples:
        From kemio phonon bands (NaCl demo data):
        >>> import pymatviz as pmv
        >>> from pymatviz.data import load_nacl_phonons
        >>> phonons = load_nacl_phonons()
        >>> bands = phonons.bands()
        >>> widget = pmv.band_structure(band_structs=bands)
        >>> widget

        Several band structures overlaid, with custom options:
        >>> widget = pmv.band_structure(
        ...     band_structs={"51 pts": bands, "11 pts": phonons.bands(n_points=11)},
        ...     show_legend=True,
        ...     fermi_level=0.0,
        ...     style="height: 500px;",
        ... )
    """

    band_structs = normalized_trait(
        lambda value: _to_dict(value, "band structure", vasprun="bands")
    )

    def __init__(self, band_structs: Any | None = None, **kwargs: Any) -> None:
        """Initialize the BandStructureWidget.

        Args:
            band_structs: Band structure data -- a kemio PhononBands or
                ElectronicBands, pymatgen BandStructure, BandStructureSymmLine or
                PhononBandStructureSymmLine, phonopy BandStructure, a vasprun.xml
                path (read by kemio), a dict, or a ``{label: band_structure}``
                dict of those to overlay.
            **kwargs: Additional widget properties.
        """
        super().__init__(band_structs=band_structs, **kwargs)
