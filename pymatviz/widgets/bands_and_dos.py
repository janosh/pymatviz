"""Combined band structure and DOS visualization widget for Jupyter notebooks."""

from __future__ import annotations

import functools
from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import _to_dict, labeled_keys, structure_to_dict
from pymatviz.widgets._schema_traits import BandsAndDosTraits
from pymatviz.widgets._traits import (
    PlotControlsTraits,
    normalized_trait,
    optional_trait,
    validator,
)
from pymatviz.widgets.dos import StackedDosCheck
from pymatviz.widgets.matterviz import MatterVizWidget


class BandsAndDosWidget(
    StackedDosCheck, PlotControlsTraits, BandsAndDosTraits, MatterVizWidget
):
    """MatterViz widget for combined band structure + DOS visualization.

    Renders bands on the left and DOS on the right with linked energy/frequency axes.
    Accepts kemio, pymatgen and phonopy band structure and DOS objects, vasprun.xml
    paths, or pre-computed dicts. Pass
    ``{label: obj}`` dicts to overlay several band structures / DOS curves.

    ``doses`` labels must match ``band_structs``'. An optional ``structure``
    (pymatgen, ASE, PhonopyAtoms or structure dict) adds its Brillouin zone with
    the bands' k-path. ``fermi_level`` applies to both panels, ``show_legend``
    to the bands, ``stack``, ``sigma`` (THz for phonons, eV for electrons),
    ``normalize`` and ``spin_mode`` to the DOS. Mismatched labels, or
    ``stack=True`` with DOS on different grids, raise TraitError.

    Examples:
        From kemio phonon bands and DOS (NaCl demo data):
        >>> import pymatviz as pmv
        >>> from pymatviz.data import load_nacl_phonons
        >>> phonons = load_nacl_phonons()
        >>> bands, dos = phonons.bands(), phonons.total_dos([10, 10, 10])
        >>> widget = pmv.bands_and_dos(band_structs=bands, doses=dos)
        >>> widget

        With custom options:
        >>> widget = pmv.bands_and_dos(
        ...     band_structs=bands,
        ...     doses=dos,
        ...     style="height: 600px;",
        ... )
    """

    # converted on every assignment, labels checked by _check_stack_grids
    band_structs = normalized_trait(
        functools.partial(_to_dict, label="band structure", vasprun="bands")
    )
    doses = normalized_trait(functools.partial(_to_dict, label="DOS", vasprun="dos"))
    structure = normalized_trait(functools.partial(structure_to_dict, periodic=True))
    # DOS smearing width in the data unit: THz for phonons, eV for electrons
    sigma = optional_trait(tl.Float, min=0.0)

    _positional_traits = ("band_structs", "doses", "structure")

    @validator("band_structs", "doses", "stack")
    def _check_stack_grids(self, proposal: tl.Bunch) -> Any:
        """Also require band structures and DOS to share labels, checked whichever
        is set last (one validator per trait, so this extends StackedDosCheck's).
        """
        name, value = proposal["trait"].name, proposal["value"]
        if name != "stack":
            bands = value if name == "band_structs" else self.band_structs
            doses = value if name == "doses" else self.doses
            if bands is not None and doses is not None:
                band_keys, dos_keys = labeled_keys(bands), labeled_keys(doses)
                if band_keys != dos_keys:
                    raise ValueError(f"{band_keys=} and {dos_keys=} must be identical")
        if name == "band_structs":
            return value
        return super()._check_stack_grids(proposal)
