"""Density of states visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import _to_dict, check_stack_grids
from pymatviz.widgets._schema_traits import DosTraits
from pymatviz.widgets._traits import (
    PlotControlsTraits,
    normalized_trait,
    optional_trait,
    validator,
)
from pymatviz.widgets.matterviz import MatterVizWidget


class StackedDosCheck(tl.HasTraits):
    """Stacked DOS (``stack=True``) need a shared grid, checked whichever of
    ``doses`` and ``stack`` is set last.
    """

    doses: Any  # traits of the widgets using this check
    stack: Any

    @validator("doses", "stack")
    def _check_stack_grids(self, proposal: tl.Bunch) -> Any:
        """Reject stacked DOS on different grids."""
        is_doses = proposal["trait"].name == "doses"
        doses = proposal["value"] if is_doses else self.doses
        if doses and (self.stack if is_doses else proposal["value"]):
            check_stack_grids(doses)
        return proposal["value"]


class DosWidget(StackedDosCheck, PlotControlsTraits, DosTraits, MatterVizWidget):
    """MatterViz widget for visualizing electronic and phonon density of states.

    Accepts kemio PhononDos and ElectronicDos, pymatgen Dos, CompleteDos and
    PhononDos, phonopy TotalDos and Phonopy objects, vasprun.xml paths, or a
    pre-computed dict. Pass a ``{label: dos}`` dict to overlay several DOS curves.
    Phonopy objects need ``run_total_dos()`` first. ``stack=True`` raises
    TraitError for DOS on different grids.

    Examples:
        From a kemio site-projected phonon DOS (NaCl demo data):
        >>> import pymatviz as pmv
        >>> from pymatviz.data import load_nacl_phonons
        >>> phonons = load_nacl_phonons()
        >>> widget = pmv.dos(doses=phonons.projected_dos([10, 10, 10]))
        >>> widget

        Several DOS curves overlaid, with custom options:
        >>> widget = pmv.dos(
        ...     doses={f"{n_k}^3": phonons.total_dos([n_k] * 3) for n_k in (4, 10)},
        ...     sigma=0.05,
        ...     style="height: 500px;",
        ... )
    """

    # Gaussian smearing width in the data unit: THz for phonon DOS (pymatgen PhononDos
    # frequencies), eV for electronic DOS. matterviz reads sigma in the data unit
    # whatever frequency unit the plot displays.
    sigma = optional_trait(tl.Float, min=0.0)
    doses = normalized_trait(lambda value: _to_dict(value, "DOS", vasprun="dos"))

    _positional_traits = ("doses",)
