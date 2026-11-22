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
)
from pymatviz.widgets.matterviz import MatterVizWidget


class DosWidget(PlotControlsTraits, DosTraits, MatterVizWidget):
    """MatterViz widget for visualizing electronic and phonon density of states.

    Accepts matscikit PhononDos and ElectronicDos, pymatgen Dos, CompleteDos and
    PhononDos, phonopy TotalDos and Phonopy objects, vasprun.xml paths, or a
    pre-computed dict. Pass a ``{label: dos}`` dict to overlay several DOS curves.

    Examples:
        From a pymatgen CompleteDos:
        >>> import pymatviz as pmv
        >>> widget = pmv.dos(doses=complete_dos)
        >>> widget

        With custom options:
        >>> widget = pmv.dos(
        ...     doses={"PBE": dos_pbe, "HSE": dos_hse},
        ...     sigma=0.05,
        ...     spin_mode="overlay",
        ...     style="height: 500px;",
        ... )
    """

    # Gaussian smearing width in the data unit: THz for phonon DOS (pymatgen PhononDos
    # frequencies), eV for electronic DOS. matterviz reads sigma in the data unit
    # whatever frequency unit the plot displays.
    sigma = optional_trait(tl.Float, min=0.0)
    doses = normalized_trait(lambda value: _to_dict(value, "DOS", vasprun="dos"))

    def __init__(self, doses: Any | None = None, **kwargs: Any) -> None:
        """Initialize the DosWidget.

        Args:
            doses: DOS data -- a matscikit PhononDos or ElectronicDos, pymatgen Dos,
                CompleteDos or PhononDos, phonopy TotalDos or Phonopy (after
                run_total_dos), a vasprun.xml path (read by matscikit), a dict, or a
                ``{label: dos}`` dict of those to overlay.
            **kwargs: Additional widget properties.

        Raises:
            ValueError: If ``stack=True`` and the DOS have different grids.
        """
        super().__init__(doses=doses, **kwargs)

    @tl.validate("doses", "stack")
    def _check_stack_grids(self, proposal: tl.Bunch) -> Any:
        """Stacked DOS need a shared grid, whichever of doses and stack is set last."""
        is_doses = proposal["trait"].name == "doses"
        doses = proposal["value"] if is_doses else self.doses
        if doses and (self.stack if is_doses else proposal["value"]):
            check_stack_grids(doses)
        return proposal["value"]
