"""Radial distribution function plot widget for Jupyter notebooks."""

from __future__ import annotations

import functools
from typing import Any

import traitlets as tl

from pymatviz.utils import str_labels
from pymatviz.widgets._normalize import normalize_plot_json, structure_to_dict
from pymatviz.widgets._schema_traits import RdfPlotTraits
from pymatviz.widgets._traits import (
    JsonBool,
    JsonFloat,
    JsonInt,
    PlotControlsTraits,
    normalized_trait,
    validator,
)
from pymatviz.widgets.matterviz import MatterVizWidget


def _rdf_structures(structures: Any) -> dict[str, Any] | list[Any] | None:
    """Structure dict of one structure (object or dict), or a list or
    ``{label: structure}`` dict of them as structure dicts. Labels that stringify
    alike (e.g. 1 and "1") raise.
    """
    if isinstance(structures, list | tuple):
        return [structure_to_dict(struct) for struct in structures]
    if isinstance(structures, dict) and "sites" not in structures:
        labeled = str_labels(structures)
        return {key: structure_to_dict(val) for key, val in labeled.items()}
    return structure_to_dict(structures)


class RdfPlotWidget(PlotControlsTraits, RdfPlotTraits, MatterVizWidget):
    """MatterViz widget for radial distribution function (RDF) plots.

    Accepts pre-computed RDF patterns or structure data for on-the-fly computation.
    Provide either ``structures`` or ``patterns``, not both.

    Examples:
        From a kemio Structure (or pymatgen Structure, ASE Atoms, structure dict):
        >>> import numpy as np
        >>> import kemio
        >>> nacl = kemio.Structure(
        ...     np.eye(3) * 5.64, ["Na", "Cl"], [[0] * 3, [0.5] * 3], fractional=True
        ... )
        >>> import pymatviz as pmv
        >>> widget = pmv.rdf_plot(structures=nacl, cutoff=10)
    """

    patterns = normalized_trait(
        functools.partial(normalize_plot_json, label="RdfPlot.patterns"),
        tl.Union([tl.Dict(), tl.List()]),
    )
    structures = normalized_trait(_rdf_structures, tl.Union([tl.Dict(), tl.List()]))
    mode = tl.CaselessStrEnum(
        ["element_pairs", "full"], default_value="element_pairs"
    ).tag(sync=True)
    show_reference_line = JsonBool(default_value=True).tag(sync=True)
    cutoff = JsonFloat(default_value=15).tag(sync=True)  # > 0, see _validate_cutoff
    n_bins = JsonInt(default_value=75, min=1).tag(sync=True)

    def __init__(
        self,
        structures: Any | None = None,
        *,
        patterns: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the RdfPlotWidget.

        Args:
            structures: A structure (kemio, pymatgen, ASE, phonopy or structure
                dict), or a list or ``{label: structure}`` dict of them, for
                on-the-fly RDF computation.
            patterns: Pre-computed RDF entries (list of
                ``{label, pattern: {r, g_r}}`` dicts).
            **kwargs: Additional widget properties.
        """
        super().__init__(structures=structures, patterns=patterns, **kwargs)

    @validator("structures", "patterns")
    def _validate_one_source(self, proposal: tl.Bunch) -> Any:
        """Reject setting structures and patterns both, on construction and
        assignment, so the state (and its saved JSON) stays loadable. Inside
        ``hold_trait_notifications()`` this checks the final state, so swapping
        sources there works.
        """
        other = "patterns" if proposal["trait"].name == "structures" else "structures"
        if proposal["value"] is not None and getattr(self, other) is not None:
            raise ValueError(
                "RdfPlotWidget accepts either 'structures' (for on-the-fly RDF "
                "computation) or 'patterns' (pre-computed), not both."
            )
        return proposal["value"]

    @validator("cutoff")
    def _validate_cutoff(self, proposal: tl.Bunch) -> float:
        """Reject non-positive cutoffs, which leave no radial range to bin."""
        if proposal["value"] <= 0:
            raise ValueError(
                f"RdfPlotWidget cutoff must be > 0, got {proposal['value']}"
            )
        return proposal["value"]
