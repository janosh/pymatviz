"""Radial distribution function plot widget for Jupyter notebooks."""

from __future__ import annotations

import functools
from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import normalize_plot_json, structure_to_dict
from pymatviz.widgets._schema_traits import RdfPlotTraits
from pymatviz.widgets._traits import PlotControlsTraits, normalized_trait
from pymatviz.widgets.matterviz import MatterVizWidget


def _rdf_structures(structures: Any) -> dict[str, Any] | list[Any]:
    """Structure dict of one structure (object or dict), or a list or
    ``{label: structure}`` dict of them as structure dicts.
    """
    if isinstance(structures, list | tuple):
        converted: Any = [structure_to_dict(struct) for struct in structures]
    elif isinstance(structures, dict) and "sites" not in structures:
        converted = {
            str(key): structure_to_dict(val) for key, val in structures.items()
        }
    else:
        converted = structure_to_dict(structures)
    return normalize_plot_json(converted, "RdfPlot.structures")


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
    show_reference_line = tl.Bool(default_value=True).tag(sync=True)
    cutoff = tl.Float(default_value=15).tag(sync=True)
    n_bins = tl.Int(default_value=75).tag(sync=True)

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
        if structures is not None and patterns is not None:
            raise ValueError(
                "RdfPlotWidget accepts either 'structures' (for on-the-fly RDF "
                "computation) or 'patterns' (pre-computed), not both."
            )

        super().__init__(structures=structures, patterns=patterns, **kwargs)
