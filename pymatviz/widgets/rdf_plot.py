"""Radial distribution function plot widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import _to_dict, normalize_plot_json
from pymatviz.widgets.matterviz import MatterVizWidget


class RdfPlotWidget(MatterVizWidget):
    """MatterViz widget for radial distribution function (RDF) plots.

    Accepts pre-computed RDF patterns or structure data for on-the-fly computation.
    Provide either ``structures`` or ``patterns``, not both.

    Examples:
        From a pymatgen Structure dict:
        >>> from pymatviz import RdfPlotWidget
        >>> widget = RdfPlotWidget(
        ...     structures={"lattice": {...}, "sites": [...]}, cutoff=10
        ... )
    """

    patterns = tl.Any(allow_none=True).tag(sync=True)
    structures = tl.Any(allow_none=True).tag(sync=True)
    mode = tl.Unicode(default_value="element_pairs").tag(sync=True)
    show_reference_line = tl.Bool(default_value=True).tag(sync=True)
    cutoff = tl.Float(default_value=15).tag(sync=True)
    n_bins = tl.Int(default_value=75).tag(sync=True)
    x_axis = tl.Dict(allow_none=True).tag(sync=True)
    y_axis = tl.Dict(allow_none=True).tag(sync=True)

    def __init__(
        self,
        structures: Any | None = None,
        *,
        patterns: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the RdfPlotWidget.

        Args:
            structures: Structure data (pymatgen Structure dict, or list/dict of them)
                for on-the-fly RDF computation.
            patterns: Pre-computed RDF data (list of ``{r, g_r, label}`` dicts).
            **kwargs: Additional widget properties.
        """
        if structures is not None and not isinstance(structures, (list, dict)):
            structures = _to_dict(structures, "RDF structures")

        if structures is not None and patterns is not None:
            raise ValueError(
                "RdfPlotWidget accepts either 'structures' (for on-the-fly RDF "
                "computation) or 'patterns' (pre-computed), not both."
            )

        super().__init__(
            widget_type="rdf_plot",
            structures=normalize_plot_json(structures, "RdfPlot.structures"),
            patterns=normalize_plot_json(patterns, "RdfPlot.patterns"),
            **kwargs,
        )
