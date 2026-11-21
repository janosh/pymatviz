"""XRD pattern visualization widget for Jupyter notebooks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import traitlets as tl

from pymatviz.widgets._traits import PlotControlsTraits, optional_trait
from pymatviz.widgets.matterviz import MatterVizWidget


class XrdWidget(PlotControlsTraits, MatterVizWidget):
    """MatterViz widget for visualizing X-ray diffraction patterns.

    Accepts a pymatgen DiffractionPattern or a dict with either:
    - canonical keys: x (2-theta), y (intensity), optional hkls/d_hkls/kind
    - Ferrox keys: two_theta, intensities, optional hkls/d_spacings/kind

    Patterns without hkls are treated as measured scans (``kind="profile"``): drawn
    as a line and never broadened. Pass ``kind="sticks"`` to force bar rendering.
    Pass a ``{label: pattern}`` mapping to overlay several patterns with a legend.

    Examples:
        From a pymatgen DiffractionPattern:
        >>> from pymatviz.widgets import XrdWidget
        >>> widget = XrdWidget(patterns=diffraction_pattern)
        >>> widget

        From a dict:
        >>> data = {"x": two_theta, "y": intensities, "hkls": hkl_list}
        >>> widget = XrdWidget(patterns=data)

        Several overlaid patterns:
        >>> widget = XrdWidget(patterns={"computed": pattern, "measured": data})

        With custom options:
        >>> widget = XrdWidget(
        ...     patterns=diffraction_pattern,
        ...     style="height: 400px;",
        ... )
    """

    patterns = optional_trait(tl.Dict)

    def __init__(self, patterns: Any | None = None, **kwargs: Any) -> None:
        """Initialize the XrdWidget.

        Args:
            patterns: XRD pattern data -- a pymatgen DiffractionPattern, a dict with
                x (2-theta angles) and y (intensities) keys, or a ``{label: pattern}``
                mapping of those.
            **kwargs: Additional widget properties (e.g. ``controls_open`` to
                start with the control pane expanded).
        """
        from pymatgen.analysis.diffraction.xrd import DiffractionPattern

        from pymatviz.widgets._normalize import normalize_xrd_pattern

        # a mapping of patterns (DiffractionPatterns or dicts) is labeled, a dict of
        # arrays is one pattern
        is_labeled = isinstance(patterns, Mapping) and all(
            isinstance(val, DiffractionPattern | Mapping) for val in patterns.values()
        )
        if is_labeled and not patterns:
            raise ValueError("XrdWidget patterns mapping must not be empty")
        normalized = (
            {
                str(label): normalize_xrd_pattern(pattern)
                for label, pattern in patterns.items()
            }
            if is_labeled
            else normalize_xrd_pattern(patterns)
        )
        super().__init__(widget_type="xrd", patterns=normalized, **kwargs)
