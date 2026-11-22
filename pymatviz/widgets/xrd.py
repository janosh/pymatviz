"""XRD pattern visualization widget for Jupyter notebooks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pymatviz.widgets._schema_traits import XrdTraits
from pymatviz.widgets._traits import PlotControlsTraits, normalized_trait
from pymatviz.widgets.matterviz import MatterVizWidget


def _normalize_patterns(patterns: Any) -> dict[str, Any] | None:
    """Pattern dict of one pattern, or ``{label: pattern dict}`` of a mapping of
    pattern objects or dicts (a dict of arrays is one pattern).

    Raises:
        ValueError: If a labeled mapping is empty.
    """
    import kemio
    from kemio.interop import has_class

    from pymatviz.widgets._normalize import normalize_xrd_pattern

    is_labeled = isinstance(patterns, Mapping) and all(
        isinstance(val, kemio.XrdPattern | Mapping)
        or has_class(val, "pymatgen.analysis.diffraction.core.DiffractionPattern")
        for val in patterns.values()
    )
    if is_labeled and not patterns:
        raise ValueError("XrdWidget patterns mapping must not be empty")
    if is_labeled:
        return {
            str(label): normalize_xrd_pattern(pattern)
            for label, pattern in patterns.items()
        }
    return normalize_xrd_pattern(patterns)


class XrdWidget(PlotControlsTraits, XrdTraits, MatterVizWidget):
    """MatterViz widget for visualizing X-ray diffraction patterns.

    Accepts a kemio XrdPattern, pymatgen DiffractionPattern or a dict with keys
    x (2-theta), y (intensity) and optional hkls/d_hkls/kind.

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

    patterns = normalized_trait(_normalize_patterns)

    def __init__(self, patterns: Any | None = None, **kwargs: Any) -> None:
        """Initialize the XrdWidget.

        Args:
            patterns: XRD pattern data -- a kemio XrdPattern, pymatgen
                DiffractionPattern, a dict with
                x (2-theta angles) and y (intensities) keys, or a ``{label: pattern}``
                mapping of those.
            **kwargs: Additional widget properties (e.g. ``controls_open`` to
                start with the control pane expanded).
        """
        super().__init__(patterns=patterns, **kwargs)
