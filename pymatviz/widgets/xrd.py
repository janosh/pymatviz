"""XRD pattern visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets.matterviz import MatterVizWidget


class XrdWidget(MatterVizWidget):
    """MatterViz widget for visualizing X-ray diffraction patterns.

    Accepts a pymatgen DiffractionPattern or a dict with either:
    - canonical keys: x (2-theta), y (intensity), optional hkls/d_hkls
    - Ferrox keys: two_theta, intensities, optional hkls/d_spacings

    Examples:
        From a pymatgen DiffractionPattern:
        >>> from pymatviz import XrdWidget
        >>> widget = XrdWidget(patterns=diffraction_pattern)
        >>> widget

        From a dict:
        >>> data = {"x": two_theta, "y": intensities, "hkls": hkl_list}
        >>> widget = XrdWidget(patterns=data)

        With custom options:
        >>> widget = XrdWidget(
        ...     patterns=diffraction_pattern,
        ...     style="height: 400px;",
        ... )
    """

    patterns = tl.Dict(allow_none=True).tag(sync=True)

    def __init__(self, patterns: Any | None = None, **kwargs: Any) -> None:
        """Initialize the XrdWidget.

        Args:
            patterns: XRD pattern data -- a pymatgen DiffractionPattern or dict with
                x (2-theta angles) and y (intensities) keys.
            **kwargs: Additional widget properties.
        """
        from pymatviz.widgets._normalize import normalize_xrd_pattern

        super().__init__(
            widget_type="xrd", patterns=normalize_xrd_pattern(patterns), **kwargs
        )
