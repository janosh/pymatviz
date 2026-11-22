"""Plot X-ray diffraction patterns with the MatterViz XRD widget."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import kemio
import numpy as np
import pandas as pd

from pymatviz.structures import is_structure_like, to_structure
from pymatviz.widgets._normalize import normalize_xrd_pattern
from pymatviz.widgets.xrd import XrdWidget


if TYPE_CHECKING:
    from kemio.interop import XrdPatternLike

    from pymatviz.typing import AnyStructure


type PatternOrStruct = XrdPatternLike | AnyStructure
cu_k_alpha_wavelength = 1.54184  # Angstroms


def _to_pattern_dict(
    label: str, pattern_or_struct: Any, wavelength: float
) -> dict[str, Any]:
    """Pattern dict of a structure (computed with kemio) or pattern object, with
    its intensities checked to be present, finite and non-negative.
    """
    if is_structure_like(pattern_or_struct):
        struct = to_structure(pattern_or_struct)
        (pattern_or_struct,) = kemio.xrd_patterns([struct], wavelength=wavelength)
    expected = (
        f"{label!r}: expected a structure (kemio, pymatgen, ASE, phonopy), "
        f"XrdPattern or DiffractionPattern, got {type(pattern_or_struct).__name__}"
    )
    try:
        pattern = normalize_xrd_pattern(pattern_or_struct)
    except TypeError as exc:
        raise TypeError(expected) from exc
    if pattern is None:
        raise TypeError(expected)
    intensities = np.asarray(pattern.get("y", []), dtype=float)
    if len(intensities) == 0:
        raise ValueError(f"No intensities found in the diffraction pattern for {label}")
    if not np.isfinite(intensities).all() or np.any(intensities < 0):
        raise ValueError(f"Intensities for {label!r} must be finite and non-negative")
    return pattern


def xrd_plot(
    patterns: PatternOrStruct | Mapping[Any, PatternOrStruct] | pd.Series,
    *,
    wavelength: float = cu_k_alpha_wavelength,
    **kwargs: Any,
) -> XrdWidget:
    """Plot XRD patterns of structures (computed with kemio's xrd_patterns) or
    precomputed kemio XrdPatterns / pymatgen DiffractionPatterns.

    Patterns of several inputs are overlaid with a legend and scaled to a shared
    maximum of 100. MatterViz labels each pattern's 5 strongest peaks with their
    Miller indices (and 2θ angles for up to 2 patterns).

    Args:
        patterns (PatternOrStruct | Mapping[str, PatternOrStruct]): A single
            structure (kemio, pymatgen, ASE, phonopy; disordered sites scatter
            by occupancy), XrdPattern, DiffractionPattern or pattern dict (2θ and
            intensity arrays under x and y), or a mapping or pd.Series from legend
            labels to them.
        wavelength (float): X-ray wavelength in Å used to compute patterns of
            structures. Defaults to 1.54184 (Cu K-alpha).
        **kwargs: Passed to XrdWidget, e.g. ``style`` or ``controls_open``.

    Raises:
        ValueError: If patterns are empty, an intensity is negative or not finite,
            or no intensity is positive.
        TypeError: If patterns is not a structure or pattern, or a mapping of them.

    Returns:
        XrdWidget: The MatterViz XRD plot.
    """
    # a pattern dict (x/y keys holding arrays) is one pattern; labels named x and y
    # hold pattern objects
    is_pattern_dict = isinstance(patterns, Mapping) and (
        {"x", "y"} <= patterns.keys()
        and isinstance(patterns["x"], list | tuple | np.ndarray | pd.Series)
    )
    is_single = not isinstance(patterns, Mapping | pd.Series) or is_pattern_dict
    # (label, pattern) pairs: a Series (e.g. a df column) may repeat index labels
    labeled = [("XRD Pattern", patterns)] if is_single else list(patterns.items())
    if not labeled:
        raise ValueError("patterns must not be empty")
    computed = {
        str(label): _to_pattern_dict(str(label), value, wavelength)
        for label, value in labeled
    }
    if len(computed) != len(labeled):
        labels = [label for label, _ in labeled]
        raise ValueError(f"Labels must be unique as strings, got {labels}")
    if not any(np.any(np.asarray(pat["y"]) > 0) for pat in computed.values()):
        raise ValueError("At least one diffraction intensity must be positive")
    return XrdWidget(
        patterns=computed.popitem()[1] if is_single else computed, **kwargs
    )
