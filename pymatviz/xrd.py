"""Plot X-ray diffraction patterns with the MatterViz XRD widget."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import kemio
import numpy as np
import pandas as pd

from pymatviz.structures import is_structure_like, to_structure
from pymatviz.utils import str_labels
from pymatviz.widgets._normalize import normalize_xrd_pattern
from pymatviz.widgets.xrd import XrdWidget


if TYPE_CHECKING:
    from kemio.interop import XrdPatternLike

    from pymatviz.typing import AnyStructure


type PatternOrStruct = XrdPatternLike | AnyStructure
cu_k_alpha_wavelength = 1.54184  # Angstroms


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
        ValueError: If patterns are empty, labels collide as strings, an intensity
            is negative or not finite, or no intensity is positive.
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
    labeled = str_labels({"XRD Pattern": patterns} if is_single else patterns)
    if not labeled:
        raise ValueError("patterns must not be empty")
    # patterns of all structures in one (parallel) kemio call
    structs = {
        label: to_structure(val)
        for label, val in labeled.items()
        if is_structure_like(val)
    }
    if structs:
        xrd_patterns = kemio.xrd_patterns([*structs.values()], wavelength=wavelength)
        labeled |= dict(zip(structs, xrd_patterns, strict=True))
    computed: dict[str, dict[str, Any]] = {}
    for label, value in labeled.items():
        expected = (
            f"{label!r}: expected a structure (kemio, pymatgen, ASE, phonopy), "
            f"XrdPattern or DiffractionPattern, got {type(value).__name__}"
        )
        try:  # non-finite intensities raise here
            pattern = normalize_xrd_pattern(value)
        except TypeError as exc:
            raise TypeError(expected) from exc
        if pattern is None:
            raise TypeError(expected)
        intensities = np.asarray(pattern["y"], dtype=float)
        if len(intensities) == 0:
            raise ValueError(f"No intensities found in the diffraction pattern {label}")
        if not np.isfinite(intensities).all() or np.any(intensities < 0):
            raise ValueError(
                f"Intensities for {label!r} must be finite and non-negative"
            )
        computed[label] = pattern
    if not any(np.any(np.asarray(pat["y"]) > 0) for pat in computed.values()):
        raise ValueError("At least one diffraction intensity must be positive")
    return XrdWidget(
        patterns=computed.popitem()[1] if is_single else computed, **kwargs
    )
